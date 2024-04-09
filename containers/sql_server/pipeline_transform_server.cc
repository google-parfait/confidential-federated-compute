// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "containers/sql_server/pipeline_transform_server.h"

#include <memory>
#include <optional>
#include <string>

#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/status/status.h"
#include "containers/crypto.h"
#include "containers/sql_server/sqlite_adapter.h"
#include "fcp/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "fcp/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/status_converters.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"
#include "fcp/protos/confidentialcompute/sql_query.pb.h"
#include "google/protobuf/repeated_ptr_field.h"

namespace confidential_federated_compute::sql_server {

using ::fcp::aggregation::CheckpointBuilder;
using ::fcp::aggregation::CheckpointParser;
using ::fcp::aggregation::DataType;
using ::fcp::aggregation::FederatedComputeCheckpointBuilderFactory;
using ::fcp::aggregation::FederatedComputeCheckpointParserFactory;
using ::fcp::aggregation::Tensor;
using ::fcp::base::ToGrpcStatus;
using ::fcp::confidentialcompute::ColumnSchema;
using ::fcp::confidentialcompute::ConfigureAndAttestRequest;
using ::fcp::confidentialcompute::ConfigureAndAttestResponse;
using ::fcp::confidentialcompute::PipelineTransform;
using ::fcp::confidentialcompute::Record;
using ::fcp::confidentialcompute::SqlQuery;
using ::fcp::confidentialcompute::TableSchema;
using ::fcp::confidentialcompute::TransformRequest;
using ::fcp::confidentialcompute::TransformResponse;
using ::google::protobuf::RepeatedPtrField;
using ::grpc::ServerContext;

namespace {

absl::StatusOr<std::vector<TensorColumn>> Deserialize(
    absl::Cord serialized_data, const TableSchema& table_schema) {
  std::vector<TensorColumn> columns(table_schema.column_size());
  FederatedComputeCheckpointParserFactory parser_factory;
  FCP_ASSIGN_OR_RETURN(std::unique_ptr<CheckpointParser> parser,
                       parser_factory.Create(serialized_data));
  std::optional<int> num_rows;
  for (int i = 0; i < table_schema.column_size(); i++) {
    FCP_ASSIGN_OR_RETURN(Tensor tensor_column_values,
                         parser->GetTensor(table_schema.column(i).name()));
    if (!num_rows.has_value()) {
      num_rows.emplace(tensor_column_values.num_elements());
    } else if (num_rows.value() != tensor_column_values.num_elements()) {
      return absl::InvalidArgumentError(
          "Record has columns with differing numbers of rows.");
    }
    FCP_ASSIGN_OR_RETURN(TensorColumn tensor_column,
                         TensorColumn::Create(table_schema.column(i),
                                              std::move(tensor_column_values)));
    columns[i] = std::move(tensor_column);
  }
  return columns;
}

absl::StatusOr<absl::Cord> Serialize(std::vector<TensorColumn> columns) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> ckpt_builder = builder_factory.Create();

  for (auto& column : columns) {
    FCP_RETURN_IF_ERROR(
        ckpt_builder->Add(column.column_schema_.name(), column.tensor_));
  }

  return ckpt_builder->Build();
}

}  // namespace

absl::Status SqlPipelineTransform::SqlConfigureAndAttest(
    const ConfigureAndAttestRequest* request,
    ConfigureAndAttestResponse* response) {
  SqlQuery sql_query;
  request->configuration().UnpackTo(&sql_query);
  if (sql_query.database_schema().table_size() != 1) {
    return absl::InvalidArgumentError(
        "SQL query input or output schema does not contain exactly "
        "one table schema.");
  }
  if (sql_query.database_schema().table(0).column_size() == 0) {
    return absl::InvalidArgumentError("SQL query input schema has no columns.");
  }
  const RecordDecryptor* record_decryptor;
  {
    absl::MutexLock l(&mutex_);
    if (configuration_ != std::nullopt) {
      return absl::FailedPreconditionError(
          "ConfigureAndAttest can only be called once.");
    }

    configuration_.emplace(
        SqlConfiguration{.query = sql_query.raw_sql(),
                         .input_schema = sql_query.database_schema().table(0),
                         .output_columns = sql_query.output_columns()});
    record_decryptor_.emplace(crypto_stub_);

    // Since record_decryptor_ is set once in ConfigureAndAttest and never
    // modified, and the underlying object is threadsafe, it is safe to store a
    // local pointer to it and access the object without a lock after we check
    // under the mutex that a value has been set for the std::optional wrapper.
    record_decryptor = &*record_decryptor_;
  }

  FCP_ASSIGN_OR_RETURN(*response->mutable_public_key(),
                       record_decryptor->GetPublicKey());
  return absl::OkStatus();
}

absl::Status SqlPipelineTransform::SqlGenerateNonces(
    const fcp::confidentialcompute::GenerateNoncesRequest* request,
    fcp::confidentialcompute::GenerateNoncesResponse* response) {
  RecordDecryptor* record_decryptor;
  {
    absl::MutexLock l(&mutex_);
    if (record_decryptor_ == std::nullopt) {
      return absl::FailedPreconditionError(
          "ConfigureAndAttest must be called before GenerateNonces.");
    }
    // Since record_decryptor_ is set once in ConfigureAndAttest and never
    // modified, and the underlying object is threadsafe, it is safe to store a
    // local pointer to it and access the object without a lock after we check
    // under the mutex that a value has been set for the std::optional wrapper.
    record_decryptor = &*record_decryptor_;
  }
  for (int i = 0; i < request->nonces_count(); ++i) {
    FCP_ASSIGN_OR_RETURN(std::string nonce, record_decryptor->GenerateNonce());
    response->add_nonces(std::move(nonce));
  }
  return absl::OkStatus();
}

absl::Status SqlPipelineTransform::SqlTransform(const TransformRequest* request,
                                                TransformResponse* response) {
  if (request->inputs_size() != 1) {
    return absl::InvalidArgumentError(
        "Transform requires exactly one `Record` per request.");
  }
  RecordDecryptor* record_decryptor;
  const SqlConfiguration* configuration;
  {
    absl::MutexLock l(&mutex_);
    if (configuration_ == std::nullopt || record_decryptor_ == std::nullopt) {
      return absl::FailedPreconditionError(
          "ConfigureAndAttest must be called before Transform.");
    }

    // Since record_decryptor_ and configuration_ are set once in
    // ConfigureAndAttest and never modified, and both underlying objects are
    // threadsafe, it is safe to store a local pointer to them and access the
    // objects without a lock after we check under the mutex that values have
    // been set for the std::optional wrappers.
    record_decryptor = &*record_decryptor_;
    configuration = &*configuration_;
  }

  FCP_ASSIGN_OR_RETURN(std::unique_ptr<SqliteAdapter> sqlite,
                       SqliteAdapter::Create());
  FCP_RETURN_IF_ERROR(sqlite->DefineTable(configuration->input_schema));
  const Record& record = request->inputs(0);
  FCP_ASSIGN_OR_RETURN(std::string unencrypted_data,
                       record_decryptor->DecryptRecord(record));
  FCP_ASSIGN_OR_RETURN(
      std::vector<TensorColumn> contents,
      Deserialize(absl::Cord(unencrypted_data), configuration->input_schema));
  if (contents.size() > 0) {
    int num_rows = contents.at(0).tensor_.num_elements();
    FCP_RETURN_IF_ERROR(
        sqlite->AddTableContents(std::move(contents), num_rows));
  }

  FCP_ASSIGN_OR_RETURN(std::vector<TensorColumn> result,
                       sqlite->EvaluateQuery(configuration->query,
                                             configuration->output_columns));

  FCP_ASSIGN_OR_RETURN(absl::Cord output_data, Serialize(std::move(result)));
  // Protobuf version 23.0 is required to use [ctype = CORD], however, we can't
  // use this since it isn't currently compatible with TensorFlow.
  std::string ckpt_string;
  absl::CopyCordToString(output_data, &ckpt_string);
  Record* output = response->add_outputs();
  output->set_unencrypted_data(std::move(ckpt_string));
  output->set_compression_type(Record::COMPRESSION_TYPE_NONE);

  return absl::OkStatus();
}

SqlPipelineTransform::SqlPipelineTransform(
    oak::containers::v1::OrchestratorCrypto::StubInterface* crypto_stub)
    : crypto_stub_(*ABSL_DIE_IF_NULL(crypto_stub)) {
  CHECK_OK(SqliteAdapter::Initialize());
}

SqlPipelineTransform::~SqlPipelineTransform() { SqliteAdapter::ShutDown(); };

grpc::Status SqlPipelineTransform::ConfigureAndAttest(
    ServerContext* context, const ConfigureAndAttestRequest* request,
    ConfigureAndAttestResponse* response) {
  return ToGrpcStatus(SqlConfigureAndAttest(request, response));
}

grpc::Status SqlPipelineTransform::GenerateNonces(
    grpc::ServerContext* context,
    const fcp::confidentialcompute::GenerateNoncesRequest* request,
    fcp::confidentialcompute::GenerateNoncesResponse* response) {
  return ToGrpcStatus(SqlGenerateNonces(request, response));
}

grpc::Status SqlPipelineTransform::Transform(ServerContext* context,
                                             const TransformRequest* request,
                                             TransformResponse* response) {
  return ToGrpcStatus(SqlTransform(request, response));
}

}  // namespace confidential_federated_compute::sql_server
