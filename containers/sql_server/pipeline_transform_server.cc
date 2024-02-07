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

#include <stdio.h>

#include <memory>
#include <optional>
#include <string>

#include "absl/log/log.h"
#include "containers/crypto.h"
#include "containers/sql_server/sql_data.pb.h"
#include "containers/sql_server/sql_data_converter.h"
#include "fcp/base/status_converters.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"
#include "fcp/protos/confidentialcompute/sql_query.pb.h"

namespace confidential_federated_compute::sql_server {

using ::confidential_federated_compute::RecordDecryptor;
using ::fcp::base::ToGrpcStatus;
using ::fcp::confidentialcompute::ConfigureAndAttestRequest;
using ::fcp::confidentialcompute::ConfigureAndAttestResponse;
using ::fcp::confidentialcompute::PipelineTransform;
using ::fcp::confidentialcompute::Record;
using ::fcp::confidentialcompute::TransformRequest;
using ::fcp::confidentialcompute::TransformResponse;
using ::grpc::ServerContext;
using ::sql_data::SqlData;
using ::fcp::confidentialcompute::SqlQuery;

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
    record_decryptor_.emplace(request->configuration());

    // Since record_decryptor_ is set once in ConfigureAndAttest and never
    // modified, and the underlying object is threadsafe, it is safe to store a
    // local pointer to it and access the object without a lock after we check
    // under the mutex that a value has been set for the std::optional wrapper.
    record_decryptor = &*record_decryptor_;
  }

  FCP_ASSIGN_OR_RETURN(const PublicKeyAndSignature* public_key_and_signature,
                       record_decryptor->GetPublicKeyAndSignature());
  response->set_public_key(public_key_and_signature->public_key);
  // TODO(nfallen): Set the signature on the ConfigureAndAttestResponse once a
  // signature rooted in the attestation evidence is available.
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

  sql_data::SqlData sql_input_data;
  for (const fcp::confidentialcompute::Record& record : request->inputs()) {
    FCP_ASSIGN_OR_RETURN(std::string unencrypted_data,
                         record_decryptor->DecryptRecord(record));
    FCP_RETURN_IF_ERROR(AddWireFormatDataToSqlData(
        unencrypted_data, configuration->input_schema, sql_input_data));
  }

  SqlData result;

  FCP_ASSIGN_OR_RETURN(std::unique_ptr<SqliteAdapter> sqlite,
                       SqliteAdapter::Create());
  FCP_RETURN_IF_ERROR(
      sqlite->DefineTable(configuration->input_schema.create_table_sql()));

  FCP_RETURN_IF_ERROR(
      sqlite->SetTableContents(configuration->input_schema, sql_input_data));

  FCP_ASSIGN_OR_RETURN(result,
                       sqlite->EvaluateQuery(configuration->query,
                                             configuration->output_columns));

  FCP_ASSIGN_OR_RETURN(std::string output_data,
                       ConvertSqlDataToWireFormat(result));
  Record* output = response->add_outputs();
  output->set_unencrypted_data(std::move(output_data));

  return absl::OkStatus();
}

SqlPipelineTransform::SqlPipelineTransform() {
  FCP_CHECK_STATUS(SqliteAdapter::Initialize());
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
