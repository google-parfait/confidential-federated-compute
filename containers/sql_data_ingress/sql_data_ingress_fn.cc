// Copyright 2026 Google LLC.
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

#include "containers/sql_data_ingress/sql_data_ingress_fn.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "containers/common/input.h"
#include "containers/common/row_set.h"
#include "containers/common/sqlite_adapter.h"
#include "containers/fns/do_fn.h"
#include "containers/fns/fn.h"
#include "containers/fns/fn_factory.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/message_description.pb.h"
#include "fcp/protos/confidentialcompute/sql_data_ingress_config.pb.h"
#include "fcp/protos/confidentialcompute/sql_query.pb.h"
#include "google/protobuf/descriptor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"

namespace confidential_federated_compute::sql_data_ingress {

namespace {

using ::fcp::confidentialcompute::MessageDescription;
using ::fcp::confidentialcompute::
    SqlDataIngressContainerInitializeConfiguration;
using ::fcp::confidentialcompute::SqlQuery;
using ::google::protobuf::FileDescriptorSet;
using ::tensorflow_federated::aggregation::CheckpointBuilder;
using ::tensorflow_federated::aggregation::CheckpointParser;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::Tensor;

class SqlDataIngressDoFn : public fns::DoFn {
 public:
  explicit SqlDataIngressDoFn(SqlConfiguration sql_configuration,
                              std::shared_ptr<MessageFactory> message_factory,
                              std::string on_device_query_name)
      : sql_configuration_(std::move(sql_configuration)),
        message_factory_(std::move(message_factory)),
        on_device_query_name_(std::move(on_device_query_name)) {}
  absl::Status Do(Session::KV input, Context& context) override;

 private:
  SqlConfiguration sql_configuration_;
  std::shared_ptr<MessageFactory> message_factory_;
  std::string on_device_query_name_;
};

class SqlDataIngressFnFactory : public fns::FnFactory {
 public:
  explicit SqlDataIngressFnFactory(
      SqlConfiguration sql_configuration,
      std::shared_ptr<MessageFactory> message_factory,
      std::string on_device_query_name)
      : sql_configuration_(std::move(sql_configuration)),
        message_factory_(std::move(message_factory)),
        on_device_query_name_(std::move(on_device_query_name)) {}
  absl::StatusOr<std::unique_ptr<fns::Fn>> CreateFn() const override {
    return std::make_unique<SqlDataIngressDoFn>(
        sql_configuration_, message_factory_, on_device_query_name_);
  }

 private:
  SqlConfiguration sql_configuration_;
  std::shared_ptr<MessageFactory> message_factory_;
  std::string on_device_query_name_;
};

}  // namespace

// This function executes the SQL query on the input data, and emits the result
// as an encrypted checkpoint. The output checkpoint will contain one tensor
// for each output column, using the output column's name.
//
// For example, if the SQL query is:
// SELECT text AS result FROM input
// and the input data is:
// | text     |
// |----------|
// | example1 |
// | example2 |
// | example3 |
// then the output checkpoint will contain one string tensor named "result" with
// the values ["example1", "example2", "example3"].
//
// Note that this function hasn't supported executing SQL queries at DP unit
// level. If a single incoming input have data for multiple DP units and the SQL
// query contains GROUP BY aggregation, the result could be incorrect.
absl::Status SqlDataIngressDoFn::Do(KV input, Context& context) {
  if (input.blob_id.empty()) {
    return absl::InvalidArgumentError("Missing input blob id.");
  }

  FederatedComputeCheckpointParserFactory parser_factory;
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<CheckpointParser> parser,
      parser_factory.Create(absl::Cord(std::move(input.data))));

  ABSL_ASSIGN_OR_RETURN(auto tensor_map, parser->LoadAllTensors());
  std::vector<Tensor> tensors;
  tensors.reserve(sql_configuration_.input_schema.column_size());
  for (const auto& col : sql_configuration_.input_schema.column()) {
    auto it = tensor_map.find(col.name());
    if (it == tensor_map.end()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Missing required column in input: ", col.name()));
    }
    tensors.push_back(std::move(it->second));
  }

  ABSL_ASSIGN_OR_RETURN(Input sql_input,
                        Input::CreateFromTensors(std::move(tensors)));

  ABSL_ASSIGN_OR_RETURN(RowSet row_set, RowSet::Create(&sql_input));
  ABSL_ASSIGN_OR_RETURN(
      std::vector<Tensor> sql_result,
      SqliteAdapter::ExecuteQuery(sql_configuration_, row_set));

  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> builder = builder_factory.Create();

  for (auto& tensor : sql_result) {
    std::string name = tensor.name();
    ABSL_RETURN_IF_ERROR(builder->Add(name, std::move(tensor)));
  }
  ABSL_ASSIGN_OR_RETURN(absl::Cord checkpoint, builder->Build());

  if (!context.EmitEncrypted(
          /*reencryption_key_index=*/0,
          KV(std::move(input.key), std::string(std::move(checkpoint)),
             std::move(input.blob_id)))) {
    return absl::InvalidArgumentError("Failed to emit encrypted checkpoint.");
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<fns::FnFactory>> ProvideSqlDataIngressFnFactory(
    const google::protobuf::Any& configuration,
    const google::protobuf::Any& config_constraints,
    const confidential_federated_compute::fns::WriteConfigurationMap&
        write_configuration_map) {
  SqlDataIngressContainerInitializeConfiguration config;
  if (!configuration.UnpackTo(&config)) {
    return absl::InvalidArgumentError(
        "SqlDataIngressContainerInitializeConfiguration cannot be unpacked.");
  }
  SqlQuery sql_query = config.sql_query();
  if (sql_query.database_schema().table_size() != 1) {
    return absl::InvalidArgumentError(
        "SQL query input schema does not contain exactly one table schema.");
  }
  if (sql_query.output_columns_size() == 0) {
    return absl::InvalidArgumentError(
        "SQL query output schema has no columns.");
  }
  if (sql_query.database_schema().table(0).column_size() == 0) {
    return absl::InvalidArgumentError("SQL query input schema has no columns.");
  }
  SqlConfiguration sql_configuration{
      std::move(sql_query.raw_sql()),
      std::move(sql_query.database_schema().table(0)),
      std::move(sql_query.output_columns())};

  std::shared_ptr<MessageFactory> message_factory = nullptr;
  std::string on_device_query_name;

  if (config.has_private_logger_uploads_config()) {
    const MessageDescription& message_description =
        config.private_logger_uploads_config().message_description();
    if (message_description.message_descriptor_set().empty() ||
        message_description.message_name().empty()) {
      return absl::InvalidArgumentError(
          "If private_logger_uploads_config is set, both "
          "message_descriptor_set and "
          "message_name must be set within message_description.");
    }
    FileDescriptorSet descriptor_set;
    if (!descriptor_set.ParseFromString(
            message_description.message_descriptor_set())) {
      return absl::InvalidArgumentError(
          "Failed to parse logged_message_descriptor_set.");
    }
    ABSL_ASSIGN_OR_RETURN(
        message_factory,
        FileDescriptorSetMessageFactory::Create(
            descriptor_set, message_description.message_name()));
    on_device_query_name =
        config.private_logger_uploads_config().on_device_query_name();
  }

  ABSL_RETURN_IF_ERROR(SqliteAdapter::Initialize());

  return std::make_unique<SqlDataIngressFnFactory>(
      std::move(sql_configuration), std::move(message_factory),
      std::move(on_device_query_name));
}

}  // namespace confidential_federated_compute::sql_data_ingress
