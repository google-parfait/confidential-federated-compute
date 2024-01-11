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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SQL_SERVER_PIPELINE_TRANSFORM_SERVER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SQL_SERVER_PIPELINE_TRANSFORM_SERVER_H_

#include <stdio.h>

#include <memory>
#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "containers/crypto.h"
#include "containers/sql_server/sql_data.pb.h"
#include "containers/sql_server/sql_data_converter.h"
#include "containers/sql_server/sqlite_adapter.h"
#include "fcp/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/example_query_result.pb.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"

namespace confidential_federated_compute::sql_server {

class SqlPipelineTransform final
    : public fcp::confidentialcompute::PipelineTransform::Service {
 public:
  explicit SqlPipelineTransform();

  ~SqlPipelineTransform();

  grpc::Status ConfigureAndAttest(
      grpc::ServerContext* context,
      const fcp::confidentialcompute::ConfigureAndAttestRequest* request,
      fcp::confidentialcompute::ConfigureAndAttestResponse* response) override;

  grpc::Status GenerateNonces(
      grpc::ServerContext* context,
      const fcp::confidentialcompute::GenerateNoncesRequest* request,
      fcp::confidentialcompute::GenerateNoncesResponse* response) override;

  grpc::Status Transform(
      grpc::ServerContext* context,
      const fcp::confidentialcompute::TransformRequest* request,
      fcp::confidentialcompute::TransformResponse* response) override;

 private:
  // Configuration of the SQL PipelineTransform server set by
  // ConfigureAndAttest.
  struct SqlConfiguration {
    std::string query;
    sql_data::TableSchema input_schema;
    sql_data::TableSchema output_schema;
  };

  absl::Status SqlConfigureAndAttest(
      const fcp::confidentialcompute::ConfigureAndAttestRequest* request,
      fcp::confidentialcompute::ConfigureAndAttestResponse* response);

  absl::Status SqlGenerateNonces(
      const fcp::confidentialcompute::GenerateNoncesRequest* request,
      fcp::confidentialcompute::GenerateNoncesResponse* response);

  absl::Status SqlTransform(
      const fcp::confidentialcompute::TransformRequest* request,
      fcp::confidentialcompute::TransformResponse* response);

  absl::Mutex mutex_;
  // The mutex is used to protect the optional wrapping the SqlConfiguration to
  // ensure the SqlConfiguration is initialized, but the SqlConfiguration itself
  // is threadsafe.
  std::optional<const SqlConfiguration> configuration_ ABSL_GUARDED_BY(mutex_);
  // The mutex is used to protect the optional wrapping the RecordDecryptor to
  // ensure the RecordDecryptor is initialized, but the RecordDecryptor itself
  // is threadsafe.
  std::optional<confidential_federated_compute::RecordDecryptor>
      record_decryptor_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace confidential_federated_compute::sql_server

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SQL_SERVER_PIPELINE_TRANSFORM_SERVER_H_
