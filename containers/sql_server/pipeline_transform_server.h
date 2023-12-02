#include <stdio.h>

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
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
  grpc::Status ConfigureAndAttest(
      grpc::ServerContext* context,
      const fcp::confidentialcompute::ConfigureAndAttestRequest* request,
      fcp::confidentialcompute::ConfigureAndAttestResponse* response) override;

  grpc::Status Transform(
      grpc::ServerContext* context,
      const fcp::confidentialcompute::TransformRequest* request,
      fcp::confidentialcompute::TransformResponse* response) override;

 private:
  absl::Status SqlConfigureAndAttest(
      const fcp::confidentialcompute::ConfigureAndAttestRequest* request,
      fcp::confidentialcompute::ConfigureAndAttestResponse* response);

  absl::Status SqlTransform(
      const fcp::confidentialcompute::TransformRequest* request,
      fcp::confidentialcompute::TransformResponse* response);

  absl::Mutex mutex_;
  std::string query_ ABSL_GUARDED_BY(mutex_) = "";
  sql_data::TableSchema input_schema_ ABSL_GUARDED_BY(mutex_);
  sql_data::TableSchema output_schema_ ABSL_GUARDED_BY(mutex_);
  std::unique_ptr<SqliteAdapter> sqlite_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace confidential_federated_compute::sql_server
