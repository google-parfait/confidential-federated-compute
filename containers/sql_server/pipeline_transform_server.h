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

using ::fcp::aggregation::FederatedComputeCheckpointParserFactory;
using ::fcp::client::ExampleQueryResult_VectorData;
using ::fcp::client::ExampleQueryResult_VectorData_Values;
using ::fcp::confidentialcompute::ConfigureAndAttestRequest;
using ::fcp::confidentialcompute::ConfigureAndAttestResponse;
using ::fcp::confidentialcompute::PipelineTransform;
using ::fcp::confidentialcompute::Record;
using ::fcp::confidentialcompute::TransformRequest;
using ::fcp::confidentialcompute::TransformResponse;
using grpc::Server;
using grpc::ServerBuilder;
using ::grpc::ServerContext;
using ::grpc::Status;
using ::grpc::StatusCode;
using ::sql_data::ColumnSchema;
using ::sql_data::SqlQuery;
using ::sql_data::TableSchema;

class SqlPipelineTransform final : public PipelineTransform::Service {
 public:
  Status ConfigureAndAttest(ServerContext* context,
                            const ConfigureAndAttestRequest* request,
                            ConfigureAndAttestResponse* response) override;

  Status Transform(ServerContext* context, const TransformRequest* request,
                   TransformResponse* response) override;

 private:
  absl::Status SqlConfigureAndAttest(const ConfigureAndAttestRequest* request,
                                  ConfigureAndAttestResponse* response);

  absl::Status SqlTransform(const TransformRequest* request,
                   TransformResponse* response);

  absl::Mutex mutex_;
  std::string query_ ABSL_GUARDED_BY(mutex_) = "";
  TableSchema input_schema_ ABSL_GUARDED_BY(mutex_);
  TableSchema output_schema_ ABSL_GUARDED_BY(mutex_);
  std::unique_ptr<SqliteAdapter> sqlite_ ABSL_GUARDED_BY(mutex_);
};
