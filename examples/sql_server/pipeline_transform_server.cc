#include "examples/sql_server/pipeline_transform_server.h"

#include <stdio.h>

#include <memory>
#include <string>

#include "absl/log/log.h"
#include "examples/sql_server/sql_data.pb.h"
#include "examples/sql_server/sql_data_converter.h"
#include "fcp/base/status_converters.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"

using ::fcp::base::ToGrpcStatus;
using ::fcp::confidentialcompute::ConfigureAndAttestRequest;
using ::fcp::confidentialcompute::ConfigureAndAttestResponse;
using ::fcp::confidentialcompute::PipelineTransform;
using ::fcp::confidentialcompute::TransformRequest;
using ::fcp::confidentialcompute::TransformResponse;
using ::grpc::ServerContext;
using ::sql_data::SqlQuery;

absl::Status SqlPipelineTransform::SqlTransform(const TransformRequest* request,
                                                TransformResponse* response) {
  absl::MutexLock l(&mutex_);
  if (query_.empty()) {
    return absl::FailedPreconditionError(
        "ConfigureAndAttest must be called before Transform.");
  }
  FCP_ASSIGN_OR_RETURN(SqlData input_data, ConvertWireFormatRecordsToSqlData(
                                               request, input_schema_));

  FCP_RETURN_IF_ERROR(sqlite_->SetTableContents(input_schema_, input_data));

  FCP_ASSIGN_OR_RETURN(SqlData result,
                       sqlite_->EvaluateQuery(query_, output_schema_));
  response->add_outputs()->set_unencrypted_data(result.SerializeAsString());
  return absl::OkStatus();
}

absl::Status SqlPipelineTransform::SqlConfigureAndAttest(
    const ConfigureAndAttestRequest* request,
    ConfigureAndAttestResponse* response) {
  absl::MutexLock l(&mutex_);
  if (!query_.empty()) {
    return absl::FailedPreconditionError(
        "ConfigureAndAttest can only be called once.");
  }
  SqlQuery sql_query;
  request->configuration().UnpackTo(&sql_query);
  query_ = sql_query.raw_sql();
  if (sql_query.input_schema().table_size() != 1 ||
      sql_query.output_schema().table_size() != 1) {
    return absl::InvalidArgumentError(
        "SQL query input or output schema does not contain exactly "
        "one table schema.");
  }
  input_schema_ = sql_query.input_schema().table(0);
  output_schema_ = sql_query.output_schema().table(0);

  FCP_ASSIGN_OR_RETURN(sqlite_, SqliteAdapter::Create());
  FCP_RETURN_IF_ERROR(sqlite_->DefineTable(input_schema_.create_table_sql()));
  // We don't support encryption yet, so no need to fill out `response`
  return absl::OkStatus();
}

grpc::Status SqlPipelineTransform::ConfigureAndAttest(
    ServerContext* context, const ConfigureAndAttestRequest* request,
    ConfigureAndAttestResponse* response) {
  return ToGrpcStatus(SqlConfigureAndAttest(request, response));
}

grpc::Status SqlPipelineTransform::Transform(ServerContext* context,
                                             const TransformRequest* request,
                                             TransformResponse* response) {
  return ToGrpcStatus(SqlTransform(request, response));
}
