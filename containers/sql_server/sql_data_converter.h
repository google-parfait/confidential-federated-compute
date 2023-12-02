#include <stdio.h>

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "containers/sql_server/sql_data.pb.h"
#include "fcp/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"

// These conversions are unnecessary and are implemented to get the SQL server
// working end-to-end sooner. This is not the desired end state, and these
// conversions will eventually be removed.
namespace confidential_federated_compute::sql_server {

namespace sql_data_converter_internal {
// Convert a `Values` to a `Tensor`.
absl::StatusOr<fcp::aggregation::Tensor> ConvertValuesToTensor(
    const fcp::client::ExampleQueryResult_VectorData_Values& values);
}  // namespace sql_data_converter_internal

// Convert `Record`s with unencrypted SQL results in the federated compute
// wire format to SqlData. All rows from each record are added to the same
// `SqlData` result.
//
// TODO: Remove this when the SQLite adapter switches to using an interface
// that abstracts away the data format.
absl::StatusOr<sql_data::SqlData> ConvertWireFormatRecordsToSqlData(
    const fcp::confidentialcompute::TransformRequest* request,
    const sql_data::TableSchema& table_schema);

// Convert `SqlData` to the federated compute wire format.
absl::StatusOr<fcp::confidentialcompute::TransformResponse>
ConvertSqlDataToWireFormat(sql_data::SqlData data);

}  // namespace confidential_federated_compute::sql_server
