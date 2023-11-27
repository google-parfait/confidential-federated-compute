#include <stdio.h>

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "containers/sql_server/sql_data.pb.h"
#include "fcp/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"

using ::fcp::confidentialcompute::TransformRequest;
using ::sql_data::ColumnSchema;
using ::sql_data::SqlData;
using ::sql_data::TableSchema;

// Convert `Record`s with unencrypted SQL results in the federated compute wire
// format to SqlData. All rows from each record are added to the same `SqlData`
// result.
//
// TODO: Remove this when the SQLite adapter switches to using an interface that
// abstracts away the data format.
absl::StatusOr<SqlData> ConvertWireFormatRecordsToSqlData(
    const TransformRequest* request, const TableSchema& table_schema);
