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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SQL_SERVER_SQL_DATA_CONVERTER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SQL_SERVER_SQL_DATA_CONVERTER_H_

#include <stdio.h>

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "containers/sql_server/sql_data.pb.h"
#include "fcp/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "fcp/protos/confidentialcompute/sql_query.pb.h"

// These conversions are unnecessary and are implemented to get the SQL server
// working end-to-end sooner. This is not the desired end state, and these
// conversions will eventually be removed.
namespace confidential_federated_compute::sql_server {

namespace sql_data_converter_internal {
// Convert a `Values` to a `Tensor`.
absl::StatusOr<fcp::aggregation::Tensor> ConvertValuesToTensor(
    const fcp::client::ExampleQueryResult_VectorData_Values& values);
}  // namespace sql_data_converter_internal

// Converts a `Record` with unencrypted SQL results in the federated compute
// wire format to SqlData. All rows from the record are added to the
// `SqlData` proto.
//
// This function may be called multiple times with the same `SqlData` proto and
// different `Record` protos in order to add the contents of each `Record` proto
// to the same `SqlData` proto.
//
// TODO: Remove this when the SQLite adapter switches to using an interface
// that abstracts away the data format.
absl::Status AddWireFormatDataToSqlData(
    absl::string_view wire_format_data,
    const fcp::confidentialcompute::TableSchema& table_schema, sql_data::SqlData& sql_data);

// Convert `SqlData` to the federated compute wire format.
absl::StatusOr<std::string> ConvertSqlDataToWireFormat(sql_data::SqlData data);

}  // namespace confidential_federated_compute::sql_server

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SQL_SERVER_SQL_DATA_CONVERTER_H_
