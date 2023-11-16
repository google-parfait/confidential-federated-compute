#include <stdio.h>

#include <memory>
#include <string>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "examples/sql_server/sql_data.pb.h"
#include "fcp/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "fcp/base/monitoring.h"
#include "fcp/client/example_query_result.pb.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"

using ::fcp::aggregation::DataType;
using ::fcp::aggregation::FederatedComputeCheckpointParserFactory;
using ::fcp::aggregation::Tensor;
using ::fcp::client::ExampleQueryResult_VectorData;
using ::fcp::client::ExampleQueryResult_VectorData_Values;
using ::fcp::confidentialcompute::Record;
using ::fcp::confidentialcompute::TransformRequest;
using ::sql_data::ColumnSchema;
using ::sql_data::SqlData;
using ::sql_data::SqlQuery;
using ::sql_data::TableSchema;

bool TensorTypeMatchesColumnType(ColumnSchema::DataType column_type,
                                 DataType tensor_type) {
  switch (column_type) {
    case ColumnSchema::DOUBLE:
      return tensor_type == DataType::DT_DOUBLE;
    case ColumnSchema::FLOAT:
      return tensor_type == DataType::DT_FLOAT;
    case ColumnSchema::INT32:
      return tensor_type == DataType::DT_INT32;
    case ColumnSchema::INT64:
      return tensor_type == DataType::DT_INT64;
    case ColumnSchema::BOOL:
      // TODO: Update this when bool values are supported by the lightweight
      // client wire format.
      return tensor_type == DataType::DT_INT32;
    case ColumnSchema::STRING:
      return tensor_type == DataType::DT_STRING;
    case ColumnSchema::BYTES:
      return tensor_type == DataType::DT_STRING;
    default:
      return false;
  }
}

absl::StatusOr<SqlData> ConvertWireFormatRecordsToSqlData(
    const TransformRequest* request, const TableSchema& table_schema) {
  SqlData sql_data;
  ExampleQueryResult_VectorData* vector_data = sql_data.mutable_vector_data();
  for (auto& column : table_schema.column()) {
    (*vector_data->mutable_vectors())[column.name()];
  }

  FederatedComputeCheckpointParserFactory parser_factory;
  int total_num_rows = 0;
  for (auto& record : request->inputs()) {
    absl::Cord client_stacked_tensor_result(record.unencrypted_data());
    auto parser = parser_factory.Create(client_stacked_tensor_result);
    int record_num_rows = -1;
    for (auto& column : table_schema.column()) {
      FCP_ASSIGN_OR_RETURN(Tensor tensor_column_values,
                           (*parser)->GetTensor(column.name()));
      if (tensor_column_values.num_elements() == 0) {
        continue;
      }
      if (record_num_rows < 0) {
        record_num_rows = tensor_column_values.num_elements();
      } else if (record_num_rows != tensor_column_values.num_elements()) {
        return absl::InvalidArgumentError(
            "Record has columns with differing numbers of rows.");
      }
      if (!TensorTypeMatchesColumnType(column.type(),
                                       tensor_column_values.dtype())) {
        return absl::InvalidArgumentError(
            "Checkpoint column type does not match the column type specified "
            "in the TableSchema.");
      }
      switch (column.type()) {
        case ColumnSchema::INT32: {
          for (const auto& [unused_index, value] :
               tensor_column_values.AsAggVector<int32_t>()) {
            (*vector_data->mutable_vectors())[column.name()]
                .mutable_int32_values()
                ->add_value(value);
          }
          break;
        }
        case ColumnSchema::INT64: {
          auto agg_vector = tensor_column_values.AsAggVector<int64_t>();
          for (const auto& [unused_index, value] : agg_vector) {
            (*vector_data->mutable_vectors())[column.name()]
                .mutable_int64_values()
                ->add_value(value);
          }
          break;
        }
        case ColumnSchema::BOOL: {
          auto agg_vector = tensor_column_values.AsAggVector<int32_t>();
          for (const auto& [unused_index, value] : agg_vector) {
            (*vector_data->mutable_vectors())[column.name()]
                .mutable_bool_values()
                ->add_value(value);
          }
          break;
        }
        case ColumnSchema::FLOAT: {
          auto agg_vector = tensor_column_values.AsAggVector<float>();
          for (const auto& [unused_index, value] : agg_vector) {
            (*vector_data->mutable_vectors())[column.name()]
                .mutable_float_values()
                ->add_value(value);
          }
          break;
        }
        case ColumnSchema::DOUBLE: {
          auto agg_vector = tensor_column_values.AsAggVector<double>();
          for (const auto& [unused_index, value] : agg_vector) {
            (*vector_data->mutable_vectors())[column.name()]
                .mutable_double_values()
                ->add_value(value);
          }
          break;
        }
        case ColumnSchema::BYTES: {
          auto agg_vector =
              tensor_column_values.AsAggVector<absl::string_view>();
          for (const auto& [unused_index, value] : agg_vector) {
            (*vector_data->mutable_vectors())[column.name()]
                .mutable_bytes_values()
                ->add_value(std::string(value));
          }
          break;
        }
        case ColumnSchema::STRING: {
          auto agg_vector =
              tensor_column_values.AsAggVector<absl::string_view>();
          for (const auto& [unused_index, value] : agg_vector) {
            (*vector_data->mutable_vectors())[column.name()]
                .mutable_string_values()
                ->add_value(std::string(value));
          }
          break;
        }
        default:
          return absl::InvalidArgumentError(
              "Not a valid column type, can't read this column from the "
              "TransformRequest.");
      }
    }
    total_num_rows += record_num_rows;
  }
  sql_data.set_num_rows(total_num_rows);

  return sql_data;
}
