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
#include <stdio.h>

#include <memory>
#include <string>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "containers/sql_server/sql_data.pb.h"
#include "fcp/aggregation/core/mutable_vector_data.h"
#include "fcp/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "fcp/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/status_converters.h"
#include "fcp/client/example_query_result.pb.h"
#include "fcp/protos/confidentialcompute/sql_query.pb.h"

namespace confidential_federated_compute::sql_server {

using ::fcp::aggregation::CheckpointBuilder;
using ::fcp::aggregation::DataType;
using ::fcp::aggregation::FederatedComputeCheckpointBuilderFactory;
using ::fcp::aggregation::FederatedComputeCheckpointParserFactory;
using ::fcp::aggregation::MutableVectorData;
using ::fcp::aggregation::Tensor;
using ::fcp::aggregation::TensorShape;
using ::fcp::client::ExampleQueryResult_VectorData;
using ::fcp::client::ExampleQueryResult_VectorData_Values;
using ::sql_data::SqlData;
using ::fcp::confidentialcompute::SqlQuery;
using ::fcp::confidentialcompute::ColumnSchema;
using ::fcp::confidentialcompute::TableSchema;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType_BOOL;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType_BYTES;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType_DOUBLE;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType_FLOAT;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType_INT32;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType_INT64;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType_STRING;

namespace sql_data_converter_internal {

// `values` must outlive the returned Tensor.
absl::StatusOr<Tensor> ConvertValuesToTensor(
    const ExampleQueryResult_VectorData_Values& values) {
  if (values.has_int32_values()) {
    std::unique_ptr<MutableVectorData<int32_t>> vector_data =
        std::make_unique<MutableVectorData<int32_t>>(
            values.int32_values().value().begin(),
            values.int32_values().value().end());
    return Tensor::Create(DataType::DT_INT32,
                          TensorShape({values.int32_values().value_size()}),
                          std::move(vector_data));
  } else if (values.has_int64_values()) {
    std::unique_ptr<MutableVectorData<int64_t>> vector_data =
        std::make_unique<MutableVectorData<int64_t>>(
            values.int64_values().value().begin(),
            values.int64_values().value().end());
    return Tensor::Create(DataType::DT_INT64,
                          TensorShape({values.int64_values().value_size()}),
                          std::move(vector_data));

  } else if (values.has_bool_values()) {
    std::unique_ptr<MutableVectorData<int32_t>> vector_data =
        std::make_unique<MutableVectorData<int32_t>>(
            values.bool_values().value().begin(),
            values.bool_values().value().end());
    return Tensor::Create(DataType::DT_INT32,
                          TensorShape({values.bool_values().value_size()}),
                          std::move(vector_data));
  } else if (values.has_float_values()) {
    std::unique_ptr<MutableVectorData<float>> vector_data =
        std::make_unique<MutableVectorData<float>>(
            values.float_values().value().begin(),
            values.float_values().value().end());
    return Tensor::Create(DataType::DT_FLOAT,
                          TensorShape({values.float_values().value_size()}),
                          std::move(vector_data));
  } else if (values.has_double_values()) {
    std::unique_ptr<MutableVectorData<double>> vector_data =
        std::make_unique<MutableVectorData<double>>(
            values.double_values().value().begin(),
            values.double_values().value().end());
    return Tensor::Create(DataType::DT_DOUBLE,
                          TensorShape({values.double_values().value_size()}),
                          std::move(vector_data));

  } else if (values.has_string_values()) {
    std::unique_ptr<MutableVectorData<absl::string_view>> vector_data =
        std::make_unique<MutableVectorData<absl::string_view>>(
            values.string_values().value().begin(),
            values.string_values().value().end());
    return Tensor::Create(DataType::DT_STRING,
                          TensorShape({values.string_values().value_size()}),
                          std::move(vector_data));
  } else if (values.has_bytes_values()) {
    std::unique_ptr<MutableVectorData<absl::string_view>> vector_data =
        std::make_unique<MutableVectorData<absl::string_view>>(
            values.bytes_values().value().begin(),
            values.bytes_values().value().end());
    return Tensor::Create(DataType::DT_STRING,
                          TensorShape({values.bytes_values().value_size()}),
                          std::move(vector_data));
  } else {
    return absl::InvalidArgumentError(
        "Not a valid Values type, can't convert this value to a Tensor.");
  }
}

}  // namespace sql_data_converter_internal

bool TensorTypeMatchesColumnType(ExampleQuerySpec_OutputVectorSpec_DataType column_type,
                                 DataType tensor_type) {
  switch (column_type) {
    case ExampleQuerySpec_OutputVectorSpec_DataType_DOUBLE:
      return tensor_type == DataType::DT_DOUBLE;
    case ExampleQuerySpec_OutputVectorSpec_DataType_FLOAT:
      return tensor_type == DataType::DT_FLOAT;
    case ExampleQuerySpec_OutputVectorSpec_DataType_INT32:
      return tensor_type == DataType::DT_INT32;
    case ExampleQuerySpec_OutputVectorSpec_DataType_INT64:
      return tensor_type == DataType::DT_INT64;
    case ExampleQuerySpec_OutputVectorSpec_DataType_BOOL:
      // TODO: Update this when bool values are supported by the lightweight
      // client wire format.
      return tensor_type == DataType::DT_INT32;
    case ExampleQuerySpec_OutputVectorSpec_DataType_STRING:
      return tensor_type == DataType::DT_STRING;
    case ExampleQuerySpec_OutputVectorSpec_DataType_BYTES:
      return tensor_type == DataType::DT_STRING;
    default:
      return false;
  }
}

absl::Status AddWireFormatDataToSqlData(
    absl::string_view wire_format_data,
    const TableSchema& table_schema, sql_data::SqlData& sql_data) {
  ExampleQueryResult_VectorData* vector_data = sql_data.mutable_vector_data();
  for (auto& column : table_schema.column()) {
    (*vector_data->mutable_vectors())[column.name()];
  }

  FederatedComputeCheckpointParserFactory parser_factory;
  int total_num_rows = sql_data.num_rows();
  absl::Cord client_stacked_tensor_result(wire_format_data);
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
      case ExampleQuerySpec_OutputVectorSpec_DataType_INT32: {
        for (const auto& [unused_index, value] :
             tensor_column_values.AsAggVector<int32_t>()) {
          (*vector_data->mutable_vectors())[column.name()]
              .mutable_int32_values()
              ->add_value(value);
        }
        break;
      }
      case ExampleQuerySpec_OutputVectorSpec_DataType_INT64: {
        auto agg_vector = tensor_column_values.AsAggVector<int64_t>();
        for (const auto& [unused_index, value] : agg_vector) {
          (*vector_data->mutable_vectors())[column.name()]
              .mutable_int64_values()
              ->add_value(value);
        }
        break;
      }
      case ExampleQuerySpec_OutputVectorSpec_DataType_BOOL: {
        auto agg_vector = tensor_column_values.AsAggVector<int32_t>();
        for (const auto& [unused_index, value] : agg_vector) {
          (*vector_data->mutable_vectors())[column.name()]
              .mutable_bool_values()
              ->add_value(value);
        }
        break;
      }
      case ExampleQuerySpec_OutputVectorSpec_DataType_FLOAT: {
        auto agg_vector = tensor_column_values.AsAggVector<float>();
        for (const auto& [unused_index, value] : agg_vector) {
          (*vector_data->mutable_vectors())[column.name()]
              .mutable_float_values()
              ->add_value(value);
        }
        break;
      }
      case ExampleQuerySpec_OutputVectorSpec_DataType_DOUBLE: {
        auto agg_vector = tensor_column_values.AsAggVector<double>();
        for (const auto& [unused_index, value] : agg_vector) {
          (*vector_data->mutable_vectors())[column.name()]
              .mutable_double_values()
              ->add_value(value);
        }
        break;
      }
      case ExampleQuerySpec_OutputVectorSpec_DataType_BYTES: {
        auto agg_vector = tensor_column_values.AsAggVector<absl::string_view>();
        for (const auto& [unused_index, value] : agg_vector) {
          (*vector_data->mutable_vectors())[column.name()]
              .mutable_bytes_values()
              ->add_value(std::string(value));
        }
        break;
      }
      case ExampleQuerySpec_OutputVectorSpec_DataType_STRING: {
        auto agg_vector = tensor_column_values.AsAggVector<absl::string_view>();
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
  sql_data.set_num_rows(total_num_rows);
  return absl::OkStatus();
}

absl::StatusOr<std::string> ConvertSqlDataToWireFormat(SqlData data) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> ckpt_builder = builder_factory.Create();

  for (const auto& [col_name, column] : data.vector_data().vectors()) {
    FCP_ASSIGN_OR_RETURN(
        Tensor tensor,
        sql_data_converter_internal::ConvertValuesToTensor(column));
    FCP_RETURN_IF_ERROR(ckpt_builder->Add(col_name, tensor));
  }

  FCP_ASSIGN_OR_RETURN(absl::Cord ckpt_cord, ckpt_builder->Build());
  // Protobuf version 23.0 is required to use [ctype = CORD], however, we can't
  // use this since it isn't currently compatible with TensorFlow.
  std::string ckpt_string;
  absl::CopyCordToString(ckpt_cord, &ckpt_string);
  return ckpt_string;
}

}  // namespace confidential_federated_compute::sql_server
