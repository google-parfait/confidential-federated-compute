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
#include "containers/fed_sql/sensitive_columns.h"

#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "containers/crypto.h"
#include "containers/crypto_test_utils.h"
#include "containers/fed_sql/sensitive_columns.h"
#include "containers/sql/sqlite_adapter.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/fed_sql_container_config.pb.h"
#include "fcp/protos/confidentialcompute/sql_query.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::fed_sql {

namespace {

using ::confidential_federated_compute::sql::TensorColumn;
using ::fcp::confidential_compute::EncryptMessageResult;
using ::fcp::confidential_compute::MessageDecryptor;
using ::fcp::confidential_compute::MessageEncryptor;
using ::fcp::confidential_compute::NonceAndCounter;
using ::fcp::confidential_compute::NonceGenerator;
using ::fcp::confidential_compute::OkpCwt;
using ::fcp::confidentialcompute::AggCoreAggregationType;
using ::fcp::confidentialcompute::AGGREGATION_TYPE_ACCUMULATE;
using ::fcp::confidentialcompute::AGGREGATION_TYPE_MERGE;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::ColumnSchema;
using ::fcp::confidentialcompute::ConfidentialTransform;
using ::fcp::confidentialcompute::ConfigureRequest;
using ::fcp::confidentialcompute::ConfigureResponse;
using ::fcp::confidentialcompute::DatabaseSchema;
using ::fcp::confidentialcompute::FedSqlContainerFinalizeConfiguration;
using ::fcp::confidentialcompute::FedSqlContainerInitializeConfiguration;
using ::fcp::confidentialcompute::FedSqlContainerWriteConfiguration;
using ::fcp::confidentialcompute::FINALIZATION_TYPE_REPORT;
using ::fcp::confidentialcompute::FINALIZATION_TYPE_SERIALIZE;
using ::fcp::confidentialcompute::FinalizeRequest;
using ::fcp::confidentialcompute::InitializeRequest;
using ::fcp::confidentialcompute::InitializeResponse;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::SqlQuery;
using ::fcp::confidentialcompute::TableSchema;
using ::fcp::confidentialcompute::WriteFinishedResponse;
using ::fcp::confidentialcompute::WriteRequest;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType_BYTES;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType_INT32;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType_STRING;
using ::google::protobuf::RepeatedPtrField;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::grpc::ServerContext;
using ::grpc::StatusCode;
using ::tensorflow_federated::aggregation::CheckpointAggregator;
using ::tensorflow_federated::aggregation::CheckpointBuilder;
using ::tensorflow_federated::aggregation::Configuration;
using ::tensorflow_federated::aggregation::CreateTestData;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;
using ::testing::HasSubstr;
using ::testing::SizeIs;
using ::testing::Test;
using testing::UnorderedElementsAre;

absl::StatusOr<TensorColumn> CreateStringTensorColumn(
    const std::string& name,
    const std::initializer_list<absl::string_view>& data) {
  absl::StatusOr<Tensor> tensor =
      Tensor::Create(DataType::DT_STRING, {static_cast<int64_t>(data.size())},
                     std::move(CreateTestData<absl::string_view>(data)));
  FCP_RETURN_IF_ERROR(tensor.status());
  ColumnSchema schema;
  schema.set_name(name);
  schema.set_type(ExampleQuerySpec_OutputVectorSpec_DataType_STRING);
  return TensorColumn::Create(schema, std::move(*tensor));
}

TEST(SensitiveColumnsTest, NoSensitiveColumns) {
  absl::StatusOr<TensorColumn> tensor_column =
      CreateStringTensorColumn("not_sensitive", {"foo"});
  CHECK_OK(tensor_column);
  std::vector<TensorColumn> columns;

  columns.push_back(std::move(*tensor_column));

  std::string key = "unused_key";

  ASSERT_TRUE(HashSensitiveColumns(columns, key).ok());
  ASSERT_EQ(columns.size(), 1);
  EXPECT_THAT(columns[0].tensor_.AsSpan<absl::string_view>(),
              UnorderedElementsAre("foo"));
}

TEST(SensitiveColumnsTest, SensitiveColumnWithStringType) {
  std::string sensitive_value1 = "sensitive_value1";
  std::string sensitive_value2 = "sensitive_value2";

  absl::StatusOr<TensorColumn> tensor_column = CreateStringTensorColumn(
      "SENSITIVE_str_col", {sensitive_value1, sensitive_value2});
  CHECK_OK(tensor_column);
  std::vector<TensorColumn> columns;
  columns.push_back(std::move(*tensor_column));

  std::string key = "test_key";
  absl::StatusOr<std::string> hash1 = KeyedHash(sensitive_value1, key);
  CHECK_OK(hash1);
  absl::StatusOr<std::string> hash2 = KeyedHash(sensitive_value2, key);
  CHECK_OK(hash2);

  ASSERT_TRUE(HashSensitiveColumns(columns, key).ok());
  ASSERT_EQ(columns.size(), 1);

  absl::Span<const absl::string_view> column_span =
      columns[0].tensor_.AsSpan<absl::string_view>();
  EXPECT_THAT(column_span, UnorderedElementsAre(*hash1, *hash2));
}

TEST(SensitiveColumnsTest, SensitiveColumnWithBytesType) {
  std::string sensitive_value = "sensitive_value";

  absl::StatusOr<TensorColumn> tensor_column =
      CreateStringTensorColumn("SENSITIVE_byte_col", {sensitive_value});
  CHECK_OK(tensor_column);
  tensor_column->column_schema_.set_type(
      ExampleQuerySpec_OutputVectorSpec_DataType_BYTES);
  std::vector<TensorColumn> columns;
  columns.push_back(std::move(*tensor_column));

  std::string key = "test_key";
  absl::StatusOr<std::string> hash = KeyedHash(sensitive_value, key);
  CHECK_OK(hash);

  ASSERT_TRUE(HashSensitiveColumns(columns, key).ok());
  ASSERT_EQ(columns.size(), 1);

  absl::Span<const absl::string_view> column_span =
      columns[0].tensor_.AsSpan<absl::string_view>();
  EXPECT_THAT(column_span, UnorderedElementsAre(*hash));
}

TEST(SensitiveColumnsTest, SensitiveColumnWithPrefix) {
  std::string sensitive_value1 = "sensitive_value1";
  std::string sensitive_value2 = "sensitive_value2";

  absl::StatusOr<TensorColumn> tensor_column = CreateStringTensorColumn(
      "query-name/SENSITIVE_str_col", {sensitive_value1, sensitive_value2});
  CHECK_OK(tensor_column);
  std::vector<TensorColumn> columns;
  columns.push_back(std::move(*tensor_column));

  std::string key = "test_key";
  absl::StatusOr<std::string> hash1 = KeyedHash(sensitive_value1, key);
  CHECK_OK(hash1);
  absl::StatusOr<std::string> hash2 = KeyedHash(sensitive_value2, key);
  CHECK_OK(hash2);

  ASSERT_TRUE(HashSensitiveColumns(columns, key).ok());
  ASSERT_EQ(columns.size(), 1);

  absl::Span<const absl::string_view> column_span =
      columns[0].tensor_.AsSpan<absl::string_view>();
  EXPECT_THAT(column_span, UnorderedElementsAre(*hash1, *hash2));
}

TEST(SensitiveColumnsTest, SensitiveColumnWithInvalidType) {
  absl::StatusOr<TensorColumn> tensor_column =
      CreateStringTensorColumn("SENSITIVE_invalid_col", {"unused"});
  CHECK_OK(tensor_column);
  tensor_column->column_schema_.set_type(
      ExampleQuerySpec_OutputVectorSpec_DataType_INT32);

  std::vector<TensorColumn> columns;
  columns.push_back(std::move(*tensor_column));

  std::string key = "test_key";

  // INT32 is not a valid type for sensitive columns.
  absl::Status result = HashSensitiveColumns(columns, key);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(result.message(),
              HasSubstr("Only STRING or BYTES types are supported"));
}

TEST(SensitiveColumnsTest, MultipleSensitiveColumns) {
  std::string sensitive_value1 = "sensitive_value1";
  std::string sensitive_value2 = "sensitive_value2";

  absl::StatusOr<TensorColumn> tensor_column1 =
      CreateStringTensorColumn("SENSITIVE_col_1", {sensitive_value1});
  CHECK_OK(tensor_column1);
  absl::StatusOr<TensorColumn> tensor_column2 =
      CreateStringTensorColumn("prefix/SENSITIVE_col_2", {sensitive_value2});
  CHECK_OK(tensor_column2);

  std::vector<TensorColumn> columns;
  columns.push_back(std::move(*tensor_column1));
  columns.push_back(std::move(*tensor_column2));

  std::string key = "test_key";

  ASSERT_TRUE(HashSensitiveColumns(columns, key).ok());
  ASSERT_EQ(columns.size(), 2);
  ASSERT_EQ(columns[0].tensor_.num_elements(), 1);
  ASSERT_EQ(columns[1].tensor_.num_elements(), 1);

  absl::StatusOr<std::string> hash1 = KeyedHash(sensitive_value1, key);
  CHECK_OK(hash1);
  absl::StatusOr<std::string> hash2 = KeyedHash(sensitive_value2, key);
  CHECK_OK(hash2);

  absl::Span<const absl::string_view> column_span1 =
      columns[0].tensor_.AsSpan<absl::string_view>();
  EXPECT_EQ(column_span1.at(0), hash1.value());
  absl::Span<const absl::string_view> column_span2 =
      columns[1].tensor_.AsSpan<absl::string_view>();
  EXPECT_EQ(column_span2.at(0), hash2.value());
}

}  // namespace

}  // namespace confidential_federated_compute::fed_sql
