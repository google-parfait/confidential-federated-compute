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
#include "gmock/gmock.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::fed_sql {

namespace {

using ::fcp::confidentialcompute::ColumnSchema;
using ::google::internal::federated::plan::
    ExampleQuerySpec_OutputVectorSpec_DataType;
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

absl::StatusOr<Tensor> CreateStringTensor(
    const std::string& name,
    const std::initializer_list<absl::string_view>& data) {
  return Tensor::Create(DataType::DT_STRING,
                        {static_cast<int64_t>(data.size())},
                        CreateTestData<absl::string_view>(data), name);
}

TEST(SensitiveColumnsTest, NoSensitiveColumns) {
  absl::StatusOr<Tensor> column = CreateStringTensor("not_sensitive", {"foo"});
  CHECK_OK(column);
  std::vector<Tensor> columns;

  columns.push_back(std::move(*column));

  std::string key = "unused_key";

  ASSERT_TRUE(HashSensitiveColumns(columns, key).ok());
  ASSERT_EQ(columns.size(), 1);
  EXPECT_THAT(columns[0].AsSpan<absl::string_view>(),
              UnorderedElementsAre("foo"));
}

TEST(SensitiveColumnsTest, SensitiveColumnWithStringType) {
  std::string sensitive_value1 = "sensitive_value1";
  std::string sensitive_value2 = "sensitive_value2";

  absl::StatusOr<Tensor> column = CreateStringTensor(
      "SENSITIVE_str_col", {sensitive_value1, sensitive_value2});
  CHECK_OK(column);
  std::vector<Tensor> columns;
  columns.push_back(std::move(*column));

  std::string key = "test_key";
  absl::StatusOr<std::string> hash1 = KeyedHash(sensitive_value1, key);
  CHECK_OK(hash1);
  absl::StatusOr<std::string> hash2 = KeyedHash(sensitive_value2, key);
  CHECK_OK(hash2);

  ASSERT_TRUE(HashSensitiveColumns(columns, key).ok());
  ASSERT_EQ(columns.size(), 1);

  absl::Span<const absl::string_view> column_span =
      columns[0].AsSpan<absl::string_view>();
  EXPECT_THAT(column_span, UnorderedElementsAre(*hash1, *hash2));
}

TEST(SensitiveColumnsTest, SensitiveColumnWithPrefix) {
  std::string sensitive_value1 = "sensitive_value1";
  std::string sensitive_value2 = "sensitive_value2";

  absl::StatusOr<Tensor> column = CreateStringTensor(
      "query-name/SENSITIVE_str_col", {sensitive_value1, sensitive_value2});
  CHECK_OK(column);
  std::vector<Tensor> columns;
  columns.push_back(std::move(*column));

  std::string key = "test_key";
  absl::StatusOr<std::string> hash1 = KeyedHash(sensitive_value1, key);
  CHECK_OK(hash1);
  absl::StatusOr<std::string> hash2 = KeyedHash(sensitive_value2, key);
  CHECK_OK(hash2);

  ASSERT_TRUE(HashSensitiveColumns(columns, key).ok());
  ASSERT_EQ(columns.size(), 1);

  absl::Span<const absl::string_view> column_span =
      columns[0].AsSpan<absl::string_view>();
  EXPECT_THAT(column_span, UnorderedElementsAre(*hash1, *hash2));
}

TEST(SensitiveColumnsTest, SensitiveColumnWithInvalidType) {
  absl::StatusOr<Tensor> column =
      Tensor::Create(DataType::DT_INT64, {1}, CreateTestData<int64_t>({42}),
                     "SENSITIVE_invalid_col");

  std::vector<Tensor> columns;
  columns.push_back(std::move(*column));

  std::string key = "test_key";

  // INT32 is not a valid type for sensitive columns.
  absl::Status result = HashSensitiveColumns(columns, key);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(result.message(),
              HasSubstr("Only DT_STRING types are supported"));
}

TEST(SensitiveColumnsTest, MultipleSensitiveColumns) {
  std::string sensitive_value1 = "sensitive_value1";
  std::string sensitive_value2 = "sensitive_value2";

  absl::StatusOr<Tensor> column1 =
      CreateStringTensor("SENSITIVE_col_1", {sensitive_value1});
  CHECK_OK(column1);
  absl::StatusOr<Tensor> column2 =
      CreateStringTensor("prefix/SENSITIVE_col_2", {sensitive_value2});
  CHECK_OK(column2);

  std::vector<Tensor> columns;
  columns.push_back(std::move(*column1));
  columns.push_back(std::move(*column2));

  std::string key = "test_key";

  ASSERT_TRUE(HashSensitiveColumns(columns, key).ok());
  ASSERT_EQ(columns.size(), 2);
  ASSERT_EQ(columns[0].num_elements(), 1);
  ASSERT_EQ(columns[1].num_elements(), 1);

  absl::StatusOr<std::string> hash1 = KeyedHash(sensitive_value1, key);
  CHECK_OK(hash1);
  absl::StatusOr<std::string> hash2 = KeyedHash(sensitive_value2, key);
  CHECK_OK(hash2);

  absl::Span<const absl::string_view> column_span1 =
      columns[0].AsSpan<absl::string_view>();
  EXPECT_EQ(column_span1.at(0), hash1.value());
  absl::Span<const absl::string_view> column_span2 =
      columns[1].AsSpan<absl::string_view>();
  EXPECT_EQ(column_span2.at(0), hash2.value());
}

}  // namespace

}  // namespace confidential_federated_compute::fed_sql
