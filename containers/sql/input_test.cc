// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "containers/sql/input.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "containers/sql/row_view.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.pb.h"
#include "google/protobuf/dynamic_message.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "testing/matchers.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::sql {
namespace {

using ::absl_testing::IsOk;
using ::google::protobuf::Descriptor;
using ::google::protobuf::DescriptorPool;
using ::google::protobuf::DynamicMessageFactory;
using ::google::protobuf::FieldDescriptorProto;
using ::google::protobuf::FileDescriptor;
using ::google::protobuf::FileDescriptorProto;
using ::google::protobuf::Message;
using ::google::protobuf::Reflection;
using ::tensorflow_federated::aggregation::CreateTestData;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;

class InputTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto t1 = Tensor::Create(DataType::DT_INT64, TensorShape({2}),
                             CreateTestData<int64_t>({1, 2}), "col1");
    CHECK_OK(t1);
    contents_.push_back(*std::move(t1));
    auto t2 = Tensor::Create(DataType::DT_STRING, TensorShape({2}),
                             CreateTestData<absl::string_view>({"foo", "bar"}),
                             "col2");
    CHECK_OK(t2);
    contents_.push_back(*std::move(t2));

    blob_header_.set_access_policy_node_id(42);
  }
  std::vector<Tensor> contents_;
  fcp::confidentialcompute::BlobHeader blob_header_;
};

TEST_F(InputTest, CreateFromTensors) {
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(contents_), blob_header_);
  ASSERT_THAT(input, IsOk());
}

TEST_F(InputTest, CreateFromTensorsFailsWithNoColumns) {
  absl::StatusOr<Input> input = Input::CreateFromTensors({}, blob_header_);
  EXPECT_THAT(input.status(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     "No columns provided."));
}

TEST_F(InputTest, CreateFromTensorsFailsWithNoRows) {
  contents_.clear();
  contents_.push_back(*Tensor::Create(DataType::DT_INT64, TensorShape({}),
                                      CreateTestData<int64_t>({1}), "col1"));
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(contents_), blob_header_);
  EXPECT_THAT(input.status(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     "Column has no rows."));
}

TEST_F(InputTest, CreateFromTensorsFailsWithMultiDimensionalRows) {
  contents_.clear();
  contents_.push_back(*Tensor::Create(DataType::DT_INT64, TensorShape({1, 2}),
                                      CreateTestData<int64_t>({1, 2}), "col1"));
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(contents_), blob_header_);
  EXPECT_THAT(input.status(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     "Column has more than one dimension."));
}

TEST_F(InputTest, CreateFromTensorsFailsWithMismatchedRows) {
  contents_.push_back(*Tensor::Create(DataType::DT_INT64, TensorShape({3}),
                                      CreateTestData<int64_t>({1, 2, 3}),
                                      "col3"));
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(contents_), blob_header_);
  EXPECT_THAT(
      input.status(),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             "All columns must have the same number of rows."));
}

TEST_F(InputTest, GetColumnNames) {
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(contents_), blob_header_);
  ASSERT_THAT(input, IsOk());
  EXPECT_EQ(input->GetColumnNames(),
            std::vector<std::string>({"col1", "col2"}));
}

TEST_F(InputTest, GetBlobHeader) {
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(contents_), blob_header_);
  ASSERT_THAT(input, IsOk());
  EXPECT_THAT(input->GetBlobHeader(), EqualsProto(blob_header_));
}

TEST_F(InputTest, GetRow) {
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(contents_), blob_header_);
  ASSERT_THAT(input, IsOk());
  absl::StatusOr<RowView> row = input->GetRow(0);
  ASSERT_THAT(row, IsOk());
  EXPECT_EQ(row->GetColumnCount(), 2);
  EXPECT_EQ(row->GetValue<int64_t>(0), 1);
  EXPECT_EQ(row->GetValue<absl::string_view>(1), "foo");

  absl::StatusOr<RowView> row1 = input->GetRow(1);
  ASSERT_THAT(row1, IsOk());
  EXPECT_EQ(row1->GetValue<int64_t>(0), 2);
  EXPECT_EQ(row1->GetValue<absl::string_view>(1), "bar");
}

TEST_F(InputTest, GetRowCount) {
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(contents_), blob_header_);
  ASSERT_THAT(input, IsOk());
  EXPECT_EQ(input->GetRowCount(), 2);
}

TEST_F(InputTest, MoveToTensors) {
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(contents_), blob_header_);
  ASSERT_THAT(input, IsOk());
  absl::StatusOr<std::vector<Tensor>> tensors =
      std::move(*input).MoveToTensors();
  ASSERT_THAT(tensors, IsOk());
  ASSERT_EQ(tensors->size(), 2);
  EXPECT_EQ(tensors->at(0).name(), "col1");
  EXPECT_EQ(tensors->at(0).dtype(), DataType::DT_INT64);
  EXPECT_EQ(tensors->at(0).shape(), TensorShape({2}));
  EXPECT_THAT(tensors->at(0).AsSpan<int64_t>(), testing::ElementsAre(1, 2));
  EXPECT_EQ(tensors->at(1).name(), "col2");
  EXPECT_EQ(tensors->at(1).dtype(), DataType::DT_STRING);
  EXPECT_EQ(tensors->at(1).shape(), TensorShape({2}));
  EXPECT_THAT(tensors->at(1).AsSpan<absl::string_view>(),
              testing::ElementsAre("foo", "bar"));
}

class MessageInputTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const FileDescriptorProto file_proto = PARSE_TEXT_PROTO(R"pb(
      name: "test.proto"
      package: "confidential_federated_compute.sql"
      message_type {
        name: "TestMessage"
        field { name: "col1" number: 1 type: TYPE_INT32 label: LABEL_OPTIONAL }
        field { name: "col2" number: 2 type: TYPE_INT64 label: LABEL_OPTIONAL }
        field { name: "col3" number: 3 type: TYPE_FLOAT label: LABEL_OPTIONAL }
        field { name: "col4" number: 4 type: TYPE_DOUBLE label: LABEL_OPTIONAL }
        field { name: "col5" number: 5 type: TYPE_STRING label: LABEL_OPTIONAL }
      }
    )pb");

    pool_ = std::make_unique<DescriptorPool>();
    const google::protobuf::FileDescriptor* file_descriptor =
        pool_->BuildFile(file_proto);
    ASSERT_NE(file_descriptor, nullptr);

    const Descriptor* descriptor =
        file_descriptor->FindMessageTypeByName("TestMessage");
    ASSERT_NE(descriptor, nullptr);

    factory_ = std::make_unique<DynamicMessageFactory>(pool_.get());
    messages_.push_back(CreateMessage(descriptor, 1, 2, 3.0f, 4.0, "foo"));
    messages_.push_back(CreateMessage(descriptor, 11, 12, 13.0f, 14.0, "bar"));

    auto t1 = Tensor::Create(DataType::DT_INT64, TensorShape({2}),
                             CreateTestData<int64_t>({42, 24}), "system_col1");
    CHECK_OK(t1);
    system_columns_.push_back(*std::move(t1));
    auto t2 = Tensor::Create(DataType::DT_STRING, TensorShape({2}),
                             CreateTestData<absl::string_view>({"baz", "qux"}),
                             "system_col2");
    CHECK_OK(t2);
    system_columns_.push_back(*std::move(t2));

    blob_header_.set_access_policy_node_id(42);
  }

  std::unique_ptr<Message> CreateMessage(const Descriptor* descriptor,
                                         int32_t int32_val, int64_t int64_val,
                                         float float_val, double double_val,
                                         std::string string_val) {
    std::unique_ptr<Message> message =
        std::unique_ptr<Message>(factory_->GetPrototype(descriptor)->New());

    const Reflection* reflection = message->GetReflection();
    reflection->SetInt32(message.get(), descriptor->FindFieldByName("col1"),
                         int32_val);
    reflection->SetInt64(message.get(), descriptor->FindFieldByName("col2"),
                         int64_val);
    reflection->SetFloat(message.get(), descriptor->FindFieldByName("col3"),
                         float_val);
    reflection->SetDouble(message.get(), descriptor->FindFieldByName("col4"),
                          double_val);
    reflection->SetString(message.get(), descriptor->FindFieldByName("col5"),
                          string_val);
    return message;
  }

  std::unique_ptr<DescriptorPool> pool_;
  std::unique_ptr<DynamicMessageFactory> factory_;
  std::vector<std::unique_ptr<Message>> messages_;
  std::vector<Tensor> system_columns_;
  fcp::confidentialcompute::BlobHeader blob_header_;
};

TEST_F(MessageInputTest, CreateFromMessages) {
  absl::StatusOr<Input> input = Input::CreateFromMessages(
      std::move(messages_), std::move(system_columns_), blob_header_);
  ASSERT_THAT(input, IsOk());
}

TEST_F(MessageInputTest, CreateFromMessagesFailsWithNoRows) {
  absl::StatusOr<Input> input = Input::CreateFromMessages({}, {}, blob_header_);
  EXPECT_THAT(input.status(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     "No rows provided."));
}

TEST_F(MessageInputTest, CreateFromMessagesFailsWithMismatchedMessageTypes) {
  // Create a new descriptor for a different message type.
  const FileDescriptorProto file_proto2 = PARSE_TEXT_PROTO(R"pb(
    name: "test2.proto"
    package: "confidential_federated_compute.sql"
    message_type {
      name: "TestMessage2"
      field {
        name: "another_col"
        number: 1
        type: TYPE_INT32
        label: LABEL_OPTIONAL
      }
    }
  )pb");
  const FileDescriptor* file_descriptor = pool_->BuildFile(file_proto2);
  ASSERT_NE(file_descriptor, nullptr);
  const Descriptor* descriptor2 =
      file_descriptor->FindMessageTypeByName("TestMessage2");
  ASSERT_NE(descriptor2, nullptr);

  // Create a message with the new type.
  std::unique_ptr<Message> message2 =
      std::unique_ptr<Message>(factory_->GetPrototype(descriptor2)->New());
  message2->GetReflection()->SetInt32(
      message2.get(), descriptor2->FindFieldByName("another_col"), 123);

  // Add the new message to contents.
  messages_.push_back(std::move(message2));
  absl::StatusOr<Input> input = Input::CreateFromMessages(
      std::move(messages_), std::move(system_columns_), blob_header_);
  EXPECT_THAT(input.status(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     "All messages in a table must have the "
                                     "same proto type."));
}

TEST_F(MessageInputTest,
       CreateFromMessagesFailsWithMismatchedSystemColumnRows) {
  // Add a system column with a different number of rows.
  auto t3 = Tensor::Create(DataType::DT_INT64, TensorShape({3}),
                           CreateTestData<int64_t>({1, 2, 3}), "system_col3");
  ASSERT_THAT(t3, IsOk());
  system_columns_.push_back(*std::move(t3));
  absl::StatusOr<Input> input = Input::CreateFromMessages(
      std::move(messages_), std::move(system_columns_), blob_header_);
  EXPECT_THAT(input.status(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     "System columns must have the same "
                                     "number of rows as the table."));
}

TEST_F(MessageInputTest,
       CreateFromMessagesFailsWithMultiDimensionalSystemColumn) {
  // Add a system column with more than one dimension.
  auto t3 = Tensor::Create(DataType::DT_INT64, TensorShape({2, 1}),
                           CreateTestData<int64_t>({1, 2}), "system_col3");
  ASSERT_THAT(t3, IsOk());
  system_columns_.push_back(*std::move(t3));
  absl::StatusOr<Input> input = Input::CreateFromMessages(
      std::move(messages_), std::move(system_columns_), blob_header_);
  EXPECT_THAT(
      input.status(),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             "System columns must have a single dimension."));
}

TEST_F(MessageInputTest, GetColumnNames) {
  absl::StatusOr<Input> input = Input::CreateFromMessages(
      std::move(messages_), std::move(system_columns_), blob_header_);
  ASSERT_THAT(input, IsOk());
  EXPECT_EQ(input->GetColumnNames(),
            std::vector<std::string>({"col1", "col2", "col3", "col4", "col5",
                                      "system_col1", "system_col2"}));
}

TEST_F(MessageInputTest, GetRowCount) {
  absl::StatusOr<Input> input = Input::CreateFromMessages(
      std::move(messages_), std::move(system_columns_), blob_header_);
  ASSERT_THAT(input, IsOk());
  EXPECT_EQ(input->GetRowCount(), 2);
}

TEST_F(MessageInputTest, GetRow) {
  absl::StatusOr<Input> input = Input::CreateFromMessages(
      std::move(messages_), std::move(system_columns_), blob_header_);
  ASSERT_THAT(input, IsOk());
  absl::StatusOr<RowView> row = input->GetRow(0);
  ASSERT_THAT(row, IsOk());
  EXPECT_EQ(row->GetColumnCount(), 7);
  EXPECT_EQ(row->GetValue<int32_t>(0), 1);
  EXPECT_EQ(row->GetValue<int64_t>(1), 2);
  EXPECT_EQ(row->GetValue<float>(2), 3.0f);
  EXPECT_EQ(row->GetValue<double>(3), 4.0);
  EXPECT_EQ(row->GetValue<absl::string_view>(4), "foo");
  EXPECT_EQ(row->GetValue<int64_t>(5), 42);
  EXPECT_EQ(row->GetValue<absl::string_view>(6), "baz");

  absl::StatusOr<RowView> row1 = input->GetRow(1);
  ASSERT_THAT(row1, IsOk());
  EXPECT_EQ(row1->GetValue<int32_t>(0), 11);
  EXPECT_EQ(row1->GetValue<int64_t>(1), 12);
  EXPECT_EQ(row1->GetValue<float>(2), 13.0f);
  EXPECT_EQ(row1->GetValue<double>(3), 14.0);
  EXPECT_EQ(row1->GetValue<absl::string_view>(4), "bar");
  EXPECT_EQ(row1->GetValue<int64_t>(5), 24);
  EXPECT_EQ(row1->GetValue<absl::string_view>(6), "qux");
}

TEST_F(MessageInputTest, MoveToTensors) {
  absl::StatusOr<Input> input = Input::CreateFromMessages(
      std::move(messages_), std::move(system_columns_), blob_header_);
  ASSERT_THAT(input, IsOk());
  absl::StatusOr<std::vector<Tensor>> tensors =
      std::move(*input).MoveToTensors();
  ASSERT_THAT(tensors, IsOk());
  ASSERT_EQ(tensors->size(), 7);

  EXPECT_EQ(tensors->at(0).name(), "col1");
  EXPECT_EQ(tensors->at(0).dtype(), DataType::DT_INT32);
  EXPECT_EQ(tensors->at(0).shape(), TensorShape({2}));
  EXPECT_THAT(tensors->at(0).AsSpan<int32_t>(), testing::ElementsAre(1, 11));

  EXPECT_EQ(tensors->at(1).name(), "col2");
  EXPECT_EQ(tensors->at(1).dtype(), DataType::DT_INT64);
  EXPECT_EQ(tensors->at(1).shape(), TensorShape({2}));
  EXPECT_THAT(tensors->at(1).AsSpan<int64_t>(), testing::ElementsAre(2, 12));

  EXPECT_EQ(tensors->at(2).name(), "col3");
  EXPECT_EQ(tensors->at(2).dtype(), DataType::DT_FLOAT);
  EXPECT_EQ(tensors->at(2).shape(), TensorShape({2}));
  EXPECT_THAT(tensors->at(2).AsSpan<float>(),
              testing::ElementsAre(3.0f, 13.0f));

  EXPECT_EQ(tensors->at(3).name(), "col4");
  EXPECT_EQ(tensors->at(3).dtype(), DataType::DT_DOUBLE);
  EXPECT_EQ(tensors->at(3).shape(), TensorShape({2}));
  EXPECT_THAT(tensors->at(3).AsSpan<double>(), testing::ElementsAre(4.0, 14.0));

  EXPECT_EQ(tensors->at(4).name(), "col5");
  EXPECT_EQ(tensors->at(4).dtype(), DataType::DT_STRING);
  EXPECT_EQ(tensors->at(4).shape(), TensorShape({2}));
  EXPECT_THAT(tensors->at(4).AsSpan<absl::string_view>(),
              testing::ElementsAre("foo", "bar"));

  EXPECT_EQ(tensors->at(5).name(), "system_col1");
  EXPECT_EQ(tensors->at(5).dtype(), DataType::DT_INT64);
  EXPECT_EQ(tensors->at(5).shape(), TensorShape({2}));
  EXPECT_THAT(tensors->at(5).AsSpan<int64_t>(), testing::ElementsAre(42, 24));

  EXPECT_EQ(tensors->at(6).name(), "system_col2");
  EXPECT_EQ(tensors->at(6).dtype(), DataType::DT_STRING);
  EXPECT_EQ(tensors->at(6).shape(), TensorShape({2}));
  EXPECT_THAT(tensors->at(6).AsSpan<absl::string_view>(),
              testing::ElementsAre("baz", "qux"));
}
}  // namespace
}  // namespace confidential_federated_compute::sql
