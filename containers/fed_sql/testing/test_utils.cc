// Copyright 2025 Google LLC.
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

#include "containers/fed_sql/testing/test_utils.h"

#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "fcp/confidentialcompute/constants.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.pb.h"
#include "google/protobuf/dynamic_message.h"
#include "google/protobuf/message.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::fed_sql::testing {

namespace {

using ::fcp::confidential_compute::kEventTimeColumnName;
using ::fcp::confidential_compute::kPrivateLoggerEntryKey;
using ::tensorflow_federated::aggregation::CheckpointBuilder;
using ::tensorflow_federated::aggregation::CreateTestData;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::MutableStringData;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;

}  // namespace

std::unique_ptr<MutableStringData> CreateStringTestData(
    std::vector<std::string> data) {
  std::unique_ptr<MutableStringData> tensor_data =
      std::make_unique<MutableStringData>(data.size());
  for (auto& value : data) {
    tensor_data->Add(std::move(value));
  }
  return tensor_data;
}

std::string BuildFedSqlGroupByCheckpoint(
    std::initializer_list<uint64_t> key_col_values,
    std::initializer_list<uint64_t> val_col_values,
    const std::string& key_col_name, const std::string& val_col_name) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> ckpt_builder = builder_factory.Create();

  absl::StatusOr<Tensor> key =
      Tensor::Create(DataType::DT_INT64,
                     TensorShape({static_cast<int64_t>(key_col_values.size())}),
                     CreateTestData<uint64_t>(key_col_values));
  absl::StatusOr<Tensor> val =
      Tensor::Create(DataType::DT_INT64,
                     TensorShape({static_cast<int64_t>(val_col_values.size())}),
                     CreateTestData<uint64_t>(val_col_values));
  CHECK_OK(key);
  CHECK_OK(val);
  CHECK_OK(ckpt_builder->Add(key_col_name, *key));
  CHECK_OK(ckpt_builder->Add(val_col_name, *val));
  auto checkpoint = ckpt_builder->Build();
  CHECK_OK(checkpoint.status());

  std::string checkpoint_string;
  absl::CopyCordToString(*checkpoint, &checkpoint_string);
  return checkpoint_string;
}

MessageHelper::MessageHelper() {
  const google::protobuf::FileDescriptorProto file_proto =
      PARSE_TEXT_PROTO(R"pb(
        name: "test.proto"
        package: "confidential_federated_compute.fed_sql"
        message_type {
          name: "TestMessage"
          field { name: "key" number: 1 type: TYPE_INT64 label: LABEL_OPTIONAL }
          field { name: "val" number: 2 type: TYPE_INT64 label: LABEL_OPTIONAL }
        }
      )pb");
  const google::protobuf::FileDescriptor* file_descriptor =
      pool_.BuildFile(file_proto);
  CHECK_NE(file_descriptor, nullptr);

  descriptor_ = file_descriptor->FindMessageTypeByName("TestMessage");
  CHECK_NE(descriptor_, nullptr);

  prototype_ = factory_.GetPrototype(descriptor_);
}

std::unique_ptr<google::protobuf::Message> MessageHelper::CreateMessage(
    int64_t key, int64_t val) {
  std::unique_ptr<google::protobuf::Message> message(prototype_->New());
  const google::protobuf::Reflection* reflection = message->GetReflection();
  reflection->SetInt64(message.get(), descriptor_->FindFieldByName("key"), key);
  reflection->SetInt64(message.get(), descriptor_->FindFieldByName("val"), val);
  return message;
}

const google::protobuf::Message* MessageHelper::prototype() const {
  return prototype_;
}

const google::protobuf::FileDescriptor* MessageHelper::file_descriptor() const {
  return descriptor_->file();
}

absl::string_view MessageHelper::message_name() const {
  return descriptor_->full_name();
}

absl::StatusOr<std::string> BuildMessageCheckpoint(
    std::vector<std::string> serialized_messages,
    std::vector<std::string> event_times,
    absl::string_view on_device_query_name) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> builder = builder_factory.Create();

  std::string entry_tensor_name =
      absl::StrCat(on_device_query_name, "/", kPrivateLoggerEntryKey);

  // Create entry tensor
  FCP_ASSIGN_OR_RETURN(
      auto entry_tensor,
      Tensor::Create(DataType::DT_STRING,
                     {static_cast<int64_t>(serialized_messages.size())},
                     CreateStringTestData(std::move(serialized_messages))));
  FCP_RETURN_IF_ERROR(builder->Add(entry_tensor_name, std::move(entry_tensor)));

  // Create event_time tensor
  std::string event_time_tensor_name =
      absl::StrCat(on_device_query_name, "/", kEventTimeColumnName);
  FCP_ASSIGN_OR_RETURN(
      auto time_tensor,
      Tensor::Create(DataType::DT_STRING,
                     {static_cast<int64_t>(event_times.size())},
                     CreateStringTestData(std::move(event_times))));
  FCP_RETURN_IF_ERROR(
      builder->Add(event_time_tensor_name, std::move(time_tensor)));

  FCP_ASSIGN_OR_RETURN(absl::Cord checkpoint_cord, builder->Build());
  return std::string(checkpoint_cord);
}

}  // namespace confidential_federated_compute::fed_sql::testing
