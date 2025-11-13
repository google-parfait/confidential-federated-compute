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
// Helper functions and classes for FedSQL unit tests.
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_TEST_UTILS_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_TEST_UTILS_H_

#include <string>
#include <vector>

#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.pb.h"
#include "google/protobuf/dynamic_message.h"
#include "google/protobuf/message.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace confidential_federated_compute::fed_sql::testing {

std::unique_ptr<tensorflow_federated::aggregation::MutableStringData>
CreateStringTestData(std::vector<std::string> data);

std::string BuildFedSqlGroupByCheckpoint(
    std::initializer_list<uint64_t> key_col_values,
    std::initializer_list<uint64_t> val_col_values,
    const std::string& key_col_name = "key",
    const std::string& val_col_name = "val");

// A helper class to create test protobuf messages.
// The test proto message is defined as:
//
// message TestMessage {
//   int64 key = 1;
//   int64 val = 2;
// }
class MessageHelper {
 public:
  MessageHelper();
  ~MessageHelper() = default;

  std::unique_ptr<google::protobuf::Message> CreateMessage(int64_t key,
                                                           int64_t val);

  const google::protobuf::Message* prototype() const;
  const google::protobuf::FileDescriptor* file_descriptor() const;
  absl::string_view message_name() const;

 private:
  google::protobuf::DescriptorPool pool_;
  const google::protobuf::Descriptor* descriptor_ = nullptr;
  google::protobuf::DynamicMessageFactory factory_;
  const google::protobuf::Message* prototype_ = nullptr;
};

// Helper to create a checkpoint with a serialized message tensor and an
// event_time tensor.
absl::StatusOr<std::string> BuildMessageCheckpoint(
    std::vector<std::string> serialized_messages,
    std::vector<std::string> event_times);

}  // namespace confidential_federated_compute::fed_sql::testing

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_TEST_UTILS_H_
