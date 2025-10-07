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
#include "program_executor_tee/program_context/cc/test_helpers.h"

#include <string>
#include <vector>

#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::program_executor_tee {

tensorflow_federated::v0::GetExecutorRequest GetExecutorRequest(
    int num_clients) {
  tensorflow_federated::v0::GetExecutorRequest get_executor_request;
  auto cardinalities = get_executor_request.mutable_cardinalities()->Add();
  cardinalities->mutable_placement()->set_uri("clients");
  cardinalities->set_cardinality(num_clients);
  return get_executor_request;
}

tensorflow_federated::v0::CreateValueRequest CreateIntValueRequest(
    std::string executor_id, int value) {
  tensorflow_federated::v0::Value value_pb;
  value_pb.mutable_array()->mutable_int32_list()->add_value(value);
  tensorflow_federated::v0::CreateValueRequest create_value_request;
  create_value_request.mutable_executor()->set_id(executor_id);
  *create_value_request.mutable_value() = std::move(value_pb);
  return create_value_request;
}

tensorflow_federated::v0::CreateValueRequest CreateIntrinsicValueRequest(
    std::string executor_id, std::string intrinsic_uri) {
  tensorflow_federated::v0::Value intrinsic_comp;
  intrinsic_comp.mutable_computation()->mutable_intrinsic()->set_uri(
      intrinsic_uri);
  tensorflow_federated::v0::CreateValueRequest create_value_request;
  create_value_request.mutable_executor()->set_id(executor_id);
  *create_value_request.mutable_value() = std::move(intrinsic_comp);
  return create_value_request;
}

tensorflow_federated::v0::CreateStructRequest CreateStructRequest(
    std::string executor_id, std::vector<std::string> ref_ids) {
  tensorflow_federated::v0::CreateStructRequest create_struct_request;
  create_struct_request.mutable_executor()->set_id(executor_id);
  for (const std::string& ref_id : ref_ids) {
    create_struct_request.add_element()->mutable_value_ref()->set_id(ref_id);
  }
  return create_struct_request;
}

tensorflow_federated::v0::CreateSelectionRequest CreateSelectionRequest(
    std::string executor_id, std::string source_ref_id, int index) {
  tensorflow_federated::v0::CreateSelectionRequest create_selection_request;
  create_selection_request.mutable_executor()->set_id(executor_id);
  create_selection_request.mutable_source_ref()->set_id(source_ref_id);
  create_selection_request.set_index(index);
  return create_selection_request;
}

tensorflow_federated::v0::CreateCallRequest CreateCallRequest(
    std::string executor_id, std::string function_ref_id,
    std::string arg_ref_id) {
  tensorflow_federated::v0::CreateCallRequest create_call_request;
  create_call_request.mutable_executor()->set_id(executor_id);
  create_call_request.mutable_function_ref()->set_id(function_ref_id);
  create_call_request.mutable_argument_ref()->set_id(arg_ref_id);
  return create_call_request;
}

tensorflow_federated::v0::ComputeRequest ComputeRequest(std::string executor_id,
                                                        std::string ref_id) {
  tensorflow_federated::v0::ComputeRequest compute_request;
  compute_request.mutable_executor()->set_id(executor_id);
  compute_request.mutable_value_ref()->set_id(ref_id);
  return compute_request;
}

tensorflow_federated::v0::DisposeRequest DisposeRequest(
    std::string executor_id, std::vector<std::string> ref_ids) {
  tensorflow_federated::v0::DisposeRequest dispose_request;
  dispose_request.mutable_executor()->set_id(executor_id);
  for (const std::string& ref_id : ref_ids) {
    dispose_request.add_value_ref()->set_id(ref_id);
  }
  return dispose_request;
}

tensorflow_federated::v0::DisposeExecutorRequest DisposeExecutorRequest(
    std::string executor_id) {
  tensorflow_federated::v0::DisposeExecutorRequest dispose_executor_request;
  dispose_executor_request.mutable_executor()->set_id(executor_id);
  return dispose_executor_request;
}

}  // namespace confidential_federated_compute::program_executor_tee
