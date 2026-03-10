// Copyright 2026 Google LLC.
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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINTERS_BATCHED_INFERENCE_BATCHED_INFERENCE_TEST_UTILS_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINTERS_BATCHED_INFERENCE_BATCHED_INFERENCE_TEST_UTILS_H_

#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"

namespace confidential_federated_compute::batched_inference::testing {

using ::fcp::confidentialcompute::InferenceConfiguration;
using ::fcp::confidentialcompute::StreamInitializeRequest;

InferenceConfiguration GetInferenceConfigForTest();

void AddInitConfigForTest(StreamInitializeRequest* init_request);

std::string GetPrivateInferenceInputCheckpointForTest(
    std::vector<std::string> prompts);

std::string GetPrivateInferenceOutputCheckpointForTest(
    std::vector<std::string> prompts, std::vector<std::string> results);

}  // namespace confidential_federated_compute::batched_inference::testing

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINTERS_BATCHED_INFERENCE_BATCHED_INFERENCE_TEST_UTILS_H_
