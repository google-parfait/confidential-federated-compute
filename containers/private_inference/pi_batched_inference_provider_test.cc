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

#include "pi_batched_inference_provider.h"

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute::private_inference {
namespace {

using ::fcp::confidentialcompute::InferenceConfiguration;

TEST(PiBatchedInferenceProviderTest, GetEngineFailsWhenPaicConfigMissing) {
  PiBatchedInferenceProvider provider("localhost:12345");
  InferenceConfiguration config;
  // Test with no paic_config, which should fail validation.
  auto engine = provider.GetEngineForInferenceConfig(config);
  EXPECT_EQ(engine, nullptr);
}

TEST(PiBatchedInferenceProviderTest,
     GetEngineWithPaicConfigFailsWhenServerUnreachable) {
  PiBatchedInferenceProvider provider("localhost:12345");
  InferenceConfiguration config;
  config.mutable_paic_config()->set_paic_feature_id(100);
  auto engine = provider.GetEngineForInferenceConfig(config);
  EXPECT_EQ(engine, nullptr);
}

}  // namespace
}  // namespace confidential_federated_compute::private_inference
