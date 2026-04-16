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

#include "pi_client.h"

#include <string>
#include <utility>

#include "absl/status/status.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute::private_inference {
namespace {

TEST(PiClientTest, CreatePiClientFailsWhenServerIsUnreachable) {
  auto client_or = CreatePiClient(
      "localhost:12345",
      ::private_inference::proto::FEATURE_NAME_PSI_MEMORY_GENERATION);

  // CreatePiClient now performs the handshake, so it should fail if
  // unreachable.
  EXPECT_FALSE(client_or.ok());
  EXPECT_THAT(client_or.status().message(),
              testing::AnyOf(
                  testing::HasSubstr("ExchangeHandshakeMessages failed"),
                  testing::HasSubstr("Failed to open verification keys file"),
                  testing::HasSubstr("ClientSession::Create failed")));
}

}  // namespace
}  // namespace confidential_federated_compute::private_inference
