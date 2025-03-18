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

#include "containers/private_state.h"

#include <string>

#include "absl/status/status.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "testing/matchers.h"

namespace confidential_federated_compute {
namespace {

TEST(PrivateStateTest, BundleSuccess) {
  PrivateState private_state{};
  std::string bundle = BundlePrivateState("foobar", private_state);
  std::string expected_content =
      absl::StrCat(kPrivateStateBundleSignature, "\x06", "foobar");
  EXPECT_EQ(bundle, expected_content);
}

TEST(PrivateStateTest, UnbundleSuccess) {
  std::string bundle =
      absl::StrCat(kPrivateStateBundleSignature, "\x06", "foobar");
  EXPECT_THAT(UnbundlePrivateState(bundle), IsOk());
  EXPECT_EQ(bundle, "foobar");
}

TEST(PrivateStateTest, UnbundleSignatureMimatch) {
  std::string bundle1;
  EXPECT_THAT(UnbundlePrivateState(bundle1),
              IsCode(absl::StatusCode::kInvalidArgument));
  std::string bundle2("abcd");
  EXPECT_THAT(UnbundlePrivateState(bundle2),
              IsCode(absl::StatusCode::kInvalidArgument));
}

TEST(PrivateStateTest, UnbundleMissingPayloadSize) {
  std::string bundle(kPrivateStateBundleSignature);
  EXPECT_THAT(UnbundlePrivateState(bundle),
              IsCode(absl::StatusCode::kInvalidArgument));
}

TEST(PrivateStateTest, UnbundleIncompletePayload) {
  std::string bundle(absl::StrCat(kPrivateStateBundleSignature, "\x06", "ab"));
  EXPECT_THAT(UnbundlePrivateState(bundle),
              IsCode(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace confidential_federated_compute