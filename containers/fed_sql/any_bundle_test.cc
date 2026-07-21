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

#include "containers/fed_sql/any_bundle.h"

#include <string>

#include "absl/strings/cord.h"
#include "containers/common/time_budget/budget.pb.h"
#include "containers/fed_sql/range_tracker.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute::fed_sql {
namespace {

TEST(AnyBundleTest, BundleAndUnbundleSuccess) {
  RangeTrackerState state;
  state.add_keys("key1");
  state.add_values(10);
  state.add_values(20);

  absl::Cord payload("Some payload data");
  absl::Cord bundled = BundleAny(state, payload);

  RangeTrackerState unbundled_state;
  absl::Cord unbundled_payload = bundled;
  EXPECT_TRUE(UnbundleAny(unbundled_state, unbundled_payload));

  EXPECT_EQ(unbundled_state.keys(0), "key1");
  EXPECT_EQ(unbundled_state.values(0), 10);
  EXPECT_EQ(unbundled_state.values(1), 20);
  EXPECT_EQ(std::string(unbundled_payload), "Some payload data");
}

TEST(AnyBundleTest, UnbundleMismatchedMessageType) {
  RangeTrackerState state;
  state.add_keys("key1");
  absl::Cord bundled = BundleAny(state, absl::Cord("data"));

  BudgetState mismatched_state;
  absl::Cord unbundled_payload = bundled;
  EXPECT_FALSE(UnbundleAny(mismatched_state, unbundled_payload));
}

TEST(AnyBundleTest, UnbundleMissingAnySize) {
  RangeTrackerState state;
  absl::Cord unbundled_payload("");
  EXPECT_FALSE(UnbundleAny(state, unbundled_payload));
}

TEST(AnyBundleTest, UnbundleInsufficientAnyData) {
  RangeTrackerState state;
  state.add_keys("key1");
  absl::Cord bundled = BundleAny(state, absl::Cord("data"));

  // Truncate before Any message finished
  absl::Cord truncated = bundled.Subcord(0, bundled.size() - 10);
  EXPECT_FALSE(UnbundleAny(state, truncated));
}

TEST(AnyBundleTest, UnbundleMissingPayloadSize) {
  RangeTrackerState state;
  state.add_keys("key1");
  google::protobuf::Any any;
  any.PackFrom(state);
  std::string any_serialized = any.SerializeAsString();

  std::string prefix;
  {
    google::protobuf::io::StringOutputStream stream(&prefix);
    google::protobuf::io::CodedOutputStream coded_stream(&stream);
    coded_stream.WriteVarint64(any_serialized.size());
    coded_stream.WriteString(any_serialized);
    // Missing payload size
  }

  absl::Cord bundled(prefix);
  absl::Cord unbundled_payload = bundled;
  EXPECT_FALSE(UnbundleAny(state, unbundled_payload));
}

TEST(AnyBundleTest, UnbundleIncompletePayload) {
  RangeTrackerState state;
  state.add_keys("key1");
  absl::Cord payload("payload");
  absl::Cord bundled = BundleAny(state, payload);

  absl::Cord truncated = bundled.Subcord(0, bundled.size() - 1);
  EXPECT_FALSE(UnbundleAny(state, truncated));
}

}  // namespace
}  // namespace confidential_federated_compute::fed_sql
