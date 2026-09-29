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

#include "containers/common/io/any_bundle.h"

#include <string>

#include "absl/strings/cord.h"
#include "gmock/gmock.h"
#include "google/protobuf/any.pb.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/struct.pb.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute {
namespace {

using ::google::protobuf::ListValue;
using ::google::protobuf::Struct;

Struct CreateTestStruct() {
  Struct state;
  (*state.mutable_fields())["key1"].set_number_value(10);
  (*state.mutable_fields())["key2"].set_string_value("value2");
  return state;
}

TEST(AnyBundleTest, BundleAndUnbundleSuccess) {
  Struct state = CreateTestStruct();

  absl::Cord payload("Some payload data");
  absl::Cord bundled = BundleAny(state, payload);

  Struct unbundled_state;
  absl::Cord unbundled_payload = bundled;
  EXPECT_TRUE(UnbundleAny(unbundled_state, unbundled_payload));

  EXPECT_EQ(unbundled_state.fields().at("key1").number_value(), 10);
  EXPECT_EQ(unbundled_state.fields().at("key2").string_value(), "value2");
  EXPECT_EQ(std::string(unbundled_payload), "Some payload data");
}

TEST(AnyBundleTest, UnbundleMismatchedMessageType) {
  absl::Cord bundled = BundleAny(CreateTestStruct(), absl::Cord("data"));

  ListValue mismatched_state;
  absl::Cord unbundled_payload = bundled;
  EXPECT_FALSE(UnbundleAny(mismatched_state, unbundled_payload));
}

TEST(AnyBundleTest, UnbundleMissingAnySize) {
  Struct state;
  absl::Cord unbundled_payload("");
  EXPECT_FALSE(UnbundleAny(state, unbundled_payload));
}

TEST(AnyBundleTest, UnbundleInsufficientAnyData) {
  Struct state = CreateTestStruct();
  absl::Cord bundled = BundleAny(state, absl::Cord("data"));

  // Truncate before Any message finished
  absl::Cord truncated = bundled.Subcord(0, bundled.size() - 10);
  EXPECT_FALSE(UnbundleAny(state, truncated));
}

TEST(AnyBundleTest, UnbundleMissingPayloadSize) {
  Struct state = CreateTestStruct();
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
  Struct state = CreateTestStruct();
  absl::Cord payload("payload");
  absl::Cord bundled = BundleAny(state, payload);

  absl::Cord truncated = bundled.Subcord(0, bundled.size() - 1);
  EXPECT_FALSE(UnbundleAny(state, truncated));
}

}  // namespace
}  // namespace confidential_federated_compute
