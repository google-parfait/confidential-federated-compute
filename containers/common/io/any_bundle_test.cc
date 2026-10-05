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
#include "absl/strings/string_view.h"
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

TEST(AnyBundleTest, UnbundleFailureLeavesCordUnchanged) {
  absl::Cord bundled = BundleAny(CreateTestStruct(), absl::Cord("data"));
  absl::Cord data = bundled;
  ListValue mismatched_state;
  EXPECT_FALSE(UnbundleAny(mismatched_state, data));
  EXPECT_EQ(data, bundled);
}

TEST(AnyBundleTest, StringBundleAndUnbundleSuccess) {
  std::string data(BundleAny(CreateTestStruct(), absl::Cord("payload")));

  Struct unbundled_state;
  EXPECT_TRUE(UnbundleAny(unbundled_state, data));
  EXPECT_EQ(unbundled_state.fields().at("key1").number_value(), 10);
  EXPECT_EQ(data, "payload");
}

TEST(AnyBundleTest, StringUnbundleEmptyPayload) {
  std::string data(BundleAny(CreateTestStruct(), absl::Cord()));

  Struct unbundled_state;
  EXPECT_TRUE(UnbundleAny(unbundled_state, data));
  EXPECT_EQ(data, "");
}

TEST(AnyBundleTest, StringUnbundleMismatchedTypeLeavesDataUnchanged) {
  std::string bundled(BundleAny(CreateTestStruct(), absl::Cord("data")));
  std::string data = bundled;

  ListValue mismatched_state;
  EXPECT_FALSE(UnbundleAny(mismatched_state, data));
  EXPECT_EQ(data, bundled);
}

TEST(AnyBundleTest, StringUnbundleNotABundleLeavesDataUnchanged) {
  std::string data = "FCv1 this is not a bundle";
  Struct state;
  EXPECT_FALSE(UnbundleAny(state, data));
  EXPECT_EQ(data, "FCv1 this is not a bundle");
}

TEST(AnyBundleTest, StringUnbundleTruncatedPayloadLeavesDataUnchanged) {
  std::string bundled(BundleAny(CreateTestStruct(), absl::Cord("payload")));
  std::string data = bundled.substr(0, bundled.size() - 1);
  std::string original = data;

  Struct state;
  EXPECT_FALSE(UnbundleAny(state, data));
  EXPECT_EQ(data, original);
}

TEST(AnyBundleTest, ParseDelimitedAnyAdvancesInputToPayload) {
  std::string bundled(BundleAny(CreateTestStruct(), absl::Cord("payload")));
  absl::string_view input = bundled;

  Struct state;
  EXPECT_TRUE(internal::ParseDelimitedAny(state, input));
  EXPECT_EQ(state.fields().at("key1").number_value(), 10);
  EXPECT_EQ(input, "payload");
}

TEST(AnyBundleTest, ParseDelimitedAnyMismatchedTypeLeavesInputUnchanged) {
  std::string bundled(BundleAny(CreateTestStruct(), absl::Cord("payload")));
  absl::string_view input = bundled;

  ListValue mismatched_state;
  EXPECT_FALSE(internal::ParseDelimitedAny(mismatched_state, input));
  EXPECT_EQ(input, bundled);
}

TEST(AnyBundleTest, ParseDelimitedAnyTruncatedLeavesInputUnchanged) {
  std::string bundled(BundleAny(CreateTestStruct(), absl::Cord("payload")));
  std::string truncated = bundled.substr(0, bundled.size() - 1);
  absl::string_view input = truncated;

  Struct state;
  EXPECT_FALSE(internal::ParseDelimitedAny(state, input));
  EXPECT_EQ(input, truncated);
}

// Binary payload large enough to need multi-byte size varints, containing
// NUL bytes and bytes that look like varint continuation bytes.
std::string CreateBinaryPayload() {
  std::string payload;
  for (int i = 0; i < 100000; ++i) {
    payload.push_back(static_cast<char>(i % 256));
  }
  return payload;
}

TEST(AnyBundleTest, CordRoundTripLargeBinaryPayload) {
  Struct state = CreateTestStruct();
  std::string payload = CreateBinaryPayload();
  // Build the payload from several chunks so that the bundle isn't flat.
  absl::Cord payload_cord;
  for (size_t i = 0; i < payload.size(); i += 4096) {
    payload_cord.Append(payload.substr(i, 4096));
  }

  absl::Cord data = BundleAny(state, payload_cord);
  EXPECT_GT(data.size(), payload.size());

  Struct unbundled_state;
  ASSERT_TRUE(UnbundleAny(unbundled_state, data));
  EXPECT_EQ(unbundled_state.fields_size(), 2);
  EXPECT_EQ(unbundled_state.fields().at("key1").number_value(), 10);
  EXPECT_EQ(unbundled_state.fields().at("key2").string_value(), "value2");
  EXPECT_EQ(data, payload);
}

TEST(AnyBundleTest, StringRoundTripLargeBinaryPayloadInPlace) {
  Struct state = CreateTestStruct();
  std::string payload = CreateBinaryPayload();

  std::string data(BundleAny(state, absl::Cord(payload)));
  EXPECT_GT(data.size(), payload.size());
  const char* buffer = data.data();

  Struct unbundled_state;
  ASSERT_TRUE(UnbundleAny(unbundled_state, data));
  EXPECT_EQ(unbundled_state.fields_size(), 2);
  EXPECT_EQ(unbundled_state.fields().at("key1").number_value(), 10);
  EXPECT_EQ(unbundled_state.fields().at("key2").string_value(), "value2");
  EXPECT_EQ(data, payload);
  // The header was removed in place; the buffer was not reallocated.
  EXPECT_EQ(data.data(), buffer);
}

}  // namespace
}  // namespace confidential_federated_compute
