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

#include "containers/fed_sql/range_tracker.h"

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "testing/matchers.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::fed_sql {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

TEST(RangeTrackerTest, AddRanges) {
  RangeTracker range_tracker;
  EXPECT_TRUE(range_tracker.AddRange(1, 4));
  EXPECT_TRUE(range_tracker.AddRange(4, 5));
  EXPECT_TRUE(range_tracker.AddRange(0, 1));
  EXPECT_TRUE(range_tracker.AddRange(8, 10));
  EXPECT_THAT(range_tracker.GetRanges(),
              ElementsAre(Interval<uint64_t>(0, 5), Interval<uint64_t>(8, 10)));

  EXPECT_TRUE(range_tracker.AddRange(5, 8));
  EXPECT_THAT(range_tracker.GetRanges(),
              ElementsAre(Interval<uint64_t>(0, 10)));
}

TEST(RangeTrackerTest, AddKeys) {
  RangeTracker range_tracker;
  range_tracker.AddKey("foo");
  range_tracker.AddKey("bar");
  EXPECT_THAT(range_tracker.GetKeys(), UnorderedElementsAre("foo", "bar"));
  range_tracker.AddKey("foo");
  EXPECT_THAT(range_tracker.GetKeys(), UnorderedElementsAre("foo", "bar"));
  range_tracker.AddKey("baz");
  EXPECT_THAT(range_tracker.GetKeys(),
              UnorderedElementsAre("foo", "bar", "baz"));
}

TEST(RangeTrackerTest, AddOverlappingRanges) {
  RangeTracker range_tracker;
  EXPECT_TRUE(range_tracker.AddRange(1, 4));
  EXPECT_FALSE(range_tracker.AddRange(1, 4));
  EXPECT_FALSE(range_tracker.AddRange(3, 5));
}

TEST(RangeTrackerTest, Merge) {
  RangeTrackerState state1 =
      PARSE_TEXT_PROTO(R"pb(
        keys: "foo" keys: "bar" values: 0 values: 5
      )pb");
  RangeTrackerState state2 =
      PARSE_TEXT_PROTO(R"pb(
        keys: "foo" keys: "baz" values: 7 values: 9
      )pb");
  RangeTrackerState state3 =
      PARSE_TEXT_PROTO(R"pb(
        keys: "baz" values: 5 values: 6
      )pb");

  auto range_tracker1 = RangeTracker::Parse(state1);
  auto range_tracker2 = RangeTracker::Parse(state2);
  auto range_tracker3 = RangeTracker::Parse(state3);
  EXPECT_THAT(range_tracker1, IsOk());
  EXPECT_THAT(range_tracker2, IsOk());
  EXPECT_THAT(range_tracker3, IsOk());

  EXPECT_TRUE(range_tracker1->Merge(*range_tracker2));
  EXPECT_TRUE(range_tracker1->Merge(*range_tracker3));
  EXPECT_THAT(range_tracker1->Serialize(),
              EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                keys: "foo"
                keys: "bar"
                keys: "baz"
                values: 0
                values: 6
                values: 7
                values: 9
              )pb"));
}

TEST(RangeTrackerTest, MergeOverlappingRanges) {
  RangeTrackerState state1 =
      PARSE_TEXT_PROTO(R"pb(
        keys: "foo" values: 1 values: 5
      )pb");
  RangeTrackerState state2 = PARSE_TEXT_PROTO(R"pb(
    values: 4 values: 6
  )pb");
  auto range_tracker1 = RangeTracker::Parse(state1);
  auto range_tracker2 = RangeTracker::Parse(state2);
  EXPECT_THAT(range_tracker1, IsOk());
  EXPECT_THAT(range_tracker2, IsOk());
  EXPECT_FALSE(range_tracker1->Merge(*range_tracker2));
}

TEST(RangeTrackerTest, StringParseAndSerialize) {
  RangeTrackerState state = PARSE_TEXT_PROTO(R"pb(
    keys: "foo"
    keys: "bar"
    keys: "baz"
    values: 1
    values: 5
    values: 7
    values: 9
  )pb");
  std::string serialized_state = state.SerializeAsString();
  auto range_tracker = RangeTracker::Parse(serialized_state);
  EXPECT_THAT(range_tracker, IsOk());
  std::string new_serialized_state = range_tracker->SerializeAsString();
  // The new serialized state may not be the same due to the repeated
  // field reordering, but the size should be the same.
  EXPECT_EQ(serialized_state.size(), new_serialized_state.size());
  // Make sure we can parse again
  EXPECT_THAT(RangeTracker::Parse(new_serialized_state), IsOk());
}

TEST(RangeTrackerTest, StringParseBadInput) {
  EXPECT_THAT(RangeTracker::Parse("foobar"),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(RangeTrackerTest, ParseUnexpectedNumberOfValues) {
  RangeTrackerState state = PARSE_TEXT_PROTO(R"pb(
    values: 1
  )pb");
  EXPECT_THAT(RangeTracker::Parse(state),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(RangeTrackerTest, ParseUnexpectedOrderOfValues) {
  RangeTrackerState state = PARSE_TEXT_PROTO(R"pb(
    values: 2 values: 1
  )pb");
  EXPECT_THAT(RangeTracker::Parse(state),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(RangeTrackerTest, ParseUnexpectedOrderOfIntervals) {
  RangeTrackerState state =
      PARSE_TEXT_PROTO(R"pb(
        values: 2 values: 3 values: 0 values: 1
      )pb");
  EXPECT_THAT(RangeTracker::Parse(state),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(RangeTrackerTest, BundleEmptyRangeTracker) {
  RangeTracker range_tracker;
  EXPECT_EQ(BundleRangeTracker("foo", range_tracker),
            absl::StrCat(kRangeTrackerBundleSignature, std::string(1, '\0'),
                         std::string(1, '\x3'), "foo"));
}

TEST(RangeTrackerTest, UnbundleEmptyRangeTracker) {
  std::string blob =
      absl::StrCat(kRangeTrackerBundleSignature, std::string(1, '\0'),
                   std::string(1, '\x3'), "foo");
  auto range_tracker = UnbundleRangeTracker(blob);
  EXPECT_EQ(blob, "foo");
  EXPECT_TRUE(range_tracker->GetKeys().empty());
  EXPECT_TRUE(range_tracker->GetRanges().empty());
}

TEST(RangeTrackerTest, BundleAndUnbundleSuccess) {
  RangeTrackerState state =
      PARSE_TEXT_PROTO(R"pb(
        keys: "foo" keys: "bar" values: 1 values: 10
      )pb");
  auto range_tracker = RangeTracker::Parse(state);
  EXPECT_THAT(range_tracker, IsOk());

  std::string bundle = BundleRangeTracker("foobar", *range_tracker);

  size_t expected_bundle_size =
      sizeof(kRangeTrackerBundleSignature) - 1  // size of signature
      + 1 +
      range_tracker->SerializeAsString()
          .size()                          // range tracker state + its size
      + 1 + std::string("foobar").size();  // blob + its size
  EXPECT_EQ(bundle.size(), expected_bundle_size);

  auto unbundled_range_tracker = UnbundleRangeTracker(bundle);
  EXPECT_THAT(unbundled_range_tracker, IsOk());
  EXPECT_EQ(bundle, "foobar");
  EXPECT_THAT(unbundled_range_tracker->Serialize(),
              EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                keys: "foo"
                keys: "bar"
                values: 1
                values: 10
              )pb"));
}

TEST(RangeTrackerTest, UnbundleSignatureMimatch) {
  std::string bundle1;
  EXPECT_THAT(UnbundleRangeTracker(bundle1),
              StatusIs(absl::StatusCode::kInvalidArgument));
  std::string bundle2("abcd");
  EXPECT_THAT(UnbundleRangeTracker(bundle2),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(RangeTrackerTest, UnbundleMissingStateSize) {
  std::string bundle(kRangeTrackerBundleSignature);
  EXPECT_THAT(UnbundleRangeTracker(bundle),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(RangeTrackerTest, UnbundleIncompleteState) {
  std::string bundle(absl::StrCat(kRangeTrackerBundleSignature, "\x06", "ab"));
  EXPECT_THAT(UnbundleRangeTracker(bundle),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(RangeTrackerTest, UnbundleMissingPayloadSize) {
  std::string bundle =
      absl::StrCat(kRangeTrackerBundleSignature, std::string(1, '\0'));
  EXPECT_THAT(UnbundleRangeTracker(bundle),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(RangeTrackerTest, UnbundleIncompletePayload) {
  std::string bundle =
      absl::StrCat(kRangeTrackerBundleSignature, std::string(1, '\0'),
                   std::string(1, '\x6'), "ab");
  EXPECT_THAT(UnbundleRangeTracker(bundle),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(RangeTrackerTest, SerializeWithPartitionIndex) {
  RangeTracker range_tracker;
  range_tracker.AddKey("foo");
  range_tracker.AddKey("bar");
  range_tracker.AddKey("baz");
  EXPECT_TRUE(range_tracker.AddRange(1, 4));
  EXPECT_TRUE(range_tracker.AddRange(4, 5));
  EXPECT_TRUE(range_tracker.AddRange(0, 1));
  EXPECT_TRUE(range_tracker.AddRange(8, 10));
  range_tracker.SetPartitionIndex(123);

  EXPECT_THAT(range_tracker.Serialize(),
              EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                keys: "foo"
                keys: "bar"
                keys: "baz"
                values: 0
                values: 5
                values: 8
                values: 10
                partition_index: 123
              )pb"));
}

TEST(RangeTrackerTest, MergeSamePartitionIndex) {
  RangeTrackerState state1 = PARSE_TEXT_PROTO(R"pb(
    keys: "foo"
    keys: "bar"
    values: 0
    values: 5
    partition_index: 123
    expired_keys: "expired_key1"
    expired_keys: "expired_key2"
  )pb");
  RangeTrackerState state2 = PARSE_TEXT_PROTO(R"pb(
    keys: "foo"
    keys: "baz"
    values: 7
    values: 9
    partition_index: 123
    expired_keys: "expired_key2"
    expired_keys: "expired_key1"
  )pb");
  auto range_tracker1 = RangeTracker::Parse(state1);
  auto range_tracker2 = RangeTracker::Parse(state2);
  EXPECT_THAT(range_tracker1, IsOk());
  EXPECT_THAT(range_tracker2, IsOk());

  EXPECT_TRUE(range_tracker1->Merge(*range_tracker2));
  EXPECT_THAT(range_tracker1->Serialize(),
              EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                keys: "foo"
                keys: "bar"
                keys: "baz"
                values: 0
                values: 5
                values: 7
                values: 9
                partition_index: 123
                expired_keys: "expired_key1"
                expired_keys: "expired_key2"
              )pb"));
}

TEST(RangeTrackerTest, MergeDifferentPartitions) {
  RangeTrackerState state1 =
      PARSE_TEXT_PROTO(R"pb(
        keys: "foo" values: 1 values: 5 partition_index: 123
      )pb");
  RangeTrackerState state2 =
      PARSE_TEXT_PROTO(R"pb(
        keys: "foo" values: 7 values: 9 partition_index: 456
      )pb");
  auto range_tracker1 = RangeTracker::Parse(state1);
  auto range_tracker2 = RangeTracker::Parse(state2);
  EXPECT_THAT(range_tracker1, IsOk());
  EXPECT_THAT(range_tracker2, IsOk());
  EXPECT_FALSE(range_tracker1->Merge(*range_tracker2));
}

TEST(RangeTrackerTest, MergeDifferentExpiredKeys) {
  RangeTrackerState state1 = PARSE_TEXT_PROTO(R"pb(
    keys: "foo"
    values: 0
    values: 5
    partition_index: 123
    expired_keys: "expired_key1"
  )pb");
  RangeTrackerState state2 = PARSE_TEXT_PROTO(R"pb(
    keys: "bar"
    values: 5
    values: 10
    partition_index: 123
    expired_keys: "expired_key2"
  )pb");
  auto range_tracker1 = RangeTracker::Parse(state1);
  auto range_tracker2 = RangeTracker::Parse(state2);
  EXPECT_THAT(range_tracker1, IsOk());
  EXPECT_THAT(range_tracker2, IsOk());
  EXPECT_TRUE(range_tracker1->Merge(*range_tracker2));

  EXPECT_THAT(range_tracker1->Serialize(),
              EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                keys: "foo"
                keys: "bar"
                values: 0
                values: 10
                partition_index: 123
                expired_keys: "expired_key1"
                expired_keys: "expired_key2"
              )pb"));
}

TEST(RangeTrackerTest, MergeWithEmptyRangeTracker) {
  RangeTrackerState state = PARSE_TEXT_PROTO(R"pb(
    keys: "foo"
    keys: "bar"
    values: 0
    values: 5
    partition_index: 123
    expired_keys: "expired_key1"
    expired_keys: "expired_key2"
  )pb");
  RangeTracker range_tracker;
  auto other_range_tracker = RangeTracker::Parse(state);
  EXPECT_THAT(other_range_tracker, IsOk());

  EXPECT_TRUE(range_tracker.Merge(*other_range_tracker));
  EXPECT_THAT(range_tracker.Serialize(),
              EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                keys: "foo"
                keys: "bar"
                values: 0
                values: 5
                partition_index: 123
                expired_keys: "expired_key1"
                expired_keys: "expired_key2"
              )pb"));
}

TEST(RangeTrackerTest, ExpiredKeys) {
  RangeTracker range_tracker;
  range_tracker.SetExpiredKeys({"expired_key1", "expired_key2"});
  range_tracker.AddKey("foo");
  range_tracker.AddKey("bar");
  EXPECT_TRUE(range_tracker.AddRange(1, 4));
  EXPECT_TRUE(range_tracker.AddRange(0, 1));

  RangeTrackerState serialized_state = range_tracker.Serialize();
  EXPECT_THAT(serialized_state, EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                keys: "foo"
                keys: "bar"
                values: 0
                values: 4
                expired_keys: "expired_key1"
                expired_keys: "expired_key2"
              )pb"));

  RangeTracker parsed_range_tracker =
      RangeTracker::Parse(serialized_state).value();
  auto expired_keys = parsed_range_tracker.GetExpiredKeys();
  EXPECT_THAT(expired_keys.size(), 2);
  EXPECT_TRUE(expired_keys.contains("expired_key1"));
  EXPECT_TRUE(expired_keys.contains("expired_key2"));
}

}  // namespace
}  // namespace confidential_federated_compute::fed_sql