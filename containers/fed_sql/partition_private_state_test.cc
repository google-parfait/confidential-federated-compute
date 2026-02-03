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

#include "containers/fed_sql/partition_private_state.h"

#include "absl/status/status_matchers.h"
#include "containers/fed_sql/partition_private_state.pb.h"
#include "fcp/protos/confidentialcompute/fed_sql_container_config.pb.h"
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

TEST(PartitionPrivateStateTest, ParseAndSerialize) {
  PartitionPrivateStateProto proto = PARSE_TEXT_PROTO(R"pb(
    symmetric_keys { id: 1 symmetric_key: "key1" }
    symmetric_keys { id: 2 symmetric_key: "key2" }
    expired_keys: "expired_key1"
    expired_keys: "expired_key2"
    buckets { key: "foo" values: 1 values: 4 values: 7 values: 10 }
    buckets { key: "bar" values: 0 values: 3 }
  )pb");
  auto state = PartitionPrivateState::Parse(proto);
  EXPECT_THAT(state, IsOk());
  EXPECT_THAT(state->Serialize(), EqualsProtoIgnoringRepeatedFieldOrder(proto));
}

TEST(PartitionPrivateStateTest, ParseAndSerializeAsString) {
  PartitionPrivateStateProto proto = PARSE_TEXT_PROTO(R"pb(
    symmetric_keys { id: 1 symmetric_key: "key1" }
    symmetric_keys { id: 2 symmetric_key: "key2" }
    expired_keys: "expired_key1"
    expired_keys: "expired_key2"
    buckets { key: "foo" values: 1 values: 4 values: 7 values: 10 }
    buckets { key: "bar" values: 0 values: 3 }
  )pb");
  std::string serialized = proto.SerializeAsString();
  auto state = PartitionPrivateState::Parse(serialized);
  EXPECT_THAT(state, IsOk());
  std::string new_serialized = state->SerializeAsString();
  EXPECT_EQ(serialized.size(), new_serialized.size());
  EXPECT_THAT(PartitionPrivateState::Parse(new_serialized), IsOk());
}

TEST(PartitionPrivateStateTest, ParseInvalidString) {
  std::string invalid_data = "invalid_data";
  EXPECT_THAT(PartitionPrivateState::Parse(invalid_data),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(PartitionPrivateStateTest, ParseInvalidProto) {
  PartitionPrivateStateProto proto = PARSE_TEXT_PROTO(R"pb(
    buckets { key: "foo" values: 1 values: 4 values: 7 }
  )pb");
  EXPECT_THAT(PartitionPrivateState::Parse(proto),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(PartitionPrivateStateTest, GetSerializedKeys) {
  PartitionPrivateStateProto proto = PARSE_TEXT_PROTO(R"pb(
    symmetric_keys { id: 1 symmetric_key: "key1" }
    symmetric_keys { id: 2 symmetric_key: "key2" }
  )pb");
  auto private_state = PartitionPrivateState::Parse(proto).value();
  std::string serialized_keys = private_state.GetSerializedKeys();

  fcp::confidentialcompute::FedSqlContainerPartitionKeys serialized_keys_proto;
  serialized_keys_proto.ParseFromString(serialized_keys);
  EXPECT_THAT(serialized_keys_proto, EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                keys { partition_index: 1 symmetric_key: "key1" }
                keys { partition_index: 2 symmetric_key: "key2" }
              )pb"));
}

TEST(PartitionPrivateStateTest, AddPartition) {
  RangeTrackerState range_tracker_state_1 = PARSE_TEXT_PROTO(R"pb(
    buckets { key: "foo" values: 1 values: 5 values: 7 values: 10 }
    buckets { key: "bar" values: 0 values: 4 }
    partition_index: 123
    expired_keys: "expired_key1"
    expired_keys: "expired_key2"
  )pb");
  RangeTracker range_tracker_1 =
      RangeTracker::Parse(range_tracker_state_1).value();
  RangeTrackerState range_tracker_state_2 = PARSE_TEXT_PROTO(R"pb(
    buckets { key: "foo" values: 1 values: 5 values: 7 values: 10 }
    buckets { key: "bar" values: 0 values: 4 }
    partition_index: 456
    expired_keys: "expired_key1"
    expired_keys: "expired_key2"
  )pb");
  RangeTracker range_tracker_2 =
      RangeTracker::Parse(range_tracker_state_2).value();

  PartitionPrivateState state;
  EXPECT_TRUE(state.AddPartition(range_tracker_1, "symmetric_key1"));
  EXPECT_TRUE(state.AddPartition(range_tracker_2, "symmetric_key2"));
  EXPECT_THAT(state.Serialize(), EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                symmetric_keys { id: 123 symmetric_key: "symmetric_key1" }
                symmetric_keys { id: 456 symmetric_key: "symmetric_key2" }
                expired_keys: "expired_key1"
                expired_keys: "expired_key2"
                buckets { key: "foo" values: 1 values: 5 values: 7 values: 10 }
                buckets { key: "bar" values: 0 values: 4 }
              )pb"));
  RangeTracker::InnerMap per_key_ranges = state.GetPerKeyRanges();
  EXPECT_THAT(per_key_ranges,
              UnorderedElementsAre(
                  Pair(Eq("foo"), ElementsAre(Interval<uint64_t>(1, 5),
                                              Interval<uint64_t>(7, 10))),
                  Pair(Eq("bar"), ElementsAre(Interval<uint64_t>(0, 4)))));
  auto expired_keys = state.GetExpiredKeys();
  EXPECT_THAT(expired_keys.size(), 2);
  EXPECT_TRUE(expired_keys.contains("expired_key1"));
  EXPECT_TRUE(expired_keys.contains("expired_key2"));
}

TEST(PartitionPrivateStateTest, AddPartitionNoPartitionId) {
  RangeTracker range_tracker;
  range_tracker.AddRange("foo", 1, 5);
  range_tracker.SetExpiredKeys({"expired_key1"});
  // No partition index set.
  PartitionPrivateState state;
  EXPECT_FALSE(state.AddPartition(range_tracker, "symmetric_key1"));
}

TEST(PartitionPrivateStateTest, AddPartitionConflictingPartitionId) {
  RangeTrackerState range_tracker_state_1 = PARSE_TEXT_PROTO(R"pb(
    buckets { key: "foo" values: 1 values: 5 }
    partition_index: 123
    expired_keys: "expired_key1"
  )pb");
  RangeTracker range_tracker_1 =
      RangeTracker::Parse(range_tracker_state_1).value();
  RangeTrackerState range_tracker_state_2 = PARSE_TEXT_PROTO(R"pb(
    buckets { key: "foo" values: 1 values: 5 }
    partition_index: 123
    expired_keys: "expired_key1"
  )pb");
  RangeTracker range_tracker_2 =
      RangeTracker::Parse(range_tracker_state_2).value();

  PartitionPrivateState state;
  EXPECT_TRUE(state.AddPartition(range_tracker_1, "symmetric_key1"));
  EXPECT_FALSE(state.AddPartition(range_tracker_2, "symmetric_key2"));
}

TEST(PartitionPrivateStateTest, AddPartitionMismatchRangeValues) {
  RangeTrackerState range_tracker_state_1 = PARSE_TEXT_PROTO(R"pb(
    buckets { key: "foo" values: 1 values: 5 }
    partition_index: 123
    expired_keys: "expired_key1"
  )pb");
  RangeTracker range_tracker_1 =
      RangeTracker::Parse(range_tracker_state_1).value();
  RangeTrackerState range_tracker_state_2 = PARSE_TEXT_PROTO(R"pb(
    buckets { key: "foo" values: 1 values: 7 }
    partition_index: 456
    expired_keys: "expired_key1"
  )pb");
  RangeTracker range_tracker_2 =
      RangeTracker::Parse(range_tracker_state_2).value();

  PartitionPrivateState state;
  EXPECT_TRUE(state.AddPartition(range_tracker_1, "symmetric_key1"));
  EXPECT_FALSE(state.AddPartition(range_tracker_2, "symmetric_key2"));
}

TEST(PartitionPrivateStateTest, AddPartitionMismatchRangeKeys) {
  RangeTrackerState range_tracker_state_1 = PARSE_TEXT_PROTO(R"pb(
    buckets { key: "foo" values: 1 values: 5 }
    partition_index: 123
    expired_keys: "expired_key1"
  )pb");
  RangeTracker range_tracker_1 =
      RangeTracker::Parse(range_tracker_state_1).value();
  RangeTrackerState range_tracker_state_2 = PARSE_TEXT_PROTO(R"pb(
    buckets { key: "bar" values: 1 values: 5 }
    partition_index: 456
    expired_keys: "expired_key1"
  )pb");
  RangeTracker range_tracker_2 =
      RangeTracker::Parse(range_tracker_state_2).value();

  PartitionPrivateState state;
  EXPECT_TRUE(state.AddPartition(range_tracker_1, "symmetric_key1"));
  EXPECT_FALSE(state.AddPartition(range_tracker_2, "symmetric_key2"));
}

TEST(PartitionPrivateStateTest, AddPartitionMismatchExpiredKeys) {
  RangeTrackerState range_tracker_state_1 = PARSE_TEXT_PROTO(R"pb(
    buckets { key: "foo" values: 1 values: 5 }
    partition_index: 123
    expired_keys: "expired_key1"
  )pb");
  RangeTracker range_tracker_1 =
      RangeTracker::Parse(range_tracker_state_1).value();
  RangeTrackerState range_tracker_state_2 = PARSE_TEXT_PROTO(R"pb(
    buckets { key: "foo" values: 1 values: 5 }
    partition_index: 456
    expired_keys: "expired_key2"
  )pb");
  RangeTracker range_tracker_2 =
      RangeTracker::Parse(range_tracker_state_2).value();

  PartitionPrivateState state;
  EXPECT_TRUE(state.AddPartition(range_tracker_1, "symmetric_key1"));
  EXPECT_FALSE(state.AddPartition(range_tracker_2, "symmetric_key2"));
}

TEST(PartitionPrivateState, Merge) {
  PartitionPrivateStateProto proto_1 = PARSE_TEXT_PROTO(R"pb(
    symmetric_keys { id: 1 symmetric_key: "key1" }
    symmetric_keys { id: 2 symmetric_key: "key2" }
    expired_keys: "expired_key1"
    expired_keys: "expired_key2"
    buckets { key: "foo" values: 1 values: 4 values: 7 values: 10 }
    buckets { key: "bar" values: 0 values: 3 }
  )pb");
  PartitionPrivateStateProto proto_2 = PARSE_TEXT_PROTO(R"pb(
    symmetric_keys { id: 3 symmetric_key: "key3" }
    symmetric_keys { id: 4 symmetric_key: "key4" }
    expired_keys: "expired_key1"
    expired_keys: "expired_key2"
    buckets { key: "foo" values: 1 values: 4 values: 7 values: 10 }
    buckets { key: "bar" values: 0 values: 3 }
  )pb");

  PartitionPrivateState state;
  EXPECT_TRUE(state.Merge(PartitionPrivateState::Parse(proto_1).value()));
  EXPECT_TRUE(state.Merge(PartitionPrivateState::Parse(proto_2).value()));
  EXPECT_THAT(state.Serialize(), EqualsProtoIgnoringRepeatedFieldOrder(R"pb(
                symmetric_keys { id: 1 symmetric_key: "key1" }
                symmetric_keys { id: 2 symmetric_key: "key2" }
                symmetric_keys { id: 3 symmetric_key: "key3" }
                symmetric_keys { id: 4 symmetric_key: "key4" }
                expired_keys: "expired_key1"
                expired_keys: "expired_key2"
                buckets { key: "foo" values: 1 values: 4 values: 7 values: 10 }
                buckets { key: "bar" values: 0 values: 3 }
              )pb"));
}

TEST(PartitionPrivateState, MergeConflictingPartitionId) {
  PartitionPrivateStateProto proto_1 = PARSE_TEXT_PROTO(R"pb(
    symmetric_keys { id: 1 symmetric_key: "key1" }
    expired_keys: "expired_key1"
    buckets { key: "foo" values: 1 values: 5 }
  )pb");
  PartitionPrivateStateProto proto_2 = PARSE_TEXT_PROTO(R"pb(
    symmetric_keys { id: 1 symmetric_key: "key2" }
    expired_keys: "expired_key1"
    buckets { key: "foo" values: 1 values: 5 }
  )pb");

  PartitionPrivateState state;
  EXPECT_TRUE(state.Merge(PartitionPrivateState::Parse(proto_1).value()));
  EXPECT_FALSE(state.Merge(PartitionPrivateState::Parse(proto_2).value()));
}

TEST(PartitionPrivateState, MergeMismatchRangeValues) {
  PartitionPrivateStateProto proto_1 = PARSE_TEXT_PROTO(R"pb(
    symmetric_keys { id: 1 symmetric_key: "key1" }
    expired_keys: "expired_key1"
    buckets { key: "foo" values: 1 values: 5 }
  )pb");
  PartitionPrivateStateProto proto_2 = PARSE_TEXT_PROTO(R"pb(
    symmetric_keys { id: 2 symmetric_key: "key2" }
    expired_keys: "expired_key1"
    buckets { key: "foo" values: 1 values: 7 }
  )pb");

  PartitionPrivateState state;
  EXPECT_TRUE(state.Merge(PartitionPrivateState::Parse(proto_1).value()));
  EXPECT_FALSE(state.Merge(PartitionPrivateState::Parse(proto_2).value()));
}

TEST(PartitionPrivateState, MergeMismatchRangeKeys) {
  PartitionPrivateStateProto proto_1 = PARSE_TEXT_PROTO(R"pb(
    symmetric_keys { id: 1 symmetric_key: "key1" }
    expired_keys: "expired_key1"
    buckets { key: "foo" values: 1 values: 5 }
  )pb");
  PartitionPrivateStateProto proto_2 = PARSE_TEXT_PROTO(R"pb(
    symmetric_keys { id: 2 symmetric_key: "key2" }
    expired_keys: "expired_key1"
    buckets { key: "bar" values: 1 values: 5 }
  )pb");

  PartitionPrivateState state;
  EXPECT_TRUE(state.Merge(PartitionPrivateState::Parse(proto_1).value()));
  EXPECT_FALSE(state.Merge(PartitionPrivateState::Parse(proto_2).value()));
}

TEST(PartitionPrivateState, MergeMismatchExpiredKeys) {
  PartitionPrivateStateProto proto_1 = PARSE_TEXT_PROTO(R"pb(
    symmetric_keys { id: 1 symmetric_key: "key1" }
    expired_keys: "expired_key1"
    buckets { key: "foo" values: 1 values: 5 }
  )pb");
  PartitionPrivateStateProto proto_2 = PARSE_TEXT_PROTO(R"pb(
    symmetric_keys { id: 2 symmetric_key: "key2" }
    expired_keys: "expired_key2"
    buckets { key: "foo" values: 1 values: 5 }
  )pb");

  PartitionPrivateState state;
  EXPECT_TRUE(state.Merge(PartitionPrivateState::Parse(proto_1).value()));
  EXPECT_FALSE(state.Merge(PartitionPrivateState::Parse(proto_2).value()));
}
}  // namespace
}  // namespace confidential_federated_compute::fed_sql