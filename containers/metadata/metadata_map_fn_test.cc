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
#include "containers/metadata/metadata_map_fn.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "fcp/confidentialcompute/constants.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/tee_payload_metadata.pb.h"
#include "gmock/gmock.h"
#include "google/rpc/code.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::metadata {
namespace {

using ::absl::StatusCode;
using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::confidential_federated_compute::fns::KeyValue;
using ::fcp::confidential_compute::kEventTimeColumnName;
using ::fcp::confidential_compute::kPrivacyIdColumnName;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::PayloadMetadataSet;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::WriteRequest;
using ::tensorflow_federated::aggregation::CreateTestData;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::MutableStringData;
using ::tensorflow_federated::aggregation::Tensor;
using ::testing::HasSubstr;
using ::testing::Test;

MATCHER_P(LowerNBitsAreZero, n, "") {
  if (n < 0 || n > 64) {
    *result_listener << "n must be between 0 and 64, but is " << n;
    return false;
  }
  if (n == 0) return true;
  if (n == 64) return arg == 0;

  uint64_t mask = (1ULL << n) - 1;
  if ((arg & mask) != 0) {
    *result_listener << " has non-zero lower " << n << " bits";
    return false;
  }
  return true;
}

MATCHER_P6(EqualsEventTimeRange, start_year, start_month, start_day, end_year,
           end_month, end_day, "") {
  const auto& start_time = arg.start_event_time();
  const auto& end_time = arg.end_event_time();
  bool start_matches = start_time.year() == start_year &&
                       start_time.month() == start_month &&
                       start_time.day() == start_day;
  bool end_matches = end_time.year() == end_year &&
                     end_time.month() == end_month && end_time.day() == end_day;

  if (!start_matches) {
    *result_listener << " has start time " << start_time.year() << "-"
                     << start_time.month() << "-" << start_time.day();
  }
  if (!end_matches) {
    if (!start_matches) {
      *result_listener << " and";
    }
    *result_listener << " has end time " << end_time.year() << "-"
                     << end_time.month() << "-" << end_time.day();
  }
  return start_matches && end_matches;
}

absl::StatusOr<Tensor> BuildStringTensor(std::string name,
                                         std::vector<std::string> values) {
  auto data = std::make_unique<MutableStringData>(values.size());
  for (auto& value : values) {
    data->Add(std::move(value));
  }
  FCP_ASSIGN_OR_RETURN(
      Tensor tensor, Tensor::Create(DataType::DT_STRING,
                                    {(int64_t)values.size()}, std::move(data)));
  FCP_RETURN_IF_ERROR(tensor.set_name(std::move(name)));
  return tensor;
}

absl::StatusOr<Tensor> BuildIntTensor(std::string name,
                                      std::initializer_list<int64_t> values) {
  FCP_ASSIGN_OR_RETURN(
      Tensor tensor,
      Tensor::Create(DataType::DT_INT64, {(int64_t)values.size()},
                     CreateTestData<int64_t>(values)));
  FCP_RETURN_IF_ERROR(tensor.set_name(std::move(name)));
  return tensor;
}

absl::StatusOr<std::string> BuildCheckpointFromTensors(
    std::vector<Tensor> tensors) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  auto builder = builder_factory.Create();
  for (const auto& tensor : tensors) {
    FCP_RETURN_IF_ERROR(builder->Add(tensor.name(), tensor));
  }
  absl::StatusOr<absl::Cord> checkpoint = builder->Build();
  FCP_RETURN_IF_ERROR(checkpoint.status());
  return std::string(*checkpoint);
}

// Creates a checkpoint with a privacy ID and event times.
absl::StatusOr<std::string> BuildCheckpoint(
    std::string privacy_id_val, std::vector<std::string> event_times) {
  std::vector<Tensor> tensors;

  FCP_ASSIGN_OR_RETURN(
      Tensor privacy_id_tensor,
      BuildStringTensor(kPrivacyIdColumnName, {privacy_id_val}));
  tensors.push_back(std::move(privacy_id_tensor));

  FCP_ASSIGN_OR_RETURN(Tensor event_times_tensor,
                       BuildStringTensor(kEventTimeColumnName, event_times));
  tensors.push_back(std::move(event_times_tensor));
  return BuildCheckpointFromTensors(std::move(tensors));
}

class MockContext : public confidential_federated_compute::Session::Context {
 public:
  MOCK_METHOD(bool, Emit, (ReadResponse), (override));
};

class MetadataMapFnTest : public Test {
 protected:
  void SetUp() override {
    config_ = PARSE_TEXT_PROTO(R"pb(
      metadata_configs {
        key: "test_config"
        value {
          num_partitions: 10
          event_time_range_granularity: EVENT_TIME_GRANULARITY_DAY
        }
      }
    )pb");
    session_ = std::make_unique<MetadataMapFn>(config_);
  }

  fcp::confidentialcompute::MetadataContainerConfig config_;
  std::unique_ptr<MetadataMapFn> session_;
  testing::StrictMock<MockContext> context_;
};

TEST_F(MetadataMapFnTest, ConfigureIsNoOp) {
  fcp::confidentialcompute::ConfigureRequest request;
  EXPECT_THAT(session_->Configure(request, context_), IsOk());
}

TEST_F(MetadataMapFnTest, FinalizeIsNoOp) {
  fcp::confidentialcompute::FinalizeRequest request;
  BlobMetadata metadata;
  EXPECT_THAT(session_->Finalize(request, metadata, context_), IsOk());
}

TEST_F(MetadataMapFnTest, CommitIsNoOp) {
  fcp::confidentialcompute::CommitRequest request;
  EXPECT_THAT(session_->Commit(request, context_), IsOk());
}

TEST_F(MetadataMapFnTest, MapSucceeds) {
  std::string checkpoint =
      BuildCheckpoint("16byteprivacyid1", {"2025-01-01T12:00:00+00:00",
                                           "2025-01-02T12:00:00+00:00"})
          .value();

  KeyValue input;
  input.value.data = checkpoint;
  absl::StatusOr<KeyValue> result = session_->Map(input, context_);
  ASSERT_THAT(result, IsOk());

  PayloadMetadataSet metadata_set;
  ASSERT_TRUE(result->key.UnpackTo(&metadata_set));
  ASSERT_EQ(metadata_set.metadata_size(), 1);
  const auto& tee_metadata = metadata_set.metadata().at("test_config");

  // Verify partition key.
  EXPECT_THAT(tee_metadata.partition_key(), LowerNBitsAreZero(60));

  // Verify event time range.
  EXPECT_THAT(tee_metadata.event_time_range(),
              EqualsEventTimeRange(2025, 1, 1, 2025, 1, 3));
}

TEST_F(MetadataMapFnTest, MapSucceedsWithTimezone) {
  // Event times with different timezones but on the same days. The timezone
  // should be ignored.
  // The two timestamps below are equivalent to the same epoch time
  // (2025-01-02T02:00:00Z), but have different local dates.
  std::string checkpoint =
      BuildCheckpoint("16byteprivacyid1", {"2025-01-01T18:00:00-08:00",
                                           "2025-01-02T10:00:00+08:00"})
          .value();

  KeyValue input;
  input.value.data = checkpoint;
  absl::StatusOr<KeyValue> result = session_->Map(input, context_);
  ASSERT_THAT(result, IsOk());

  PayloadMetadataSet metadata_set;
  ASSERT_TRUE(result->key.UnpackTo(&metadata_set));
  const auto& tee_metadata = metadata_set.metadata().at("test_config");
  EXPECT_THAT(tee_metadata.event_time_range(),
              EqualsEventTimeRange(2025, 1, 1, 2025, 1, 3));
}

TEST_F(MetadataMapFnTest, MapFailsIfCheckpointIsInvalid) {
  KeyValue input;
  input.value.data = "invalid checkpoint";
  auto result = session_->Map(input, context_);
  EXPECT_THAT(result.status(), StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(MetadataMapFnTest, MapSucceedsWithOnePartition) {
  config_ = PARSE_TEXT_PROTO(R"pb(
    metadata_configs {
      key: "test_config"
      value {
        num_partitions: 1
        event_time_range_granularity: EVENT_TIME_GRANULARITY_DAY
      }
    }
  )pb");
  session_ = std::make_unique<MetadataMapFn>(config_);
  std::string checkpoint =
      BuildCheckpoint("16byteprivacyid1", {"2025-01-01T12:00:00+00:00"})
          .value();

  KeyValue input;
  input.value.data = checkpoint;
  absl::StatusOr<KeyValue> result = session_->Map(input, context_);
  ASSERT_THAT(result, IsOk());

  PayloadMetadataSet metadata_set;
  ASSERT_TRUE(result->key.UnpackTo(&metadata_set));
  const auto& tee_metadata = metadata_set.metadata().at("test_config");

  // With 1 partition, all privacy IDs map to partition key 0.
  EXPECT_EQ(tee_metadata.partition_key(), 0);
}

TEST_F(MetadataMapFnTest, MapSucceedsWithMultipleConfigs) {
  config_ = PARSE_TEXT_PROTO(R"pb(
    metadata_configs {
      key: "config_1"
      value {
        num_partitions: 1000
        event_time_range_granularity: EVENT_TIME_GRANULARITY_DAY
      }
    }
    metadata_configs {
      key: "config_2"
      value {
        num_partitions: 1
        event_time_range_granularity: EVENT_TIME_GRANULARITY_DAY
      }
    }
  )pb");
  session_ = std::make_unique<MetadataMapFn>(config_);
  std::string checkpoint =
      BuildCheckpoint("16byteprivacyid1", {"2025-01-01T12:00:00+00:00"})
          .value();

  KeyValue input;
  input.value.data = checkpoint;
  absl::StatusOr<KeyValue> result = session_->Map(input, context_);
  ASSERT_THAT(result, IsOk());

  PayloadMetadataSet metadata_set;
  ASSERT_TRUE(result->key.UnpackTo(&metadata_set));
  ASSERT_EQ(metadata_set.metadata_size(), 2);
  const auto& tee_metadata_1 = metadata_set.metadata().at("config_1");
  EXPECT_THAT(tee_metadata_1.partition_key(), LowerNBitsAreZero(54));
  const auto& tee_metadata_2 = metadata_set.metadata().at("config_2");
  EXPECT_EQ(tee_metadata_2.partition_key(), 0);
}

TEST_F(MetadataMapFnTest, MapFailsWithZeroPartitions) {
  config_ = PARSE_TEXT_PROTO(R"pb(
    metadata_configs {
      key: "test_config"
      value {
        num_partitions: 0
        event_time_range_granularity: EVENT_TIME_GRANULARITY_DAY
      }
    }
  )pb");
  session_ = std::make_unique<MetadataMapFn>(config_);
  std::string checkpoint =
      BuildCheckpoint("16byteprivacyid1", {"2025-01-01T12:00:00+00:00"})
          .value();

  KeyValue input;
  input.value.data = checkpoint;
  auto result = session_->Map(input, context_);
  EXPECT_THAT(result.status(),
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("num_partitions cannot be 0")));
}

TEST_F(MetadataMapFnTest, MapSucceedsWithSingleEventTime) {
  std::string checkpoint =
      BuildCheckpoint("16byteprivacyid1", {"2025-01-01T12:00:00+00:00"})
          .value();

  KeyValue input;
  input.value.data = checkpoint;
  absl::StatusOr<KeyValue> result = session_->Map(input, context_);
  ASSERT_THAT(result, IsOk());

  PayloadMetadataSet metadata_set;
  ASSERT_TRUE(result->key.UnpackTo(&metadata_set));
  const auto& tee_metadata = metadata_set.metadata().at("test_config");

  // Verify event time range.
  EXPECT_THAT(tee_metadata.event_time_range(),
              EqualsEventTimeRange(2025, 1, 1, 2025, 1, 2));
}

TEST_F(MetadataMapFnTest, MapWithNoEventTimesDoesNotSetEventTimeRange) {
  std::string checkpoint = BuildCheckpoint("16byteprivacyid1", {}).value();

  KeyValue input;
  input.value.data = checkpoint;
  absl::StatusOr<KeyValue> result = session_->Map(input, context_);
  ASSERT_THAT(result, IsOk());

  PayloadMetadataSet metadata_set;
  ASSERT_TRUE(result->key.UnpackTo(&metadata_set));
  const auto& tee_metadata = metadata_set.metadata().at("test_config");

  // Verify event time range is not set.
  EXPECT_FALSE(tee_metadata.has_event_time_range());
}

TEST_F(MetadataMapFnTest, MapFailsWithInvalidEventTimeFormat) {
  std::string checkpoint =
      BuildCheckpoint("16byteprivacyid1", {"invalid-time-format"}).value();

  KeyValue input;
  input.value.data = checkpoint;
  auto result = session_->Map(input, context_);
  EXPECT_THAT(result.status(),
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("Invalid event time format")));
}

TEST_F(MetadataMapFnTest, MapFailsIfPrivacyIdIsMissing) {
  absl::StatusOr<Tensor> event_times_tensor =
      BuildStringTensor(kEventTimeColumnName, {"2025-01-01T12:00:00+00:00"});
  ASSERT_THAT(event_times_tensor, IsOk());
  std::vector<Tensor> tensors;
  tensors.push_back(*std::move(event_times_tensor));
  absl::StatusOr<std::string> checkpoint =
      BuildCheckpointFromTensors(std::move(tensors));
  ASSERT_THAT(checkpoint, IsOk());

  KeyValue input;
  input.value.data = *checkpoint;
  auto result = session_->Map(input, context_);
  EXPECT_THAT(result.status(),
              StatusIs(StatusCode::kNotFound,
                       HasSubstr("No aggregation tensor found for name "
                                 "confidential_compute_privacy_id")));
}

TEST_F(MetadataMapFnTest, MapFailsIfEventTimeIsMissing) {
  absl::StatusOr<Tensor> privacy_id_tensor =
      BuildStringTensor(kPrivacyIdColumnName, {"16byteprivacyid1"});
  ASSERT_THAT(privacy_id_tensor, IsOk());
  std::vector<Tensor> tensors;
  tensors.push_back(*std::move(privacy_id_tensor));
  absl::StatusOr<std::string> checkpoint =
      BuildCheckpointFromTensors(std::move(tensors));
  ASSERT_THAT(checkpoint, IsOk());

  KeyValue input;
  input.value.data = *checkpoint;
  auto result = session_->Map(input, context_);
  EXPECT_THAT(result.status(),
              StatusIs(StatusCode::kNotFound,
                       HasSubstr("No aggregation tensor found for name "
                                 "confidential_compute_event_time")));
}

TEST_F(MetadataMapFnTest, MapFailsIfPrivacyIdHasWrongType) {
  absl::StatusOr<Tensor> privacy_id_tensor =
      BuildIntTensor(kPrivacyIdColumnName, {123});
  ASSERT_THAT(privacy_id_tensor, IsOk());
  absl::StatusOr<Tensor> event_times_tensor =
      BuildStringTensor(kEventTimeColumnName, {"2025-01-01T12:00:00+00:00"});
  ASSERT_THAT(event_times_tensor, IsOk());
  std::vector<Tensor> tensors;
  tensors.push_back(*std::move(privacy_id_tensor));
  tensors.push_back(*std::move(event_times_tensor));
  absl::StatusOr<std::string> checkpoint =
      BuildCheckpointFromTensors(std::move(tensors));
  ASSERT_THAT(checkpoint, IsOk());

  KeyValue input;
  input.value.data = *checkpoint;
  auto result = session_->Map(input, context_);
  EXPECT_THAT(
      result.status(),
      StatusIs(StatusCode::kInvalidArgument,
               HasSubstr("`confidential_compute_privacy_id` tensor must be a "
                         "string tensor")));
}

TEST_F(MetadataMapFnTest, MapFailsIfEventTimeHasWrongType) {
  absl::StatusOr<Tensor> privacy_id_tensor =
      BuildStringTensor(kPrivacyIdColumnName, {"16byteprivacyid1"});
  ASSERT_THAT(privacy_id_tensor, IsOk());
  absl::StatusOr<Tensor> event_times_tensor =
      BuildIntTensor(kEventTimeColumnName, {123});
  ASSERT_THAT(event_times_tensor, IsOk());
  std::vector<Tensor> tensors;
  tensors.push_back(*std::move(privacy_id_tensor));
  tensors.push_back(*std::move(event_times_tensor));
  absl::StatusOr<std::string> checkpoint =
      BuildCheckpointFromTensors(std::move(tensors));
  ASSERT_THAT(checkpoint, IsOk());

  KeyValue input;
  input.value.data = *checkpoint;
  auto result = session_->Map(input, context_);
  EXPECT_THAT(
      result.status(),
      StatusIs(StatusCode::kInvalidArgument,
               HasSubstr("`confidential_compute_event_time` tensor must be a "
                         "string tensor")));
}

TEST_F(MetadataMapFnTest, MapFailsIfPrivacyIdHasWrongShape) {
  absl::StatusOr<Tensor> privacy_id_tensor = BuildStringTensor(
      kPrivacyIdColumnName, {"16byteprivacyid1", "16byteprivacyid2"});
  ASSERT_THAT(privacy_id_tensor, IsOk());
  absl::StatusOr<Tensor> event_times_tensor =
      BuildStringTensor(kEventTimeColumnName, {"2025-01-01T12:00:00+00:00"});
  ASSERT_THAT(event_times_tensor, IsOk());
  std::vector<Tensor> tensors;
  tensors.push_back(*std::move(privacy_id_tensor));
  tensors.push_back(*std::move(event_times_tensor));
  absl::StatusOr<std::string> checkpoint =
      BuildCheckpointFromTensors(std::move(tensors));
  ASSERT_THAT(checkpoint, IsOk());

  KeyValue input;
  input.value.data = *checkpoint;
  auto result = session_->Map(input, context_);
  EXPECT_THAT(result.status(),
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("`confidential_compute_privacy_id` tensor "
                                 "must be a scalar")));
}

TEST_F(MetadataMapFnTest, MapFailsIfEventTimeHasWrongShape) {
  absl::StatusOr<Tensor> privacy_id_tensor =
      BuildStringTensor(kPrivacyIdColumnName, {"16byteprivacyid1"});
  ASSERT_THAT(privacy_id_tensor, IsOk());
  auto data = std::make_unique<MutableStringData>(1);
  data->Add("2025-01-01T12:00:00+00:00");
  absl::StatusOr<Tensor> event_times_tensor =
      Tensor::Create(DataType::DT_STRING, {1, 1}, std::move(data));
  ASSERT_THAT(event_times_tensor, IsOk());
  ASSERT_THAT(event_times_tensor->set_name(kEventTimeColumnName), IsOk());
  std::vector<Tensor> tensors;
  tensors.push_back(*std::move(privacy_id_tensor));
  tensors.push_back(*std::move(event_times_tensor));
  absl::StatusOr<std::string> checkpoint =
      BuildCheckpointFromTensors(std::move(tensors));
  ASSERT_THAT(checkpoint, IsOk());

  KeyValue input;
  input.value.data = *checkpoint;
  auto result = session_->Map(input, context_);
  EXPECT_THAT(result.status(),
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("`confidential_compute_event_time` tensor "
                                 "must have one dimension")));
}
}  // namespace
}  // namespace confidential_federated_compute::metadata
