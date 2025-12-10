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

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "containers/fns/map_fn.h"
#include "containers/metadata/testing/test_utils.h"
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
using ::confidential_federated_compute::metadata::testing::BuildCheckpoint;
using ::confidential_federated_compute::metadata::testing::
    BuildCheckpointFromTensors;
using ::confidential_federated_compute::metadata::testing::BuildIntTensor;
using ::confidential_federated_compute::metadata::testing::BuildStringTensor;
using ::confidential_federated_compute::metadata::testing::EqualsEventTimeRange;
using ::confidential_federated_compute::metadata::testing::LowerNBitsAreZero;
using ::fcp::confidential_compute::kEventTimeColumnName;
using ::fcp::confidential_compute::kPrivacyIdColumnName;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::MetadataContainerConfig;
using ::fcp::confidentialcompute::PayloadMetadataSet;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::WriteFinishedResponse;
using ::fcp::confidentialcompute::WriteRequest;
using ::google::protobuf::Any;
using ::tensorflow_federated::aggregation::CreateTestData;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::MutableStringData;
using ::tensorflow_federated::aggregation::Tensor;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Return;
using ::testing::SaveArg;
using ::testing::StrictMock;
using ::testing::Test;

class MetadataMapFnFactoryTest : public Test {};

TEST_F(MetadataMapFnFactoryTest,
       ProvideMetadataMapFnFactoryFailsWithInvalidConfigConstraints) {
  Any config_constraints;
  EXPECT_THAT(ProvideMetadataMapFnFactory(Any(), config_constraints),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(MetadataMapFnFactoryTest, ProvideMetadataMapFnFactorySuccess) {
  MetadataContainerConfig config;
  Any config_constraints;
  config_constraints.PackFrom(config);
  EXPECT_THAT(ProvideMetadataMapFnFactory(Any(), config_constraints), IsOk());
}

TEST_F(MetadataMapFnFactoryTest, CreateFnSuccess) {
  MetadataContainerConfig config;
  Any config_constraints;
  config_constraints.PackFrom(config);

  absl::StatusOr<std::unique_ptr<fns::FnFactory>> factory =
      ProvideMetadataMapFnFactory(Any(), config_constraints);
  EXPECT_THAT(factory, IsOk());
  // A single factory can create multiple Fns.
  EXPECT_THAT((*factory)->CreateFn(), IsOk());
  EXPECT_THAT((*factory)->CreateFn(), IsOk());
}

std::unique_ptr<fns::Fn> CreateMetadataMapFn(
    const MetadataContainerConfig& config) {
  Any config_constraints;
  config_constraints.PackFrom(config);
  absl::StatusOr<std::unique_ptr<fns::FnFactory>> factory =
      ProvideMetadataMapFnFactory(/*configuration=*/Any(), config_constraints);
  CHECK_OK(factory.status());
  absl::StatusOr<std::unique_ptr<fns::Fn>> fn = (*factory)->CreateFn();
  CHECK_OK(fn.status());
  return *std::move(fn);
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
    fn_ = CreateMetadataMapFn(config_);
  }

  MetadataContainerConfig config_;
  std::unique_ptr<fns::Fn> fn_;
  StrictMock<MockContext> context_;
};

TEST_F(MetadataMapFnTest, ConfigureIsNoOp) {
  fcp::confidentialcompute::ConfigureRequest request;
  EXPECT_THAT(fn_->Configure(request, context_), IsOk());
}

TEST_F(MetadataMapFnTest, FinalizeIsNoOp) {
  fcp::confidentialcompute::FinalizeRequest request;
  BlobMetadata metadata;
  EXPECT_THAT(fn_->Finalize(request, metadata, context_), IsOk());
}

TEST_F(MetadataMapFnTest, CommitIsNoOp) {
  fcp::confidentialcompute::CommitRequest request;
  EXPECT_THAT(fn_->Commit(request, context_), IsOk());
}

TEST_F(MetadataMapFnTest, MapSucceeds) {
  std::string checkpoint = BuildCheckpoint(
      "16byteprivacyid1",
      {"2025-01-01T12:00:00+00:00", "2025-01-02T12:00:00+00:00"});

  ReadResponse read_response;
  EXPECT_CALL(context_, Emit(_))
      .WillOnce(DoAll(SaveArg<0>(&read_response), Return(true)));
  absl::StatusOr<WriteFinishedResponse> result =
      fn_->Write(WriteRequest(), checkpoint, context_);
  ASSERT_THAT(result, IsOk());

  PayloadMetadataSet metadata_set;
  ASSERT_TRUE(
      read_response.first_response_configuration().UnpackTo(&metadata_set));
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
  std::string checkpoint = BuildCheckpoint(
      "16byteprivacyid1",
      {"2025-01-01T18:00:00-08:00", "2025-01-02T10:00:00+08:00"});

  ReadResponse read_response;
  EXPECT_CALL(context_, Emit(_))
      .WillOnce(DoAll(SaveArg<0>(&read_response), Return(true)));
  absl::StatusOr<WriteFinishedResponse> result =
      fn_->Write(WriteRequest(), checkpoint, context_);
  ASSERT_THAT(result, IsOk());

  PayloadMetadataSet metadata_set;
  ASSERT_TRUE(
      read_response.first_response_configuration().UnpackTo(&metadata_set));
  const auto& tee_metadata = metadata_set.metadata().at("test_config");
  EXPECT_THAT(tee_metadata.event_time_range(),
              EqualsEventTimeRange(2025, 1, 1, 2025, 1, 3));
}

TEST_F(MetadataMapFnTest, MapFailsIfCheckpointIsInvalid) {
  absl::StatusOr<WriteFinishedResponse> result =
      fn_->Write(WriteRequest(), "invalid checkpoint", context_);
  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(result->status().code(), Eq(google::rpc::Code::INVALID_ARGUMENT));
  EXPECT_THAT(result->status().message(),
              HasSubstr("Failed to deserialize checkpoint"));
}

TEST_F(MetadataMapFnTest, MapSucceedsWithOnePartition) {
  MetadataContainerConfig config = PARSE_TEXT_PROTO(R"pb(
    metadata_configs {
      key: "test_config"
      value {
        num_partitions: 1
        event_time_range_granularity: EVENT_TIME_GRANULARITY_DAY
      }
    }
  )pb");
  std::unique_ptr<fns::Fn> fn = CreateMetadataMapFn(config);
  std::string checkpoint =
      BuildCheckpoint("16byteprivacyid1", {"2025-01-01T12:00:00+00:00"});

  ReadResponse read_response;
  EXPECT_CALL(context_, Emit(_))
      .WillOnce(DoAll(SaveArg<0>(&read_response), Return(true)));
  absl::StatusOr<WriteFinishedResponse> result =
      fn->Write(WriteRequest(), checkpoint, context_);
  ASSERT_THAT(result, IsOk());

  PayloadMetadataSet metadata_set;
  ASSERT_TRUE(
      read_response.first_response_configuration().UnpackTo(&metadata_set));
  const auto& tee_metadata = metadata_set.metadata().at("test_config");

  // With 1 partition, all privacy IDs map to partition key 0.
  EXPECT_EQ(tee_metadata.partition_key(), 0);
}

TEST_F(MetadataMapFnTest, MapSucceedsWithMultipleConfigs) {
  MetadataContainerConfig config = PARSE_TEXT_PROTO(R"pb(
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
  std::unique_ptr<fns::Fn> fn = CreateMetadataMapFn(config);
  std::string checkpoint =
      BuildCheckpoint("16byteprivacyid1", {"2025-01-01T12:00:00+00:00"});

  ReadResponse read_response;
  EXPECT_CALL(context_, Emit(_))
      .WillOnce(DoAll(SaveArg<0>(&read_response), Return(true)));
  absl::StatusOr<WriteFinishedResponse> result =
      fn->Write(WriteRequest(), checkpoint, context_);
  ASSERT_THAT(result, IsOk());

  PayloadMetadataSet metadata_set;
  ASSERT_TRUE(
      read_response.first_response_configuration().UnpackTo(&metadata_set));
  ASSERT_EQ(metadata_set.metadata_size(), 2);
  const auto& tee_metadata_1 = metadata_set.metadata().at("config_1");
  EXPECT_THAT(tee_metadata_1.partition_key(), LowerNBitsAreZero(54));
  const auto& tee_metadata_2 = metadata_set.metadata().at("config_2");
  EXPECT_EQ(tee_metadata_2.partition_key(), 0);
}

TEST_F(MetadataMapFnTest, MapFailsWithZeroPartitions) {
  MetadataContainerConfig config = PARSE_TEXT_PROTO(R"pb(
    metadata_configs {
      key: "test_config"
      value {
        num_partitions: 0
        event_time_range_granularity: EVENT_TIME_GRANULARITY_DAY
      }
    }
  )pb");
  std::unique_ptr<fns::Fn> fn = CreateMetadataMapFn(config);
  std::string checkpoint =
      BuildCheckpoint("16byteprivacyid1", {"2025-01-01T12:00:00+00:00"});

  absl::StatusOr<WriteFinishedResponse> result =
      fn->Write(WriteRequest(), checkpoint, context_);
  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(result->status().code(), Eq(google::rpc::Code::INVALID_ARGUMENT));
  EXPECT_THAT(result->status().message(),
              HasSubstr("num_partitions cannot be 0"));
}

TEST_F(MetadataMapFnTest, MapSucceedsWithSingleEventTime) {
  std::string checkpoint =
      BuildCheckpoint("16byteprivacyid1", {"2025-01-01T12:00:00+00:00"});

  ReadResponse read_response;
  EXPECT_CALL(context_, Emit(_))
      .WillOnce(DoAll(SaveArg<0>(&read_response), Return(true)));
  absl::StatusOr<WriteFinishedResponse> result =
      fn_->Write(WriteRequest(), checkpoint, context_);
  ASSERT_THAT(result, IsOk());

  PayloadMetadataSet metadata_set;
  ASSERT_TRUE(
      read_response.first_response_configuration().UnpackTo(&metadata_set));
  const auto& tee_metadata = metadata_set.metadata().at("test_config");

  // Verify event time range.
  EXPECT_THAT(tee_metadata.event_time_range(),
              EqualsEventTimeRange(2025, 1, 1, 2025, 1, 2));
}

TEST_F(MetadataMapFnTest, MapWithNoEventTimesDoesNotSetEventTimeRange) {
  std::string checkpoint = BuildCheckpoint("16byteprivacyid1", {});

  ReadResponse read_response;
  EXPECT_CALL(context_, Emit(_))
      .WillOnce(DoAll(SaveArg<0>(&read_response), Return(true)));
  absl::StatusOr<WriteFinishedResponse> result =
      fn_->Write(WriteRequest(), checkpoint, context_);
  ASSERT_THAT(result, IsOk());

  PayloadMetadataSet metadata_set;
  ASSERT_TRUE(
      read_response.first_response_configuration().UnpackTo(&metadata_set));
  const auto& tee_metadata = metadata_set.metadata().at("test_config");

  // Verify event time range is not set.
  EXPECT_FALSE(tee_metadata.has_event_time_range());
}

TEST_F(MetadataMapFnTest, MapFailsWithInvalidEventTimeFormat) {
  std::string checkpoint =
      BuildCheckpoint("16byteprivacyid1", {"invalid-time-format"});

  auto result = fn_->Write(WriteRequest(), checkpoint, context_);
  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(result->status().code(), Eq(google::rpc::Code::INVALID_ARGUMENT));
  EXPECT_THAT(result->status().message(),
              HasSubstr("Invalid event time format"));
}

TEST_F(MetadataMapFnTest, MapFailsIfPrivacyIdIsMissing) {
  Tensor event_times_tensor =
      BuildStringTensor(kEventTimeColumnName, {"2025-01-01T12:00:00+00:00"});
  std::vector<Tensor> tensors;
  tensors.push_back(std::move(event_times_tensor));
  std::string checkpoint = BuildCheckpointFromTensors(std::move(tensors));

  auto result = fn_->Write(WriteRequest(), checkpoint, context_);
  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(result->status().code(), Eq(google::rpc::Code::NOT_FOUND));
  EXPECT_THAT(result->status().message(),
              HasSubstr("No aggregation tensor found for name "
                        "confidential_compute_privacy_id"));
}

TEST_F(MetadataMapFnTest, MapFailsIfEventTimeIsMissing) {
  Tensor privacy_id_tensor =
      BuildStringTensor(kPrivacyIdColumnName, {"16byteprivacyid1"});
  std::vector<Tensor> tensors;
  tensors.push_back(std::move(privacy_id_tensor));
  std::string checkpoint = BuildCheckpointFromTensors(std::move(tensors));

  auto result = fn_->Write(WriteRequest(), checkpoint, context_);
  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(result->status().code(), Eq(google::rpc::Code::NOT_FOUND));
  EXPECT_THAT(result->status().message(),
              HasSubstr("No aggregation tensor found for name "
                        "confidential_compute_event_time"));
}

TEST_F(MetadataMapFnTest, MapFailsIfPrivacyIdHasWrongType) {
  Tensor privacy_id_tensor = BuildIntTensor(kPrivacyIdColumnName, {123});
  Tensor event_times_tensor =
      BuildStringTensor(kEventTimeColumnName, {"2025-01-01T12:00:00+00:00"});
  std::vector<Tensor> tensors;
  tensors.push_back(std::move(privacy_id_tensor));
  tensors.push_back(std::move(event_times_tensor));
  std::string checkpoint = BuildCheckpointFromTensors(std::move(tensors));

  auto result = fn_->Write(WriteRequest(), checkpoint, context_);
  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(result->status().code(), Eq(google::rpc::Code::INVALID_ARGUMENT));
  EXPECT_THAT(result->status().message(),
              HasSubstr("`confidential_compute_privacy_id` tensor must be a "
                        "string tensor"));
}

TEST_F(MetadataMapFnTest, MapFailsIfEventTimeHasWrongType) {
  Tensor privacy_id_tensor =
      BuildStringTensor(kPrivacyIdColumnName, {"16byteprivacyid1"});
  Tensor event_times_tensor = BuildIntTensor(kEventTimeColumnName, {123});
  std::vector<Tensor> tensors;
  tensors.push_back(std::move(privacy_id_tensor));
  tensors.push_back(std::move(event_times_tensor));
  std::string checkpoint = BuildCheckpointFromTensors(std::move(tensors));

  auto result = fn_->Write(WriteRequest(), checkpoint, context_);
  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(result->status().code(), Eq(google::rpc::Code::INVALID_ARGUMENT));
  EXPECT_THAT(result->status().message(),
              HasSubstr("`confidential_compute_event_time` tensor must be a "
                        "string tensor"));
}

TEST_F(MetadataMapFnTest, MapFailsIfPrivacyIdHasWrongShape) {
  Tensor privacy_id_tensor = BuildStringTensor(
      kPrivacyIdColumnName, {"16byteprivacyid1", "16byteprivacyid2"});
  Tensor event_times_tensor =
      BuildStringTensor(kEventTimeColumnName, {"2025-01-01T12:00:00+00:00"});
  std::vector<Tensor> tensors;
  tensors.push_back(std::move(privacy_id_tensor));
  tensors.push_back(std::move(event_times_tensor));
  std::string checkpoint = BuildCheckpointFromTensors(std::move(tensors));

  auto result = fn_->Write(WriteRequest(), checkpoint, context_);
  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(result->status().code(), Eq(google::rpc::Code::INVALID_ARGUMENT));
  EXPECT_THAT(result->status().message(),
              HasSubstr("`confidential_compute_privacy_id` tensor must be a "
                        "scalar"));
}

TEST_F(MetadataMapFnTest, MapFailsIfEventTimeHasWrongShape) {
  Tensor privacy_id_tensor =
      BuildStringTensor(kPrivacyIdColumnName, {"16byteprivacyid1"});
  auto data = std::make_unique<MutableStringData>(1);
  data->Add("2025-01-01T12:00:00+00:00");
  absl::StatusOr<Tensor> event_times_tensor =
      Tensor::Create(DataType::DT_STRING, {1, 1}, std::move(data));
  ASSERT_THAT(event_times_tensor->set_name(kEventTimeColumnName), IsOk());
  std::vector<Tensor> tensors;
  tensors.push_back(std::move(privacy_id_tensor));
  tensors.push_back(*std::move(event_times_tensor));
  std::string checkpoint = BuildCheckpointFromTensors(std::move(tensors));

  auto result = fn_->Write(WriteRequest(), checkpoint, context_);
  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(result->status().code(), Eq(google::rpc::Code::INVALID_ARGUMENT));
  EXPECT_THAT(result->status().message(),
              HasSubstr("`confidential_compute_event_time` tensor "
                        "must have one dimension"));
}
}  // namespace
}  // namespace confidential_federated_compute::metadata
