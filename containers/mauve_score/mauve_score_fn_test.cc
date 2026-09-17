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

#include "mauve_score_fn.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "budget.h"
#include "containers/common/intervals/interval.h"
#include "containers/common/time_budget/budget.pb.h"
#include "containers/fns/fn_factory.h"
#include "fcp/confidentialcompute/private_state.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "fcp/protos/confidentialcompute/construct_user_session.pb.h"
#include "fcp/protos/confidentialcompute/mauve_score_config.pb.h"
#include "fcp/protos/confidentialcompute/sentence_transformers_config.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/any.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"

namespace confidential_federated_compute::mauve_score {

// Forward declarations for testing — these are internal to mauve_score_fn.cc.
using ReadRecordFn = absl::AnyInvocable<absl::StatusOr<
    std::vector<fcp::confidentialcompute::Embedding>>(absl::string_view)>;
using ComputeMauveFn =
    std::function<absl::StatusOr<fcp::confidentialcompute::MauveScoreResult>(
        const std::vector<std::vector<float>>&,
        const std::vector<std::vector<float>>&)>;
absl::StatusOr<std::unique_ptr<fns::FnFactory>> ProvideMauveScoreFnFactory(
    const google::protobuf::Any& configuration,
    const google::protobuf::Any& config_constraints,
    const confidential_federated_compute::fns::WriteConfigurationMap&
        write_configuration_map,
    ReadRecordFn read_record_fn, ComputeMauveFn compute_mauve_fn = nullptr);

namespace {

constexpr absl::string_view kDataTensorName = "data";
constexpr absl::string_view kConfigId = "config_id";
constexpr absl::string_view kPath = "path";

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::confidential_federated_compute::fns::Fn;
using ::confidential_federated_compute::fns::FnFactory;
using ::fcp::confidential_compute::kPrivateStateConfigId;
using ::fcp::confidentialcompute::AssociatedMetadata;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::Embedding;
using ::fcp::confidentialcompute::MauveScoreContainerConfigConstraints;
using ::fcp::confidentialcompute::MauveScoreContainerInitializeConfiguration;
using ::fcp::confidentialcompute::MauveScoreResult;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::SessionTimeWindowMetadata;
using ::fcp::confidentialcompute::WriteRequest;
using ::google::protobuf::Any;
using ::tensorflow_federated::aggregation::DT_FLOAT;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;
using ::tensorflow_federated::aggregation::VectorData;
using ::testing::_;
using ::testing::Return;
using ::testing::StrictMock;

class MockContext : public confidential_federated_compute::Session::Context {
 public:
  MOCK_METHOD(bool, Emit, (ReadResponse), (override));
  MOCK_METHOD(bool, EmitUnencrypted, (Session::KV), (override));
  MOCK_METHOD(bool, EmitEncrypted, (int, Session::KV), (override));
  MOCK_METHOD(bool, EmitReleasable,
              (int, Session::KV, std::optional<absl::string_view>,
               absl::string_view, std::string&),
              (override));
  MOCK_METHOD(confidential_federated_compute::Counters&, GetCounters, (),
              (override));
};

std::string CreateTempPrivateStateFile(absl::string_view content = "") {
  std::string path = absl::StrCat(testing::TempDir(), "/private_state_",
                                  std::to_string(std::rand()));
  std::ofstream file(path);
  file << content;
  file.close();
  return path;
}

Embedding CreateEmbedding(const std::vector<float>& values, int32_t index = 0) {
  Embedding emb;
  auto* values_proto = emb.mutable_values();
  for (const auto& value : values) {
    *(values_proto->Add()) = value;
  }
  emb.set_index(index);
  return emb;
}

Any CreateValidInitConfig() {
  Any config;
  MauveScoreContainerInitializeConfiguration init_config;
  init_config.set_synthetic_data_embeddings_configuration_id(
      std::string(kConfigId));
  config.PackFrom(init_config);
  return config;
}

Any CreateValidConfigConstraints(uint32_t budget_times = 5,
                                 uint64_t min_agg_window_minutes = 0) {
  Any constraints;
  MauveScoreContainerConfigConstraints mauve_constraints;
  mauve_constraints.mutable_access_budget()->set_times(budget_times);
  mauve_constraints.set_min_agg_window_minutes(min_agg_window_minutes);
  constraints.PackFrom(mauve_constraints);
  return constraints;
}

absl::flat_hash_map<std::string, std::string> CreateWriteConfigurationMap(
    std::string private_state_path) {
  absl::flat_hash_map<std::string, std::string> write_configuration_map;
  write_configuration_map[std::string(kConfigId)] = std::string(kPath);
  write_configuration_map[std::string(kPrivateStateConfigId)] =
      std::move(private_state_path);
  return write_configuration_map;
}

// Create N synthetic embeddings of dimensionality D.
ReadRecordFn CreateSyntheticReadRecordFn(int n = 50, int dim = 8) {
  return [n, dim](
             absl::string_view path) -> absl::StatusOr<std::vector<Embedding>> {
    if (path != kPath) {
      return absl::InvalidArgumentError("Invalid records path.");
    }
    std::vector<Embedding> embeddings;
    for (int i = 0; i < n; i++) {
      std::vector<float> values(dim);
      for (int j = 0; j < dim; j++) {
        // Deterministic: use index as seed.
        values[j] = static_cast<float>(i * dim + j) / (n * dim);
      }
      embeddings.push_back(CreateEmbedding(values, i));
    }
    return embeddings;
  };
}

// Build a FedCompute checkpoint with a "data" tensor of shape [batch, dim].
std::string BuildCheckpoint(int batch, int dim) {
  FederatedComputeCheckpointBuilderFactory factory;
  auto builder = factory.Create();
  std::vector<float> data(batch * dim);
  for (int i = 0; i < batch * dim; i++) {
    data[i] = static_cast<float>(i) / (batch * dim);
  }
  auto t = Tensor::Create(DT_FLOAT, TensorShape({batch, dim}),
                          std::make_unique<VectorData<float>>(std::move(data)));
  CHECK_OK(t);
  CHECK_OK(builder->Add(std::string(kDataTensorName), std::move(*t)));
  auto ckpt = builder->Build();
  CHECK_OK(ckpt);
  return std::string(*ckpt);
}

ComputeMauveFn CreateFakeComputeMauveFn() {
  return [](const std::vector<std::vector<float>>& real_embeddings,
            const std::vector<std::vector<float>>& synth_embeddings)
             -> absl::StatusOr<MauveScoreResult> {
    MauveScoreResult result;
    result.set_mauve_auc(0.95f);
    result.set_num_clusters(10);
    result.set_recall(0.9f);
    result.set_precision(0.92f);
    return result;
  };
}

TEST(MauveScoreFnFactoryTest, InvalidConfig) {
  ReadRecordFn unused_reader = [](absl::string_view) {
    return absl::InternalError("should not be called");
  };
  EXPECT_THAT(
      ProvideMauveScoreFnFactory(Any(), Any(), {}, std::move(unused_reader)),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MauveScoreFnFactoryTest, MissingEmbeddingConfigId) {
  Any config = CreateValidInitConfig();
  Any constraints = CreateValidConfigConstraints();
  std::string private_state_path = CreateTempPrivateStateFile();
  absl::flat_hash_map<std::string, std::string> write_configuration_map;
  write_configuration_map["other_id"] = "some_path";
  write_configuration_map[std::string(kPrivateStateConfigId)] =
      private_state_path;
  ReadRecordFn unused_reader = [](absl::string_view) {
    return absl::InternalError("should not be called");
  };
  EXPECT_THAT(
      ProvideMauveScoreFnFactory(config, constraints, write_configuration_map,
                                 std::move(unused_reader)),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MauveScoreFnFactoryTest, MissingPrivateStateConfigId) {
  Any config = CreateValidInitConfig();
  Any constraints = CreateValidConfigConstraints();
  absl::flat_hash_map<std::string, std::string> write_configuration_map;
  write_configuration_map[std::string(kConfigId)] = std::string(kPath);
  ReadRecordFn unused_reader = [](absl::string_view) {
    return absl::InternalError("should not be called");
  };
  EXPECT_THAT(
      ProvideMauveScoreFnFactory(config, constraints, write_configuration_map,
                                 std::move(unused_reader)),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MauveScoreFnFactoryTest, ReadRecordFailed) {
  Any config = CreateValidInitConfig();
  Any constraints = CreateValidConfigConstraints();
  std::string private_state_path = CreateTempPrivateStateFile();
  auto write_configuration_map =
      CreateWriteConfigurationMap(private_state_path);
  ReadRecordFn bad_reader = [](absl::string_view) {
    return absl::InvalidArgumentError("read failed");
  };
  EXPECT_THAT(
      ProvideMauveScoreFnFactory(config, constraints, write_configuration_map,
                                 std::move(bad_reader)),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MauveScoreFnFactoryTest, MissingConfigConstraints) {
  Any config = CreateValidInitConfig();
  std::string private_state_path = CreateTempPrivateStateFile();
  auto write_configuration_map =
      CreateWriteConfigurationMap(private_state_path);
  ReadRecordFn unused_reader = [](absl::string_view) {
    return absl::InternalError("should not be called");
  };
  EXPECT_THAT(ProvideMauveScoreFnFactory(config, Any(), write_configuration_map,
                                         std::move(unused_reader)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MauveScoreFnFactoryTest, ZeroBudgetRejected) {
  Any config = CreateValidInitConfig();
  Any constraints = CreateValidConfigConstraints(/*budget_times=*/0);
  std::string private_state_path = CreateTempPrivateStateFile();
  auto write_configuration_map =
      CreateWriteConfigurationMap(private_state_path);
  ReadRecordFn unused_reader = [](absl::string_view) {
    return absl::InternalError("should not be called");
  };
  EXPECT_THAT(
      ProvideMauveScoreFnFactory(config, constraints, write_configuration_map,
                                 std::move(unused_reader)),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MauveScoreFnFactoryTest, ValidConfigCreatesFactory) {
  Any config = CreateValidInitConfig();
  Any constraints = CreateValidConfigConstraints();
  std::string private_state_path = CreateTempPrivateStateFile();
  auto write_configuration_map =
      CreateWriteConfigurationMap(private_state_path);
  auto factory =
      ProvideMauveScoreFnFactory(config, constraints, write_configuration_map,
                                 CreateSyntheticReadRecordFn());
  ASSERT_THAT(factory, IsOk());
  auto fn = (*factory)->CreateFn();
  ASSERT_THAT(fn, IsOk());
}

class MauveScoreFnTest : public testing::Test {
 protected:
  static constexpr int kDim = 8;
  static constexpr int kNumSynthetic = 50;

  void SetUp() override {
    Any config = CreateValidInitConfig();
    Any constraints = CreateValidConfigConstraints();
    std::string private_state_path = CreateTempPrivateStateFile();
    auto write_configuration_map =
        CreateWriteConfigurationMap(private_state_path);
    auto fn_factory = ProvideMauveScoreFnFactory(
        config, constraints, write_configuration_map,
        CreateSyntheticReadRecordFn(kNumSynthetic, kDim),
        CreateFakeComputeMauveFn());
    ASSERT_THAT(fn_factory, IsOk());
    factory_ = std::move(*fn_factory);

    auto fn = factory_->CreateFn();
    ASSERT_THAT(fn, IsOk());
    fn_ = std::move(*fn);

    EXPECT_CALL(context_, GetCounters())
        .WillRepeatedly(::testing::ReturnRef(counters_));
  }

  std::unique_ptr<Fn> CreateFnWithState(absl::string_view initial_state,
                                        uint32_t budget_times = 5,
                                        uint64_t min_agg_window_minutes = 0) {
    std::string private_state_path = CreateTempPrivateStateFile(initial_state);
    auto write_configuration_map =
        CreateWriteConfigurationMap(private_state_path);
    auto fn_factory = ProvideMauveScoreFnFactory(
        CreateValidInitConfig(),
        CreateValidConfigConstraints(budget_times, min_agg_window_minutes),
        write_configuration_map,
        CreateSyntheticReadRecordFn(kNumSynthetic, kDim),
        CreateFakeComputeMauveFn());
    CHECK_OK(fn_factory);
    fn_.reset();
    factory_ = std::move(*fn_factory);
    auto fn = factory_->CreateFn();
    CHECK_OK(fn);
    return std::move(*fn);
  }

  static WriteRequest CreateKeyIdWriteRequest(absl::string_view blob_id,
                                              absl::string_view key_id) {
    WriteRequest request;
    BlobHeader header;
    header.set_key_id(std::string(key_id));
    request.mutable_first_request_metadata()
        ->mutable_hpke_plus_aead_data()
        ->set_blob_id(std::string(blob_id));
    request.mutable_first_request_metadata()
        ->mutable_hpke_plus_aead_data()
        ->mutable_kms_symmetric_key_associated_data()
        ->mutable_associated_metadata()
        ->PackFrom(header);
    return request;
  }

  static WriteRequest CreateTimeWindowWriteRequest(
      absl::string_view blob_id, int64_t start_seconds, int64_t end_seconds,
      absl::string_view key_id = "") {
    WriteRequest request;
    SessionTimeWindowMetadata time_window_metadata;
    time_window_metadata.mutable_session_window_start()->set_seconds(
        start_seconds);
    time_window_metadata.mutable_session_window_end()->set_seconds(end_seconds);
    if (!key_id.empty()) {
      time_window_metadata.add_key_ids(std::string(key_id));
    }
    AssociatedMetadata assoc_metadata;
    assoc_metadata.add_metadata()->PackFrom(time_window_metadata);
    request.mutable_first_request_metadata()
        ->mutable_hpke_plus_aead_data()
        ->set_blob_id(std::string(blob_id));
    request.mutable_first_request_metadata()
        ->mutable_hpke_plus_aead_data()
        ->mutable_kms_symmetric_key_associated_data()
        ->mutable_associated_metadata()
        ->PackFrom(assoc_metadata);
    return request;
  }

  std::unique_ptr<FnFactory> factory_;
  std::unique_ptr<Fn> fn_;
  StrictMock<MockContext> context_;
  Counters counters_;
};

TEST_F(MauveScoreFnTest, WriteAccumulatesData) {
  std::string ckpt = BuildCheckpoint(/*batch=*/10, kDim);
  WriteRequest request;
  auto result = fn_->Write(request, ckpt, context_);
  ASSERT_THAT(result, IsOk());
  // BatchDoFn returns raw checkpoint size, not parsed tensor size.
  EXPECT_GT(result->committed_size_bytes(), 0);
}

TEST_F(MauveScoreFnTest, CommitRejectsInvalidData) {
  // Write() now just buffers raw bytes — validation happens in Do().
  auto write_result = fn_->Write(WriteRequest(), "garbage", context_);
  ASSERT_THAT(write_result, IsOk());  // Write succeeds (just buffering)

  // Commit should fail when Do() tries to parse the garbage data.
  fcp::confidentialcompute::CommitRequest commit_request;
  EXPECT_THAT(fn_->Commit(commit_request, context_),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(MauveScoreFnTest, CommitRejectsWrongTensorDims) {
  FederatedComputeCheckpointBuilderFactory factory;
  auto builder = factory.Create();
  auto t =
      Tensor::Create(DT_FLOAT, TensorShape({2, 2, 2}),
                     std::make_unique<VectorData<float>>(std::vector<float>{
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}));
  CHECK_OK(t);
  CHECK_OK(builder->Add(std::string(kDataTensorName), std::move(*t)));
  auto ckpt = builder->Build();
  CHECK_OK(ckpt);

  // Write succeeds (just buffering).
  auto write_result = fn_->Write(WriteRequest(), std::string(*ckpt), context_);
  ASSERT_THAT(write_result, IsOk());

  // Commit should fail when Do() finds wrong tensor dimensions.
  fcp::confidentialcompute::CommitRequest commit_request;
  EXPECT_THAT(fn_->Commit(commit_request, context_),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(MauveScoreFnTest, CommitWithoutWriteReturnsError) {
  fcp::confidentialcompute::CommitRequest commit_request;
  EXPECT_THAT(fn_->Commit(commit_request, context_),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(MauveScoreFnTest, CommitRejectsDuplicateBlobIds) {
  std::string ckpt = BuildCheckpoint(/*batch=*/10, kDim);

  // Write two blobs with the same blob_id.
  WriteRequest request1;
  request1.mutable_first_request_metadata()->mutable_unencrypted()->set_blob_id(
      "blob-1");
  ASSERT_THAT(fn_->Write(request1, ckpt, context_), IsOk());

  WriteRequest request2;
  request2.mutable_first_request_metadata()->mutable_unencrypted()->set_blob_id(
      "blob-1");
  ASSERT_THAT(fn_->Write(request2, ckpt, context_), IsOk());

  // Commit should fail due to duplicate blob ids.
  fcp::confidentialcompute::CommitRequest commit_request;
  EXPECT_THAT(fn_->Commit(commit_request, context_),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(MauveScoreFnTest, CommitIgnoresExhaustedKeyBasedBlobAndSucceeds) {
  BudgetState state;
  auto* bucket = state.add_buckets();
  bucket->set_key("exhausted-key");
  bucket->set_budget(0);

  auto fn = CreateFnWithState(state.SerializeAsString());

  // Blob 1 has exhausted budget (10 embeddings) -> should be ignored.
  ASSERT_THAT(fn->Write(CreateKeyIdWriteRequest("blob-1", "exhausted-key"),
                        BuildCheckpoint(/*batch=*/10, kDim), context_),
              IsOk());
  // Blob 2 has valid budget (50 embeddings) -> should be processed.
  ASSERT_THAT(fn->Write(CreateKeyIdWriteRequest("blob-2", "valid-key"),
                        BuildCheckpoint(/*batch=*/50, kDim), context_),
              IsOk());

  fcp::confidentialcompute::CommitRequest commit_request;
  EXPECT_THAT(fn->Commit(commit_request, context_), IsOk());
  EXPECT_EQ(counters_["mauve-ignored-exhausted-budget-blobs-count"], 1);
  EXPECT_EQ(counters_["mauve-real-embeddings-count"], 50);
}

TEST_F(MauveScoreFnTest, CommitIgnoresExhaustedTimeBasedBlobAndSucceeds) {
  auto budget =
      Budget::Create(/*initial_state=*/std::nullopt, /*default_budget=*/1);
  ASSERT_THAT(budget, IsOk());
  ASSERT_THAT(budget->UpdateTimeBudget(Interval<uint64_t>(0, 3600)), IsOk());

  auto fn = CreateFnWithState(budget->SerializeAsString(), /*budget_times=*/1);

  // Blob 1 has exhausted time window [0, 3600) (10 embeddings) -> ignored.
  ASSERT_THAT(fn->Write(CreateTimeWindowWriteRequest("blob-1", 0, 3600),
                        BuildCheckpoint(/*batch=*/10, kDim), context_),
              IsOk());
  // Blob 2 has valid time window [3600, 7200) (50 embeddings) -> processed.
  ASSERT_THAT(fn->Write(CreateTimeWindowWriteRequest("blob-2", 3600, 7200),
                        BuildCheckpoint(/*batch=*/50, kDim), context_),
              IsOk());

  fcp::confidentialcompute::CommitRequest commit_request;
  EXPECT_THAT(fn->Commit(commit_request, context_), IsOk());
  EXPECT_EQ(counters_["mauve-ignored-exhausted-budget-blobs-count"], 1);
  EXPECT_EQ(counters_["mauve-real-embeddings-count"], 50);
}

TEST_F(MauveScoreFnTest, CommitFailsWhenAllKeyBasedBlobsHaveExhaustedBudget) {
  BudgetState exhausted_state;
  auto* bucket = exhausted_state.add_buckets();
  bucket->set_key("key-1");
  bucket->set_budget(0);

  auto fn = CreateFnWithState(exhausted_state.SerializeAsString());

  ASSERT_THAT(fn->Write(CreateKeyIdWriteRequest("blob-1", "key-1"),
                        BuildCheckpoint(/*batch=*/10, kDim), context_),
              IsOk());

  fcp::confidentialcompute::CommitRequest commit_request;
  EXPECT_THAT(fn->Commit(commit_request, context_),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_EQ(counters_["mauve-ignored-exhausted-budget-blobs-count"], 1);
}

TEST_F(MauveScoreFnTest, CommitFailsWhenAllTimeBasedBlobsHaveExhaustedBudget) {
  auto budget =
      Budget::Create(/*initial_state=*/std::nullopt, /*default_budget=*/1);
  ASSERT_THAT(budget, IsOk());
  ASSERT_THAT(budget->UpdateTimeBudget(Interval<uint64_t>(0, 3600)), IsOk());

  auto fn = CreateFnWithState(budget->SerializeAsString(), /*budget_times=*/1);

  ASSERT_THAT(
      fn->Write(CreateTimeWindowWriteRequest("blob-1", 0, 3600, "key-1"),
                BuildCheckpoint(/*batch=*/10, kDim), context_),
      IsOk());

  fcp::confidentialcompute::CommitRequest commit_request;
  EXPECT_THAT(fn->Commit(commit_request, context_),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_EQ(counters_["mauve-ignored-exhausted-budget-blobs-count"], 1);
}

TEST_F(MauveScoreFnTest, CommitRejectsTimeWindowTooShort) {
  auto fn = CreateFnWithState(/*initial_state=*/"", /*budget_times=*/5,
                              /*min_agg_window_minutes=*/60);

  // 30 minutes < 60 minutes required.
  ASSERT_THAT(
      fn->Write(CreateTimeWindowWriteRequest("blob-1", 0, 1800, "key-1"),
                BuildCheckpoint(/*batch=*/10, kDim), context_),
      IsOk());

  fcp::confidentialcompute::CommitRequest commit_request;
  EXPECT_THAT(fn->Commit(commit_request, context_),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(MauveScoreFnTest, FullLifecycleSuccess) {
  // Write enough embeddings for MAUVE to work (need >= 2).
  std::string ckpt = BuildCheckpoint(/*batch=*/50, kDim);
  WriteRequest request;
  ASSERT_THAT(fn_->Write(request, ckpt, context_), IsOk());

  // Commit should compute MAUVE and store the result.
  fcp::confidentialcompute::CommitRequest commit_request;
  auto commit_result = fn_->Commit(commit_request, context_);
  ASSERT_THAT(commit_result, IsOk());

  EXPECT_EQ(counters_["mauve-score-computed"], 1);
  EXPECT_EQ(counters_["mauve-real-embeddings-count"], 50);
  EXPECT_EQ(counters_["mauve-synth-embeddings-count"], kNumSynthetic);

  // Finalize should emit the result via EmitReleasable.
  EXPECT_CALL(context_, EmitReleasable(0, _, _, _, _))
      .WillOnce(
          [](int, Session::KV kv, std::optional<absl::string_view> src_state,
             absl::string_view dst_state, std::string& release_token) -> bool {
            // First run: src_state should be nullopt.
            EXPECT_FALSE(src_state.has_value());
            // Verify the dst_state contains a valid BudgetState.
            BudgetState dst;
            EXPECT_TRUE(dst.ParseFromString(std::string(dst_state)));
            // First run with access_budget=5: remaining = 5-1 = 4
            EXPECT_EQ(dst.buckets_size(), 1);
            if (dst.buckets_size() == 1) {
              EXPECT_EQ(dst.buckets(0).budget(), 4);
            }
            release_token = "test-release-token";
            return true;
          });
  auto finalize_result = fn_->Finalize({}, {}, context_);
  ASSERT_THAT(finalize_result, IsOk());
  EXPECT_EQ(finalize_result->release_token(), "test-release-token");
}

}  // anonymous namespace
}  // namespace confidential_federated_compute::mauve_score
