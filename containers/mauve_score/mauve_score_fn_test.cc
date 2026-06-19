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
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "containers/fns/fn_factory.h"
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
absl::StatusOr<std::unique_ptr<fns::FnFactory>> ProvideMauveScoreFnFactory(
    const google::protobuf::Any& configuration,
    const google::protobuf::Any& config_constraints,
    const confidential_federated_compute::fns::WriteConfigurationMap&
        write_configuration_map,
    ReadRecordFn read_record_fn);

namespace {

constexpr absl::string_view kDataTensorName = "data";
constexpr absl::string_view kConfigId = "config_id";
constexpr absl::string_view kPath = "path";

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::confidential_federated_compute::fns::Fn;
using ::confidential_federated_compute::fns::FnFactory;
using ::fcp::confidentialcompute::Embedding;
using ::fcp::confidentialcompute::MauveScoreContainerInitializeConfiguration;
using ::fcp::confidentialcompute::MauveScoreResult;
using ::fcp::confidentialcompute::ReadResponse;
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

absl::flat_hash_map<std::string, std::string> CreateWriteConfigurationMap() {
  absl::flat_hash_map<std::string, std::string> write_configuration_map;
  write_configuration_map[std::string(kConfigId)] = std::string(kPath);
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
  absl::flat_hash_map<std::string, std::string> write_configuration_map;
  write_configuration_map["other_id"] = "some_path";
  ReadRecordFn unused_reader = [](absl::string_view) {
    return absl::InternalError("should not be called");
  };
  EXPECT_THAT(ProvideMauveScoreFnFactory(config, Any(), write_configuration_map,
                                         std::move(unused_reader)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MauveScoreFnFactoryTest, ReadRecordFailed) {
  Any config = CreateValidInitConfig();
  auto write_configuration_map = CreateWriteConfigurationMap();
  ReadRecordFn bad_reader = [](absl::string_view) {
    return absl::InvalidArgumentError("read failed");
  };
  EXPECT_THAT(ProvideMauveScoreFnFactory(config, Any(), write_configuration_map,
                                         std::move(bad_reader)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MauveScoreFnFactoryTest, ValidConfigCreatesFactory) {
  Any config = CreateValidInitConfig();
  auto write_configuration_map = CreateWriteConfigurationMap();
  auto factory = ProvideMauveScoreFnFactory(
      config, Any(), write_configuration_map, CreateSyntheticReadRecordFn());
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
    auto write_configuration_map = CreateWriteConfigurationMap();
    auto fn_factory = ProvideMauveScoreFnFactory(
        config, Any(), write_configuration_map,
        CreateSyntheticReadRecordFn(kNumSynthetic, kDim));
    ASSERT_THAT(fn_factory, IsOk());
    factory_ = std::move(*fn_factory);

    auto fn = factory_->CreateFn();
    ASSERT_THAT(fn, IsOk());
    fn_ = std::move(*fn);
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
  // PObjectMapFn returns raw checkpoint size, not parsed tensor size.
  EXPECT_GT(result->committed_size_bytes(), 0);
}

TEST_F(MauveScoreFnTest, CommitRejectsInvalidData) {
  // Write() now just buffers raw bytes — validation happens in Map().
  auto write_result = fn_->Write(WriteRequest(), "garbage", context_);
  ASSERT_THAT(write_result, IsOk());  // Write succeeds (just buffering)

  // Commit should fail when Map() tries to parse the garbage data.
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

  // Commit should fail when Map() finds wrong tensor dimensions.
  fcp::confidentialcompute::CommitRequest commit_request;
  EXPECT_THAT(fn_->Commit(commit_request, context_),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(MauveScoreFnTest, CommitWithoutWriteReturnsError) {
  fcp::confidentialcompute::CommitRequest commit_request;
  EXPECT_THAT(fn_->Commit(commit_request, context_),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(MauveScoreFnTest, FullLifecycleSuccess) {
  // Write enough embeddings for MAUVE to work (need >= 2).
  std::string ckpt = BuildCheckpoint(/*batch=*/50, kDim);
  WriteRequest request;
  ASSERT_THAT(fn_->Write(request, ckpt, context_), IsOk());

  // Set up expectations for Commit (which calls Map()).
  EXPECT_CALL(context_, EmitUnencrypted(_)).WillOnce(Return(true));
  EXPECT_CALL(context_, GetCounters())
      .WillRepeatedly(::testing::ReturnRef(counters_));

  // Commit should compute MAUVE and emit the result.
  fcp::confidentialcompute::CommitRequest commit_request;
  auto commit_result = fn_->Commit(commit_request, context_);
  ASSERT_THAT(commit_result, IsOk());

  EXPECT_EQ(counters_["mauve-score-computed"], 1);
  EXPECT_EQ(counters_["mauve-real-embeddings-count"], 50);
  EXPECT_EQ(counters_["mauve-synth-embeddings-count"], kNumSynthetic);

  // Finalize is a no-op after Commit.
  auto finalize_result = fn_->Finalize({}, {}, context_);
  ASSERT_THAT(finalize_result, IsOk());
}

}  // anonymous namespace
}  // namespace confidential_federated_compute::mauve_score
