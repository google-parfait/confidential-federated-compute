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
#include "nn_histogram_fn.h"

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "fcp/protos/confidentialcompute/nn_histogram_config.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/any.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"

namespace confidential_federated_compute::nn_histogram {
namespace {
constexpr absl::string_view kBlobId = "blob_id";
constexpr absl::string_view kConfigId = "config_id";
constexpr absl::string_view kPath = "path";

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::confidential_federated_compute::fns::Fn;
using ::confidential_federated_compute::fns::FnFactory;
using ::fcp::confidentialcompute::Embedding;
using ::fcp::confidentialcompute::NNHistogramContainerInitializeConfiguration;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::WriteRequest;
using ::google::protobuf::Any;
using ::tensorflow_federated::aggregation::DT_FLOAT;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorData;
using ::tensorflow_federated::aggregation::TensorShape;
using ::tensorflow_federated::aggregation::VectorData;
using ::testing::_;
using ::testing::Eq;
using ::testing::FieldsAre;
using ::testing::Return;
using ::testing::StrictMock;

class MockContext : public confidential_federated_compute::Session::Context {
 public:
  MOCK_METHOD(bool, Emit, (ReadResponse), (override));
  MOCK_METHOD(bool, EmitUnencrypted, (Session::KV), (override));
  MOCK_METHOD(bool, EmitEncrypted, (int, Session::KV), (override));
};

Any CreateValidInitConfig() {
  Any config;
  NNHistogramContainerInitializeConfiguration init_config;
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

Embedding CreateEmbedding(const std::vector<float>& values, int32_t index = 0) {
  Embedding emb;
  auto* values_proto = emb.mutable_values();
  for (const auto& value : values) {
    *(values_proto->Add()) = value;
  }
  emb.set_index(index);
  return emb;
}

ReadRecordFn CreateValidReadRecordFn() {
  return [](absl::string_view path) -> absl::StatusOr<std::vector<Embedding>> {
    if (path == kPath) {
      std::vector<Embedding> embeddings;
      embeddings.push_back(CreateEmbedding({0.1, 0.2, 0.3}, 100));
      return embeddings;
    } else {
      return absl::InvalidArgumentError("Invalid records path.");
    }
  };
}

NearestNeighborFn CreateValidNearestNeighborFn() {
  return
      [](absl::Span<const float> input_data,
         const std::vector<fcp::confidentialcompute::Embedding>& embeddings) {
        return 100;
      };
}

WriteRequest CreateWriteRequest() {
  WriteRequest request;
  request.mutable_first_request_metadata()
      ->mutable_hpke_plus_aead_data()
      ->set_blob_id(std::string(kBlobId));
  return request;
}

class NNHistogramFnTest : public testing::Test {
 protected:
  void SetUp() override {
    Any config = CreateValidInitConfig();
    auto write_configuration_map = CreateWriteConfigurationMap();
    auto fn_factory = ProvideNNHistogramFnFactory(
        config, Any(), write_configuration_map, CreateValidReadRecordFn(),
        CreateValidNearestNeighborFn());
    ASSERT_THAT(fn_factory, IsOk());
    factory_ = std::move(*fn_factory);

    auto fn = factory_->CreateFn();
    ASSERT_THAT(fn, IsOk());
    fn_ = std::move(*fn);
  }

  std::unique_ptr<FnFactory> factory_;
  std::unique_ptr<Fn> fn_;
  StrictMock<MockContext> context_;
};

TEST(NNHistogramFnFactoryTest, InvalidConfig) {
  EXPECT_THAT(ProvideNNHistogramFnFactory(Any(), Any(), {}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(NNHistogramFnFactoryTest, MissingEmbeddingConfigId) {
  Any config = CreateValidInitConfig();
  absl::flat_hash_map<std::string, std::string> write_configuration_map;
  write_configuration_map["Another_id"] = "some path";

  EXPECT_THAT(
      ProvideNNHistogramFnFactory(config, Any(), write_configuration_map),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(NNHistogramFnFactoryTest, ReadRecordFailed) {
  Any config = CreateValidInitConfig();
  auto write_configuration_map = CreateWriteConfigurationMap();
  ReadRecordFn read_record = [](absl::string_view path) {
    return absl::InvalidArgumentError("Failed to read records");
  };

  EXPECT_THAT(
      ProvideNNHistogramFnFactory(config, Any(), write_configuration_map,
                                  std::move(read_record)),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(NNHistogramFnTest, InvalidInputData) {
  auto result = fn_->Write(WriteRequest(), "Invalid_input_data", context_);
  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(result->status().code(),
              static_cast<int32_t>(absl::StatusCode::kInvalidArgument));
}

TEST_F(NNHistogramFnTest, InvalidMissingRequiredTensor) {
  FederatedComputeCheckpointBuilderFactory factory;
  auto builder = factory.Create();
  Tensor t({0.2, 0.4, 0.6});
  ASSERT_THAT(builder->Add("some_data", std::move(t)), IsOk());
  auto ckpt = builder->Build();
  ASSERT_THAT(ckpt, IsOk());

  auto result = fn_->Write(WriteRequest(), std::string(*ckpt), context_);
  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(result->status().code(),
              static_cast<int32_t>(absl::StatusCode::kNotFound));
}

TEST_F(NNHistogramFnTest, WrongInputTensorType) {
  FederatedComputeCheckpointBuilderFactory factory;
  auto builder = factory.Create();
  Tensor t({12, 14, 43});
  ASSERT_THAT(builder->Add(std::string(kDataTensorName), std::move(t)), IsOk());
  auto ckpt = builder->Build();
  ASSERT_THAT(ckpt, IsOk());

  auto result = fn_->Write(WriteRequest(), std::string(*ckpt), context_);
  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(result->status().code(),
              static_cast<int32_t>(absl::StatusCode::kInvalidArgument));
}

TEST_F(NNHistogramFnTest, WrongInputTensorDimension) {
  FederatedComputeCheckpointBuilderFactory factory;
  auto builder = factory.Create();
  auto t =
      Tensor::Create(DT_FLOAT, TensorShape({2, 2, 2}),
                     std::make_unique<VectorData<float>>(std::vector<float>{
                         1.2, 1.0, 4.3, 4.6, 8.7, 9.0, 1.3, 5.4}));
  ASSERT_THAT(t, IsOk());
  ASSERT_THAT(builder->Add(std::string(kDataTensorName), std::move(*t)),
              IsOk());
  auto ckpt = builder->Build();
  ASSERT_THAT(ckpt, IsOk());

  auto result = fn_->Write(WriteRequest(), std::string(*ckpt), context_);
  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(result->status().code(),
              static_cast<int32_t>(absl::StatusCode::kInvalidArgument));
}

TEST_F(NNHistogramFnTest, MismatchEmbeddingDimension) {
  FederatedComputeCheckpointBuilderFactory factory;
  auto builder = factory.Create();
  auto t = Tensor::Create(DT_FLOAT, TensorShape({2, 2}),
                          std::make_unique<VectorData<float>>(
                              std::vector<float>{0.2, 0.3, 0.4, 0.8}));
  ASSERT_THAT(t, IsOk());
  ASSERT_THAT(builder->Add(std::string(kDataTensorName), std::move(*t)),
              IsOk());
  auto ckpt = builder->Build();
  ASSERT_THAT(ckpt, IsOk());

  auto result = fn_->Write(WriteRequest(), std::string(*ckpt), context_);
  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(result->status().code(),
              static_cast<int32_t>(absl::StatusCode::kInvalidArgument));
}

TEST_F(NNHistogramFnTest, MissingInputBlobId) {
  FederatedComputeCheckpointBuilderFactory factory;
  auto builder = factory.Create();
  auto t = Tensor::Create(
      DT_FLOAT, TensorShape({1, 3}),
      std::make_unique<VectorData<float>>(std::vector<float>{0.3, 0.4, 0.8}));
  ASSERT_THAT(t, IsOk());
  ASSERT_THAT(builder->Add(std::string(kDataTensorName), std::move(*t)),
              IsOk());
  auto ckpt = builder->Build();
  ASSERT_THAT(ckpt, IsOk());

  auto result = fn_->Write(WriteRequest(), std::string(*ckpt), context_);
  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(result->status().code(),
              static_cast<int32_t>(absl::StatusCode::kInvalidArgument));
}

TEST_F(NNHistogramFnTest, Success) {
  WriteRequest request = CreateWriteRequest();

  FederatedComputeCheckpointBuilderFactory factory;
  auto builder = factory.Create();
  auto t =
      Tensor::Create(DT_FLOAT, TensorShape({2, 3}),
                     std::make_unique<VectorData<float>>(
                         std::vector<float>{0.3, 0.4, 0.8, 1.6, 1.4, 1.7}));
  ASSERT_THAT(t, IsOk());
  ASSERT_THAT(builder->Add(std::string(kDataTensorName), std::move(*t)),
              IsOk());
  auto ckpt = builder->Build();
  ASSERT_THAT(ckpt, IsOk());
  int64_t commit_size = ckpt->size();

  builder = factory.Create();
  Tensor indices({100, 100});
  ASSERT_THAT(builder->Add(std::string(kIndexTensorName), std::move(indices)),
              IsOk());
  auto output_ckpt = builder->Build();
  ASSERT_THAT(output_ckpt, IsOk());

  EXPECT_CALL(context_,
              EmitEncrypted(Eq(0), FieldsAre(_, Eq(std::string(*output_ckpt)),
                                             Eq(std::string(kBlobId)))))
      .WillOnce(Return(true));

  auto result = fn_->Write(request, std::string(*ckpt), context_);
  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(result->committed_size_bytes(), commit_size);
}

TEST_F(NNHistogramFnTest, EmitFailed) {
  WriteRequest request = CreateWriteRequest();

  FederatedComputeCheckpointBuilderFactory factory;
  auto builder = factory.Create();
  auto t =
      Tensor::Create(DT_FLOAT, TensorShape({2, 3}),
                     std::make_unique<VectorData<float>>(
                         std::vector<float>{0.3, 0.4, 0.8, 1.6, 1.4, 1.7}));
  ASSERT_THAT(t, IsOk());
  ASSERT_THAT(builder->Add(std::string(kDataTensorName), std::move(*t)),
              IsOk());
  auto ckpt = builder->Build();
  ASSERT_THAT(ckpt, IsOk());
  EXPECT_CALL(context_, EmitEncrypted(_, _)).WillOnce(Return(false));

  auto result = fn_->Write(request, std::string(*ckpt), context_);
  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(result->status().code(),
              static_cast<int32_t>(absl::StatusCode::kInvalidArgument));
}

TEST_F(NNHistogramFnTest, CalculateNearestNeighborFailed) {
  Any config = CreateValidInitConfig();
  auto write_configuration_map = CreateWriteConfigurationMap();
  auto fn_factory = ProvideNNHistogramFnFactory(
      config, Any(), write_configuration_map, CreateValidReadRecordFn(),
      [](absl::Span<const float>,
         const std::vector<fcp::confidentialcompute::Embedding>&)
          -> absl::StatusOr<int32_t> {
        return absl::InternalError("something is wrong");
      });
  ASSERT_THAT(fn_factory, IsOk());

  auto fn = (*fn_factory)->CreateFn();
  ASSERT_THAT(fn, IsOk());

  WriteRequest request = CreateWriteRequest();

  FederatedComputeCheckpointBuilderFactory factory;
  auto builder = factory.Create();
  auto t =
      Tensor::Create(DT_FLOAT, TensorShape({2, 3}),
                     std::make_unique<VectorData<float>>(
                         std::vector<float>{0.3, 0.4, 0.8, 1.6, 1.4, 1.7}));
  ASSERT_THAT(t, IsOk());
  ASSERT_THAT(builder->Add(std::string(kDataTensorName), std::move(*t)),
              IsOk());
  auto ckpt = builder->Build();
  ASSERT_THAT(ckpt, IsOk());

  auto result = (*fn)->Write(request, std::string(*ckpt), context_);
  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(result->status().code(),
              static_cast<int32_t>(absl::StatusCode::kInternal));
}

}  // anonymous namespace
}  // namespace confidential_federated_compute::nn_histogram
