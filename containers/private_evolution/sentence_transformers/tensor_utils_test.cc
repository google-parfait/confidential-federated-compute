// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may not obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "tensor_utils.h"

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "gmock/gmock.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace confidential_federated_compute::sentence_transformers {
namespace {

using ::absl_testing::StatusIs;
using ::tensorflow_federated::aggregation::DT_FLOAT;
using ::tensorflow_federated::aggregation::Tensor;
using ::testing::ElementsAre;
using ::testing::Eq;

TEST(TensorUtilsTest, NoEmbeddingData) {
  EXPECT_THAT(CreateEmbeddingTensor({}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(TensorUtilsTest, EmptyEmbeddingVector) {
  std::vector<std::vector<float>> embeddings;
  embeddings.push_back(std::vector<float>{0.1, 0.2, 0.3});
  embeddings.push_back(std::vector<float>{});
  EXPECT_THAT(CreateEmbeddingTensor(embeddings),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(TensorUtilsTest, InnerVectorHasDifferentShapes) {
  std::vector<std::vector<float>> embeddings;
  std::vector<float> emb_vec_1{0.1, 0.2, 0.3};
  std::vector<float> emb_vec_2{0.4, 0.5};
  embeddings.push_back(emb_vec_1);
  embeddings.push_back(emb_vec_2);
  EXPECT_THAT(CreateEmbeddingTensor(embeddings),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(TensorUtilsTest, Success) {
  std::vector<std::vector<float>> embeddings;
  std::vector<float> emb_vec_1{0.1, 0.2, 0.3};
  std::vector<float> emb_vec_2{0.4, 0.5, 0.6};
  embeddings.push_back(emb_vec_1);
  embeddings.push_back(emb_vec_2);
  auto t = CreateEmbeddingTensor(embeddings);
  ASSERT_TRUE(t.ok());
  EXPECT_THAT(t->dtype(), Eq(DT_FLOAT));

  absl::Span<const float> buffer = t->AsSpan<float>();
  EXPECT_THAT(buffer, ElementsAre(0.1, 0.2, 0.3, 0.4, 0.5, 0.6));
}

}  // anonymous namespace
}  // namespace confidential_federated_compute::sentence_transformers