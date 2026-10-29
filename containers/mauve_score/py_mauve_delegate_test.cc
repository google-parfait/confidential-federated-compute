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

#include "py_mauve_delegate.h"

#include <random>
#include <thread>
#include <vector>

#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "fcp/protos/confidentialcompute/mauve_score_config.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute::mauve_score {
namespace {

using ::absl_testing::IsOk;
using ::testing::AllOf;
using ::testing::Ge;
using ::testing::Le;

std::vector<std::vector<float>> RandomEmbeddings(int count, int dim, float mean,
                                                 unsigned seed) {
  std::mt19937 generator(seed);
  std::normal_distribution<float> distribution(mean, 1.0f);
  std::vector<std::vector<float>> embeddings(count, std::vector<float>(dim));
  for (std::vector<float>& embedding : embeddings) {
    for (float& value : embedding) value = distribution(generator);
  }
  return embeddings;
}

// Runs the MAUVE computation with the Python environment that ships in the
// container (:py_env), on a thread other than the one that initialized the
// interpreter, as the gRPC server does.
TEST(PyMauveDelegateTest, ComputesMauveScoreOnWorkerThread) {
  PyRuntimeManager py_manager;
  ASSERT_THAT(py_manager.ImportLib(), IsOk());

  absl::StatusOr<fcp::confidentialcompute::MauveScoreResult> result;
  std::thread worker([&result] {
    result = ComputeMauveViaPython(
        RandomEmbeddings(/*count=*/60, /*dim=*/8, /*mean=*/0.0f, /*seed=*/1),
        RandomEmbeddings(/*count=*/60, /*dim=*/8, /*mean=*/0.5f, /*seed=*/2));
  });
  worker.join();

  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(result->mauve_auc(), AllOf(Ge(0.0), Le(1.0)));
  EXPECT_GE(result->num_clusters(), 2);
}

}  // namespace
}  // namespace confidential_federated_compute::mauve_score
