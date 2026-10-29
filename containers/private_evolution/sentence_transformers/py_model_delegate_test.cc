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

#include "py_model_delegate.h"

#include <pybind11/embed.h>

#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute::sentence_transformers {
namespace {

namespace py = ::pybind11;
using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::Each;
using ::testing::IsNan;
using ::testing::Not;
using ::testing::SizeIs;

// The embedding dimension of the model that SaveTinyModel() creates.
constexpr int kEmbeddingDimension = 8;

// The result of PyRuntimeManager::ImportLib() in the test environment.
absl::Status import_status = absl::UnknownError("ImportLib() didn't run");

// A process can initialize Python only once, so all tests share one
// PyRuntimeManager.
class PyRuntimeEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    manager_.emplace();
    // Imports the sentence-transformers stack (torch, transformers, ...) with
    // the Python environment that ships in the container (:py_env). main.cc
    // imports on the main thread; this checks that a gRPC worker thread can
    // still be the first thread to acquire the GIL, which crashed when the GIL
    // was released with PyEval_SaveThread().
    std::thread worker([this] { import_status = manager_->ImportLib(); });
    worker.join();
  }

  void TearDown() override { manager_.reset(); }

 private:
  std::optional<PyRuntimeManager> manager_;
};

const ::testing::Environment* const kEnvironment =
    ::testing::AddGlobalTestEnvironment(new PyRuntimeEnvironment);

// Saves a small BERT model with random weights in the sentence-transformers
// format to `directory`/model, so that the test doesn't need to download one.
void SaveTinyModel(const std::string& directory) {
  py::gil_scoped_acquire gil;
  py::dict scope;
  scope["directory"] = directory;
  scope["hidden_size"] = kEmbeddingDimension;
  py::exec(R"py(
import os

from sentence_transformers import SentenceTransformer, models
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

hf_directory = os.path.join(directory, "hf")
os.makedirs(hf_directory)
vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world",
         "query", ":"]
vocab_file = os.path.join(hf_directory, "vocab.txt")
with open(vocab_file, "w") as f:
  f.write("\n".join(vocab) + "\n")
BertTokenizerFast(vocab_file).save_pretrained(hf_directory)
torch.manual_seed(0)
config = BertConfig(vocab_size=len(vocab), hidden_size=hidden_size,
                    num_hidden_layers=1, num_attention_heads=2,
                    intermediate_size=16, max_position_embeddings=32)
BertModel(config).save_pretrained(hf_directory)
transformer = models.Transformer(hf_directory, max_seq_length=16)
pooling = models.Pooling(transformer.get_word_embedding_dimension())
SentenceTransformer(modules=[transformer, pooling], device="cpu").save(
    os.path.join(directory, "model"))
)py",
           scope);
}

TEST(PyRuntimeManagerTest, ImportsLibOnWorkerThread) {
  EXPECT_THAT(import_status, IsOk());
}

// Loads a model and generates embeddings with it on a thread other than the
// one that initialized the interpreter, as the gRPC server does.
TEST(PyModelDelegateTest, GeneratesEmbeddings) {
  ASSERT_THAT(import_status, IsOk());
  const std::string directory =
      absl::StrCat(::testing::TempDir(), "tiny_model");
  SaveTinyModel(directory);

  PyModelDelegate delegate;
  bool initialized_from_missing_path = true;
  bool initialized = false;
  absl::StatusOr<std::vector<std::vector<float>>> before_initialization;
  absl::StatusOr<std::vector<std::vector<float>>> embeddings;
  absl::StatusOr<std::vector<std::vector<float>>> embeddings_with_prompt;
  std::thread worker([&] {
    initialized_from_missing_path =
        delegate.InitializeModel(absl::StrCat(directory, "/missing"));
    before_initialization =
        delegate.GenerateEmbeddings({"hello"}, std::nullopt);
    initialized = delegate.InitializeModel(absl::StrCat(directory, "/model"));
    embeddings =
        delegate.GenerateEmbeddings({"hello world", "world"}, std::nullopt);
    embeddings_with_prompt =
        delegate.GenerateEmbeddings({"hello world"}, "query: ");
  });
  worker.join();

  // Python errors are reported to the caller rather than crashing.
  EXPECT_FALSE(initialized_from_missing_path);
  EXPECT_THAT(before_initialization,
              StatusIs(absl::StatusCode::kInvalidArgument));

  ASSERT_TRUE(initialized);
  ASSERT_THAT(embeddings, IsOk());
  ASSERT_THAT(*embeddings, SizeIs(2));
  EXPECT_THAT(*embeddings, Each(SizeIs(kEmbeddingDimension)));
  EXPECT_THAT(*embeddings, Each(Each(Not(IsNan()))));
  EXPECT_NE((*embeddings)[0], (*embeddings)[1]);

  // The prompt is prepended to the input, so the embedding changes.
  ASSERT_THAT(embeddings_with_prompt, IsOk());
  ASSERT_THAT(*embeddings_with_prompt, SizeIs(1));
  EXPECT_THAT((*embeddings_with_prompt)[0], SizeIs(kEmbeddingDimension));
  EXPECT_NE((*embeddings_with_prompt)[0], (*embeddings)[0]);
}

}  // namespace
}  // namespace confidential_federated_compute::sentence_transformers
