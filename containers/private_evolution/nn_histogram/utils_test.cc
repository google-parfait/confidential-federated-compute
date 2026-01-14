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
#include "utils.h"

#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "riegeli/bytes/fd_writer.h"
#include "riegeli/records/record_writer.h"

namespace confidential_federated_compute::nn_histogram {
namespace {
using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::fcp::confidentialcompute::Embedding;
using ::testing::ElementsAre;

absl::Status WriteRecords(const std::vector<Embedding>& embeddings,
                          absl::string_view file_path) {
  riegeli::RecordWriter record_writer(
      riegeli::Maker<riegeli::FdWriter>(std::string(file_path)));
  for (const auto& record : embeddings) {
    record_writer.WriteRecord(record);
  }
  if (record_writer.Close()) {
    return absl::OkStatus();
  } else {
    return record_writer.status();
  }
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

class ReadRecordTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::filesystem::path temp_base = std::filesystem::temp_directory_path();
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    int random_number = std::rand();
    temp_file_ = (temp_base / std::to_string(random_number)).string();
  }

  void TearDown() override { std::remove(temp_file_.data()); }

  std::string temp_file_;
};

TEST_F(ReadRecordTest, Success) {
  Embedding embedding_1 = CreateEmbedding({0.1, 0.2});
  Embedding embedding_2 = CreateEmbedding({0.3, 0.4});
  ASSERT_THAT(WriteRecords({embedding_1, embedding_2}, temp_file_), IsOk());

  auto result = ReadRecords(temp_file_);
  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(result->size(), 2);
  EXPECT_THAT((*result)[0].values(), ElementsAre(0.1, 0.2));
  EXPECT_THAT((*result)[1].values(), ElementsAre(0.3, 0.4));
}

TEST_F(ReadRecordTest, InvalidFile) {
  std::ofstream outFile(temp_file_);
  outFile << "Some texts";
  outFile.close();

  auto result = ReadRecords(temp_file_);
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(NearestNeighborTest, EmptyInput) {
  EXPECT_THAT(
      FindNearestNeighbor(std::vector<float>{}, {CreateEmbedding({0.1, 0.2})}),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(NearestNeighborTest, MismatchEmbeddingSize) {
  EXPECT_THAT(FindNearestNeighbor(absl::Span<const float>({0.1}),
                                  {CreateEmbedding({0.1, 0.2})}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(NearestNeighborTest, Success) {
  Embedding embedding_1 = CreateEmbedding({0.6, 0.4, 0.2}, 12);
  Embedding embedding_2 = CreateEmbedding({0.1, 0.2, 0.3}, 43);
  Embedding embedding_3 = CreateEmbedding({0.3, 0.1, 0.2}, 98);

  auto result = FindNearestNeighbor(absl::Span<const float>({0.1, 0.2, 0.3}),
                                    {embedding_1, embedding_2, embedding_3});
  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(*result, 43);
}

}  // namespace
}  // namespace confidential_federated_compute::nn_histogram
