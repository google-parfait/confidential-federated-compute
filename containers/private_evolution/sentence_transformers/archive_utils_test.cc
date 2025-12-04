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
#include "archive_utils.h"

#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace confidential_federated_compute::sentence_transformers {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::bazel::tools::cpp::runfiles::Runfiles;
using ::testing::Eq;

class ArchiveUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::filesystem::path temp_base = std::filesystem::temp_directory_path();
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    int random_number = std::rand();
    temp_dir_ = temp_base / std::to_string(random_number);
    if (!std::filesystem::exists(temp_dir_)) {
      ASSERT_TRUE(std::filesystem::create_directory(temp_dir_));
    }
  }

  void TearDown() override { std::filesystem::remove_all(temp_dir_); }

  std::filesystem::path temp_dir_;
};

TEST_F(ArchiveUtilsTest, InvalidArchivePath) {
  std::string invalid_path = "my_archive.zip";
  auto result = ExtractAll(invalid_path);
  EXPECT_THAT(result.status(), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(ArchiveUtilsTest, InvalidArchiveFile) {
  std::filesystem::path archive_path = temp_dir_ / "model.zip";
  std::ofstream outfile(archive_path);
  outfile << "Invalid zip file." << std::endl;
  outfile.close();

  auto result = ExtractAll(archive_path.string(), temp_dir_.string());
  EXPECT_THAT(result.status(), StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(ArchiveUtilsTest, Success) {
  std::string error;
  std::unique_ptr<Runfiles> runfiles(Runfiles::CreateForTest(&error));
  ASSERT_NE(runfiles, nullptr) << error;

  // Get the path to your data file
  std::string archive_path = runfiles->Rlocation("_main/test/my_model.zip");
  ASSERT_FALSE(archive_path.empty());
  auto result = ExtractAll(archive_path, temp_dir_.string());
  ASSERT_THAT(result.status(), IsOk());
  auto expected_path = temp_dir_ / "my_model";
  EXPECT_THAT(*result, Eq(expected_path.string()));
  ASSERT_TRUE(std::filesystem::exists(expected_path / "model.config"));
  ASSERT_TRUE(std::filesystem::exists(expected_path / "tokenizer.config"));
}

}  // anonymous namespace
}  // namespace confidential_federated_compute::sentence_transformers
