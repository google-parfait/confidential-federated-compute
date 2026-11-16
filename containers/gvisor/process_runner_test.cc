// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "process_runner.h"

#include <csignal>
#include <string>

#include "absl/status/status_matchers.h"
#include "absl/time/time.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute::gvisor {
namespace {

using ::absl_testing::IsOk;
using ::testing::Not;

class RunProcessTest : public ::testing::Test {
 protected:
  // The container's main() ignores SIGPIPE; do the same here.
  static void SetUpTestSuite() { std::signal(SIGPIPE, SIG_IGN); }
};

// More than the 64 KiB pipe buffer, with every byte value including NUL.
std::string BinaryInput() {
  std::string input;
  for (int i = 0; i < (1 << 20); ++i) input.push_back(static_cast<char>(i));
  return input;
}

TEST_F(RunProcessTest, PassesBinaryStdinToStdout) {
  std::string input = BinaryInput();
  absl::StatusOr<ProcessResult> result =
      RunProcess({"/bin/cat"}, input, absl::Seconds(30), input.size());
  ASSERT_THAT(result, IsOk());
  EXPECT_EQ(result->exit_code, 0);
  EXPECT_FALSE(result->timed_out);
  EXPECT_EQ(result->stdout_data, input);
  EXPECT_FALSE(result->stdout_truncated);
  EXPECT_EQ(result->stderr_data, "");
}

TEST_F(RunProcessTest, CapturesStderrAndExitCode) {
  absl::StatusOr<ProcessResult> result =
      RunProcess({"/bin/sh", "-c", "echo out; echo err >&2; exit 3"}, "",
                 absl::Seconds(30));
  ASSERT_THAT(result, IsOk());
  EXPECT_EQ(result->exit_code, 3);
  EXPECT_EQ(result->stdout_data, "out\n");
  EXPECT_EQ(result->stderr_data, "err\n");
}

TEST_F(RunProcessTest, KillsProcessAtDeadline) {
  absl::Time start = absl::Now();
  absl::StatusOr<ProcessResult> result =
      RunProcess({"/bin/sleep", "30"}, "", absl::Milliseconds(200));
  ASSERT_THAT(result, IsOk());
  EXPECT_TRUE(result->timed_out);
  EXPECT_EQ(result->exit_code, 128 + SIGKILL);
  EXPECT_LT(absl::Now() - start, absl::Seconds(10));
}

TEST_F(RunProcessTest, KillsProcessThatFloodsStdoutAtDeadline) {
  absl::Time start = absl::Now();
  absl::StatusOr<ProcessResult> result =
      RunProcess({"/bin/cat", "/dev/zero"}, "", absl::Milliseconds(200),
                 /*max_output_bytes=*/1000);
  ASSERT_THAT(result, IsOk());
  EXPECT_TRUE(result->timed_out);
  EXPECT_EQ(result->exit_code, 128 + SIGKILL);
  EXPECT_EQ(result->stdout_data.size(), 1000);
  EXPECT_TRUE(result->stdout_truncated);
  EXPECT_LT(absl::Now() - start, absl::Seconds(10));
}

TEST_F(RunProcessTest, TruncatesOutputAtLimit) {
  absl::StatusOr<ProcessResult> result = RunProcess(
      {"/bin/sh", "-c", "head -c 100000 /dev/zero; head -c 10 /dev/zero >&2"},
      "", absl::Seconds(30), /*max_output_bytes=*/1000);
  ASSERT_THAT(result, IsOk());
  EXPECT_EQ(result->exit_code, 0);
  EXPECT_EQ(result->stdout_data, std::string(1000, '\0'));
  EXPECT_TRUE(result->stdout_truncated);
  EXPECT_EQ(result->stderr_data, std::string(10, '\0'));
  EXPECT_FALSE(result->stderr_truncated);
}

TEST_F(RunProcessTest, OutputExactlyAtLimitIsNotTruncated) {
  absl::StatusOr<ProcessResult> result =
      RunProcess({"/bin/sh", "-c", "head -c 1000 /dev/zero"}, "",
                 absl::Seconds(30), /*max_output_bytes=*/1000);
  ASSERT_THAT(result, IsOk());
  EXPECT_EQ(result->stdout_data.size(), 1000);
  EXPECT_FALSE(result->stdout_truncated);
}

TEST_F(RunProcessTest, ChildThatIgnoresStdinDoesNotHang) {
  absl::StatusOr<ProcessResult> result =
      RunProcess({"/bin/true"}, BinaryInput(), absl::Seconds(30));
  ASSERT_THAT(result, IsOk());
  EXPECT_EQ(result->exit_code, 0);
  EXPECT_FALSE(result->timed_out);
}

TEST_F(RunProcessTest, InfiniteTimeoutIsAllowed) {
  absl::StatusOr<ProcessResult> result =
      RunProcess({"/bin/true"}, "", absl::InfiniteDuration());
  ASSERT_THAT(result, IsOk());
  EXPECT_EQ(result->exit_code, 0);
  EXPECT_FALSE(result->timed_out);
}

TEST_F(RunProcessTest, MissingBinaryIsAnError) {
  EXPECT_THAT(RunProcess({"/nonexistent/binary"}, "", absl::Seconds(5)),
              Not(IsOk()));
}

TEST_F(RunProcessTest, EmptyArgvIsAnError) {
  EXPECT_THAT(RunProcess({}, "", absl::Seconds(5)), Not(IsOk()));
}

}  // namespace
}  // namespace confidential_federated_compute::gvisor
