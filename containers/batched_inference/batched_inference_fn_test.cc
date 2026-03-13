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

#include "containers/batched_inference/batched_inference_fn.h"

#include <gtest/gtest.h>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "containers/batched_inference/batched_inference_test_utils.h"
#include "containers/session.h"
#include "containers/testing/mocks.h"
#include "gmock/gmock.h"
#include "google/protobuf/any.pb.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute::batched_inference {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::confidential_federated_compute::Session;
using ::fcp::confidentialcompute::CommitResponse;
using ::fcp::confidentialcompute::InferenceConfiguration;
using ::fcp::confidentialcompute::StreamInitializeRequest;
using ::fcp::confidentialcompute::WriteFinishedResponse;
using ::google::protobuf::Any;
using ::testing::_;
using ::testing::Eq;
using ::testing::Field;
using ::testing::Invoke;
using ::testing::Mock;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::Test;

class MockBatchedInferenceProvider : public BatchedInferenceProvider {
 public:
  MOCK_METHOD((std::vector<absl::StatusOr<std::string>>), DoBatchedInference,
              (std::vector<std::string> prompts), (override));
};

class BatchedInferenceFnTest : public Test {
 protected:
  void SetUp() override {}

  // First vector goes over commits, second over blobs, third over rows within a
  // blob.
  void RunTestCaseFor(
      std::vector<std::vector<std::vector<std::string>>> commits) {
    std::shared_ptr<NiceMock<MockBatchedInferenceProvider>> mock_provider =
        std::make_shared<NiceMock<MockBatchedInferenceProvider>>();
    InferenceConfiguration inference_config =
        testing::GetInferenceConfigForTest();
    absl::StatusOr<std::unique_ptr<fns::FnFactory>> factory =
        CreateBatchedInferenceFnFactory(mock_provider, inference_config);
    ASSERT_THAT(factory.status(), IsOk());
    auto fn = factory.value()->CreateFn();
    ASSERT_THAT(fn.status(), IsOk());
    MockContext mock_context;
    EXPECT_CALL(*mock_provider, DoBatchedInference(_)).Times(0);
    EXPECT_CALL(mock_context, EmitEncrypted(_, _)).Times(0);
    int commit_no = 0;
    for (auto& commit : commits) {
      ++commit_no;

      // Write all the blobs first; nothing should happen until we commit.
      int total_num_inference_calls = 0;
      int total_num_rows = 0;
      int blob_no = 0;
      for (auto& blob : commit) {
        ++blob_no;
        std::string blob_id =
            absl::StrCat("some_blob_", commit_no, "_", blob_no);
        total_num_rows += blob.size();
        fcp::confidentialcompute::WriteRequest write_request;
        write_request.mutable_first_request_metadata()
            ->mutable_unencrypted()
            ->set_blob_id(blob_id);
        *write_request.mutable_first_request_configuration() = Any();
        std::string unencrypted_data =
            testing::GetPrivateInferenceInputCheckpointForTest(blob);
        absl::StatusOr<WriteFinishedResponse> write_result =
            fn.value()->Write(write_request, unencrypted_data, mock_context);
        EXPECT_THAT(write_result.status(), IsOk());
        Mock::VerifyAndClearExpectations(mock_provider.get());
        Mock::VerifyAndClearExpectations(&mock_context);
      }

      // Now commit and verify inference calls and integrity of the results.
      EXPECT_CALL(*mock_provider, DoBatchedInference(_))
          .WillRepeatedly(Invoke(
              [&total_num_inference_calls](std::vector<std::string> prompts) {
                ++total_num_inference_calls;
                std::vector<absl::StatusOr<std::string>> results;
                for (const auto& prompt : prompts) {
                  results.push_back("Processed: " + prompt);
                }
                return results;
              }));
      blob_no = 0;
      for (auto& blob : commit) {
        ++blob_no;
        std::string blob_id =
            absl::StrCat("some_blob_", commit_no, "_", blob_no);
        std::vector<std::string> results;
        for (auto& prompt : blob) {
          results.push_back(absl::StrCat("Processed: Hello, ", prompt));
        }
        std::string unencrypted_data =
            testing::GetPrivateInferenceOutputCheckpointForTest(blob, results);
        EXPECT_CALL(
            mock_context,
            EmitEncrypted(
                0, AllOf(Field(&Session::KV::blob_id, Eq(blob_id)),
                         Field(&Session::KV::data, Eq(unencrypted_data)))))
            .Times(1)
            .WillOnce(Return(true));
      }
      fcp::confidentialcompute::CommitRequest commit_request;
      absl::StatusOr<CommitResponse> commit_result =
          fn.value()->Commit(commit_request, mock_context);
      EXPECT_THAT(commit_result.status(), IsOk());
      Mock::VerifyAndClearExpectations(mock_provider.get());
      Mock::VerifyAndClearExpectations(&mock_context);

      const int expected_total_num_inference_calls = static_cast<int>(
          std::ceil(static_cast<double>(total_num_rows) /
                    inference_config.runtime_config().max_batch_size()));
      EXPECT_EQ(expected_total_num_inference_calls, total_num_inference_calls);
    }
  }
};

TEST_F(BatchedInferenceFnTest, OneCommitOneBlobOneRow) {
  RunTestCaseFor({{{"bark"}}});
}

TEST_F(BatchedInferenceFnTest, OneCommitOneBlobTwoRows) {
  RunTestCaseFor({{{"bark", "oink"}}});
}

TEST_F(BatchedInferenceFnTest, OneCommitOneBlobThreeRows) {
  RunTestCaseFor({{{"bark", "oink", "meaow"}}});
}

TEST_F(BatchedInferenceFnTest, OneCommitOneBlobFourRows) {
  RunTestCaseFor({{{"bark", "oink", "meaow", "kwakwa"}}});
}

TEST_F(BatchedInferenceFnTest, OneCommitTwoBlobs) {
  RunTestCaseFor({{{"bark"}, {"oink"}}});
}

TEST_F(BatchedInferenceFnTest, TwoCommits) {
  RunTestCaseFor({{{"bark"}, {"oink"}}, {{"meaow", "kwakwa"}}});
}

TEST_F(BatchedInferenceFnTest, LotsOfEverything) {
  const int kNumCommits = 10;
  const int kNumBlobsPerCommit = 10;
  const int kNumRowsPerBlob = 10;
  std::vector<std::vector<std::vector<std::string>>> commits;
  for (int x = 1; x <= kNumCommits; ++x) {
    std::vector<std::vector<std::string>> blobs;
    for (int y = 1; y <= kNumBlobsPerCommit; ++y) {
      std::vector<std::string> rows;
      for (int z = 1; z <= kNumRowsPerBlob; ++z) {
        rows.push_back(absl::StrCat("commit_", x, "_blob_", y, "_row_", z));
      }
      blobs.push_back(rows);
    }
    commits.push_back(blobs);
  }
  RunTestCaseFor(commits);
}

}  // namespace
}  // namespace confidential_federated_compute::batched_inference
