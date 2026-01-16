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
using fcp::confidentialcompute::CommitResponse;
using fcp::confidentialcompute::WriteFinishedResponse;
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
};

TEST_F(BatchedInferenceFnTest, MultipleBatches) {
  std::shared_ptr<NiceMock<MockBatchedInferenceProvider>> mock_provider =
      std::make_shared<NiceMock<MockBatchedInferenceProvider>>();
  absl::StatusOr<std::unique_ptr<fns::FnFactory>> factory =
      CreateBatchedInferenceFnFactory(mock_provider);
  ASSERT_THAT(factory.status(), IsOk());

  auto fn = factory.value()->CreateFn();
  ASSERT_THAT(fn.status(), IsOk());

  MockContext mock_context;

  const int kNumBatches = 2;
  for (int batch_no = 1; batch_no <= kNumBatches; ++batch_no) {
    // Nothing happens during the first 3 writes since no Commit() was sent.
    EXPECT_CALL(*mock_provider, DoBatchedInference(_)).Times(0);
    EXPECT_CALL(mock_context, EmitEncrypted(_, _)).Times(0);
    const int kNumWrites = 3;
    for (int write_no = 1; write_no <= kNumWrites; ++write_no) {
      fcp::confidentialcompute::WriteRequest write_request;
      write_request.mutable_first_request_metadata()
          ->mutable_unencrypted()
          ->set_blob_id(absl::StrCat("blob_", batch_no, "_", write_no));
      *write_request.mutable_first_request_configuration() = Any();
      std::string unencrypted_data =
          absl::StrCat("test_data_", batch_no, "_", write_no);
      absl::StatusOr<WriteFinishedResponse> write_result =
          fn.value()->Write(write_request, unencrypted_data, mock_context);
      EXPECT_THAT(write_result.status(), IsOk());
    }
    Mock::VerifyAndClearExpectations(mock_provider.get());
    Mock::VerifyAndClearExpectations(&mock_context);

    // After the commit, the provider should be called once, and the results
    // should be pushed by EmitEncrypted.
    EXPECT_CALL(*mock_provider, DoBatchedInference(_))
        .Times(1)
        .WillOnce(Invoke([](std::vector<std::string> prompts) {
          std::vector<absl::StatusOr<std::string>> results;
          for (const auto& prompt : prompts) {
            results.push_back("Processed: " + prompt);
          }
          return results;
        }));
    for (int write_no = 1; write_no <= kNumWrites; ++write_no) {
      EXPECT_CALL(
          mock_context,
          EmitEncrypted(
              0,
              AllOf(Field(&Session::KV::data,
                          Eq(absl::StrCat("Processed: test_data_", batch_no,
                                          "_", write_no))),
                    Field(&Session::KV::blob_id,
                          Eq(absl::StrCat("blob_", batch_no, "_", write_no))))))
          .Times(1)
          .WillOnce(Return(true));
      ;
    }
    fcp::confidentialcompute::CommitRequest commit_request;
    absl::StatusOr<CommitResponse> commit_result =
        fn.value()->Commit(commit_request, mock_context);
    EXPECT_THAT(commit_result.status(), IsOk());

    Mock::VerifyAndClearExpectations(mock_provider.get());
    Mock::VerifyAndClearExpectations(&mock_context);
  }
}

}  // namespace
}  // namespace confidential_federated_compute::batched_inference
