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

#include "containers/common/inference/batched_inference_fn.h"

#include <gtest/gtest.h>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "containers/common/inference/batched_inference_engine.h"
#include "containers/common/inference/batched_inference_test_utils.h"
#include "containers/common/io/tabular/input.h"
#include "containers/session.h"
#include "containers/testing/mocks.h"
#include "fcp/confidentialcompute/constants.h"
#include "gmock/gmock.h"
#include "google/protobuf/any.pb.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.pb.h"
#include "google/protobuf/message.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::inference {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::confidential_federated_compute::Session;
using ::fcp::confidential_compute::kPrivacyIdColumnName;
using ::fcp::confidentialcompute::CommitResponse;
using ::fcp::confidentialcompute::InferenceConfiguration;
using ::fcp::confidentialcompute::StreamInitializeRequest;
using ::fcp::confidentialcompute::WriteFinishedResponse;
using ::fcp::confidentialcompute::WriteRequest;
using ::google::protobuf::Any;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::Tensor;
using ::testing::_;
using ::testing::Eq;
using ::testing::Field;
using ::testing::HasSubstr;
using ::testing::Invoke;
using ::testing::Mock;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::Test;

class MockBatchedInferenceEngine : public BatchedInferenceEngine {
 public:
  MOCK_METHOD((std::vector<absl::StatusOr<std::string>>), DoBatchedInference,
              (std::vector<std::string> prompts), (override));
};

// Returns a MessageFactory for a `test.TestLogEntry` message with a single
// string field named `transcript`.
std::shared_ptr<MessageFactory> CreateTestLogEntryMessageFactory() {
  google::protobuf::FileDescriptorSet descriptor_set;
  auto* file_proto = descriptor_set.add_file();
  file_proto->set_name("test_log.proto");
  file_proto->set_package("test");
  file_proto->set_syntax("proto3");
  auto* msg_proto = file_proto->add_message_type();
  msg_proto->set_name("TestLogEntry");
  auto* field_proto = msg_proto->add_field();
  field_proto->set_name("transcript");
  field_proto->set_number(1);
  field_proto->set_label(
      google::protobuf::FieldDescriptorProto::LABEL_OPTIONAL);
  field_proto->set_type(google::protobuf::FieldDescriptorProto::TYPE_STRING);

  auto message_factory = FileDescriptorSetMessageFactory::Create(
      descriptor_set, "test.TestLogEntry");
  CHECK_OK(message_factory.status());
  return std::move(*message_factory);
}

// Creates a BatchedInferenceFn with the default test inference config, writes
// `data` to it as a single blob, and returns the status of the write.
absl::Status WriteSingleBlob(
    std::string data, std::shared_ptr<MessageFactory> message_factory = nullptr,
    std::string on_device_query_name = "") {
  auto factory = CreateBatchedInferenceFnFactory(
      std::make_shared<NiceMock<MockBatchedInferenceEngine>>(),
      testing::GetInferenceConfigForTest(), std::move(message_factory),
      std::move(on_device_query_name));
  CHECK_OK(factory.status());
  auto fn = factory.value()->CreateFn();
  CHECK_OK(fn.status());
  MockContext mock_context;
  WriteRequest write_request;
  write_request.mutable_first_request_metadata()
      ->mutable_unencrypted()
      ->set_blob_id("blob1");
  *write_request.mutable_first_request_configuration() = Any();
  return fn.value()
      ->Write(write_request, std::move(data), mock_context)
      .status();
}

class BatchedInferenceFnTest : public Test {
 protected:
  void SetUp() override {}

  // First vector goes over commits, second over blobs, third over rows within a
  // blob.
  void RunTestCaseFor(
      std::vector<std::vector<std::vector<std::string>>> commits) {
    std::shared_ptr<NiceMock<MockBatchedInferenceEngine>> mock_engine =
        std::make_shared<NiceMock<MockBatchedInferenceEngine>>();
    InferenceConfiguration inference_config =
        testing::GetInferenceConfigForTest();
    absl::StatusOr<std::unique_ptr<fns::FnFactory>> factory =
        CreateBatchedInferenceFnFactory(mock_engine, inference_config);
    ASSERT_THAT(factory.status(), IsOk());
    auto fn = factory.value()->CreateFn();
    ASSERT_THAT(fn.status(), IsOk());
    MockContext mock_context;
    EXPECT_CALL(*mock_engine, DoBatchedInference(_)).Times(0);
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
        Mock::VerifyAndClearExpectations(mock_engine.get());
        Mock::VerifyAndClearExpectations(&mock_context);
      }

      // Now commit and verify inference calls and integrity of the results.
      EXPECT_CALL(*mock_engine, DoBatchedInference(_))
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
      Mock::VerifyAndClearExpectations(mock_engine.get());
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

TEST_F(BatchedInferenceFnTest, OneTaskWithMultiRowOutput) {
  // 1-task config, using PARSER_DELIMITER to split results into multiple rows.
  InferenceConfiguration config = PARSE_TEXT_PROTO(R"pb(
    inference_task {
      column_config {
        input_column_names: "transcript"
        output_column_name: "topic"
      }
      prompt { prompt_template: "Hello, {transcript}" parser: PARSER_DELIMITER }
    }
    runtime_config { max_prompt_size: 1000 max_batch_size: 10 }
  )pb");

  auto mock_engine = std::make_shared<NiceMock<MockBatchedInferenceEngine>>();
  auto factory = CreateBatchedInferenceFnFactory(mock_engine, config);
  ASSERT_THAT(factory.status(), IsOk());
  auto fn = factory.value()->CreateFn();
  ASSERT_THAT(fn.status(), IsOk());
  MockContext mock_context;

  // Write 2 input rows.
  std::string input =
      testing::GetPrivateInferenceInputCheckpointForTest({"foo", "bar"});
  WriteRequest write_request;
  write_request.mutable_first_request_metadata()
      ->mutable_unencrypted()
      ->set_blob_id("blob1");
  *write_request.mutable_first_request_configuration() = Any();
  ASSERT_THAT(fn.value()->Write(write_request, input, mock_context).status(),
              IsOk());

  EXPECT_CALL(*mock_engine, DoBatchedInference(_))
      .WillOnce([](std::vector<std::string> prompts) {
        std::vector<absl::StatusOr<std::string>> results;
        for (const auto& p : prompts) {
          // 2 results for row 0, 1 result for row 1.
          if (p == "Hello, foo")
            results.push_back("res_a,res_b");
          else if (p == "Hello, bar")
            results.push_back("res_c");
        }
        return results;
      });

  std::string expected = testing::GetPrivateInferenceOutputCheckpointForTest(
      {"foo", "foo", "bar"}, {"res_a", "res_b", "res_c"});
  EXPECT_CALL(mock_context,
              EmitEncrypted(0, AllOf(Field(&Session::KV::blob_id, Eq("blob1")),
                                     Field(&Session::KV::data, Eq(expected)))))
      .WillOnce(Return(true));

  fcp::confidentialcompute::CommitRequest commit_request;
  EXPECT_THAT(fn.value()->Commit(commit_request, mock_context).status(),
              IsOk());
}

TEST_F(BatchedInferenceFnTest, OneSingleRowTaskTwoMultiRowTasksSucceeds) {
  InferenceConfiguration config = PARSE_TEXT_PROTO(R"pb(
    inference_task {
      column_config {
        input_column_names: "transcript"
        output_column_name: "one_row"
      }
      prompt {
        prompt_template: "one_row {transcript}"
        parser: PARSER_DELIMITER
      }
    }
    inference_task {
      column_config {
        input_column_names: "transcript"
        output_column_name: "three_rows"
      }
      prompt {
        prompt_template: "three_rows {transcript}"
        parser: PARSER_DELIMITER
      }
    }
    inference_task {
      column_config {
        input_column_names: "transcript"
        output_column_name: "two_rows"
      }
      prompt {
        prompt_template: "two_rows {transcript}"
        parser: PARSER_DELIMITER
      }
    }
    runtime_config { max_prompt_size: 1000 max_batch_size: 10 }
  )pb");

  auto mock_engine = std::make_shared<NiceMock<MockBatchedInferenceEngine>>();
  auto factory = CreateBatchedInferenceFnFactory(mock_engine, config);
  ASSERT_THAT(factory.status(), IsOk());
  auto fn = factory.value()->CreateFn();
  ASSERT_THAT(fn.status(), IsOk());
  MockContext mock_context;

  std::string input =
      testing::GetPrivateInferenceInputCheckpointForTest({"bark"});
  WriteRequest write_request;
  write_request.mutable_first_request_metadata()
      ->mutable_unencrypted()
      ->set_blob_id("blob1");
  *write_request.mutable_first_request_configuration() = Any();
  ASSERT_THAT(fn.value()->Write(write_request, input, mock_context).status(),
              IsOk());

  EXPECT_CALL(*mock_engine, DoBatchedInference(_))
      .WillOnce([](std::vector<std::string> prompts) {
        std::vector<absl::StatusOr<std::string>> results;
        for (const auto& p : prompts) {
          if (p == "one_row bark")
            results.push_back("1");
          else if (p == "three_rows bark")
            results.push_back("a,b,c");
          else if (p == "two_rows bark")
            results.push_back("x,y");
        }
        return results;
      });

  std::string expected = testing::GetCustomInferenceOutputCheckpointForTest(
      {{"transcript", {"bark", "bark", "bark", "bark", "bark", "bark"}},
       {"one_row", {"1", "1", "1", "1", "1", "1"}},
       {"three_rows", {"a", "a", "b", "b", "c", "c"}},
       {"two_rows", {"x", "y", "x", "y", "x", "y"}}});
  EXPECT_CALL(mock_context,
              EmitEncrypted(0, Field(&Session::KV::data, Eq(expected))))
      .WillOnce(Return(true));

  fcp::confidentialcompute::CommitRequest commit_request;
  EXPECT_THAT(fn.value()->Commit(commit_request, mock_context).status(),
              IsOk());
}

TEST_F(BatchedInferenceFnTest, OneTaskProducesZeroValuesForRow) {
  InferenceConfiguration config = PARSE_TEXT_PROTO(R"pb(
    inference_task {
      column_config {
        input_column_names: "transcript"
        output_column_name: "topic"
      }
      prompt { prompt_template: "topic {transcript}" parser: PARSER_DELIMITER }
    }
    inference_task {
      column_config {
        input_column_names: "transcript"
        output_column_name: "keywords"
      }
      prompt {
        prompt_template: "keywords {transcript}"
        parser: PARSER_DELIMITER
      }
    }
    runtime_config { max_prompt_size: 1000 max_batch_size: 10 }
  )pb");

  auto mock_engine = std::make_shared<NiceMock<MockBatchedInferenceEngine>>();
  auto factory = CreateBatchedInferenceFnFactory(mock_engine, config);
  ASSERT_THAT(factory.status(), IsOk());
  auto fn = factory.value()->CreateFn();
  ASSERT_THAT(fn.status(), IsOk());
  MockContext mock_context;

  // Write 1 input row.
  std::string input =
      testing::GetPrivateInferenceInputCheckpointForTest({"bark"});
  WriteRequest write_request;
  write_request.mutable_first_request_metadata()
      ->mutable_unencrypted()
      ->set_blob_id("blob1");
  *write_request.mutable_first_request_configuration() = Any();
  ASSERT_THAT(fn.value()->Write(write_request, input, mock_context).status(),
              IsOk());

  // One task produces 2 values, the other produces 0 value.
  EXPECT_CALL(*mock_engine, DoBatchedInference(_))
      .WillOnce([](std::vector<std::string> prompts) {
        std::vector<absl::StatusOr<std::string>> results;
        for (const auto& p : prompts) {
          if (p == "topic bark")
            results.push_back("a,b");
          else if (p == "keywords bark")
            results.push_back("");  // 0 values!
        }
        return results;
      });

  std::string expected = testing::GetCustomInferenceOutputCheckpointForTest(
      {{"transcript", {"bark", "bark"}},
       {"topic", {"a", "b"}},
       {"keywords", {"", ""}}});
  EXPECT_CALL(mock_context,
              EmitEncrypted(0, Field(&Session::KV::data, Eq(expected))))
      .WillOnce(Return(true));

  fcp::confidentialcompute::CommitRequest commit_request;
  EXPECT_THAT(fn.value()->Commit(commit_request, mock_context).status(),
              IsOk());
}

TEST_F(BatchedInferenceFnTest, AllTasksProduceZeroValuesForRow) {
  // 3 inference tasks, each using a different parser.
  InferenceConfiguration config = PARSE_TEXT_PROTO(R"pb(
    inference_task {
      column_config {
        input_column_names: "transcript"
        output_column_name: "delimiter_col"
      }
      prompt {
        prompt_template: "delimiter {transcript}"
        parser: PARSER_DELIMITER
      }
    }
    inference_task {
      column_config {
        input_column_names: "transcript"
        output_column_name: "auto_col"
      }
      prompt { prompt_template: "auto {transcript}" parser: PARSER_AUTO }
    }
    inference_task {
      column_config {
        input_column_names: "transcript"
        output_column_name: "none_col"
      }
      prompt { prompt_template: "none {transcript}" }
    }
    runtime_config { max_prompt_size: 1000 max_batch_size: 10 }
  )pb");

  auto mock_engine = std::make_shared<NiceMock<MockBatchedInferenceEngine>>();
  auto factory = CreateBatchedInferenceFnFactory(mock_engine, config);
  ASSERT_THAT(factory.status(), IsOk());
  auto fn = factory.value()->CreateFn();
  ASSERT_THAT(fn.status(), IsOk());
  MockContext mock_context;

  std::string input =
      testing::GetPrivateInferenceInputCheckpointForTest({"bark"});
  WriteRequest write_request;
  write_request.mutable_first_request_metadata()
      ->mutable_unencrypted()
      ->set_blob_id("blob1");
  *write_request.mutable_first_request_configuration() = Any();
  ASSERT_THAT(fn.value()->Write(write_request, input, mock_context).status(),
              IsOk());

  // All three tasks produce 0 values:
  EXPECT_CALL(*mock_engine, DoBatchedInference(_))
      .WillOnce([](std::vector<std::string> prompts) {
        std::vector<absl::StatusOr<std::string>> results;
        for (const auto& p : prompts) {
          if (p == "delimiter bark") results.push_back("");
          // PARSER_AUTO appends system instructions, so use StartsWith.
          else if (absl::StartsWith(p, "auto bark"))
            results.push_back("{\"auto_col\": []}");
          else if (p == "none bark")
            results.push_back("");
        }
        return results;
      });

  std::string expected = testing::GetCustomInferenceOutputCheckpointForTest(
      {{"transcript", {"bark"}},
       {"delimiter_col", {""}},
       {"auto_col", {""}},
       {"none_col", {""}}});
  EXPECT_CALL(mock_context,
              EmitEncrypted(0, Field(&Session::KV::data, Eq(expected))))
      .WillOnce(Return(true));

  fcp::confidentialcompute::CommitRequest commit_request;
  EXPECT_THAT(fn.value()->Commit(commit_request, mock_context).status(),
              IsOk());
}
TEST_F(BatchedInferenceFnTest, InferenceHandlesEmptyResponse) {
  InferenceConfiguration config = PARSE_TEXT_PROTO(R"pb(
    inference_task {
      column_config {
        input_column_names: "transcript"
        output_column_name: "none_col"
      }
      prompt { prompt_template: "none {transcript}" }
    }
    runtime_config { max_prompt_size: 1000 max_batch_size: 10 }
  )pb");

  auto mock_engine = std::make_shared<NiceMock<MockBatchedInferenceEngine>>();
  auto factory = CreateBatchedInferenceFnFactory(mock_engine, config);
  ASSERT_THAT(factory.status(), IsOk());
  auto fn = factory.value()->CreateFn();
  ASSERT_THAT(fn.status(), IsOk());
  MockContext mock_context;

  std::string input =
      testing::GetPrivateInferenceInputCheckpointForTest({"bark", "meow"});
  WriteRequest write_request;
  write_request.mutable_first_request_metadata()
      ->mutable_unencrypted()
      ->set_blob_id("blob1");
  *write_request.mutable_first_request_configuration() = Any();
  ASSERT_THAT(fn.value()->Write(write_request, input, mock_context).status(),
              IsOk());

  EXPECT_CALL(*mock_engine, DoBatchedInference(_))
      .WillOnce([](std::vector<std::string> prompts) {
        std::vector<absl::StatusOr<std::string>> results;
        for (const auto& p : prompts) {
          if (p == "none bark") {
            // First row succeeds with output
            results.push_back("bark_result");
          } else if (p == "none meow") {
            // Second row returns empty string (e.g. filtered by safety filter)
            results.push_back("");
          }
        }
        return results;
      });

  std::string expected = testing::GetCustomInferenceOutputCheckpointForTest(
      {{"transcript", {"bark", "meow"}}, {"none_col", {"bark_result", ""}}});
  EXPECT_CALL(mock_context,
              EmitEncrypted(0, Field(&Session::KV::data, Eq(expected))))
      .WillOnce(Return(true));

  fcp::confidentialcompute::CommitRequest commit_request;
  EXPECT_THAT(fn.value()->Commit(commit_request, mock_context).status(),
              IsOk());
  EXPECT_EQ(
      mock_context
          .GetCounters()["BatchedInferenceContainer-empty-inference-response"],
      1);
}

TEST_F(BatchedInferenceFnTest, InferenceHandlesParsingFailureGracefully) {
  InferenceConfiguration config = PARSE_TEXT_PROTO(R"pb(
    inference_task {
      column_config {
        input_column_names: "transcript"
        output_column_name: "auto_col"
      }
      prompt { prompt_template: "auto {transcript}" parser: PARSER_AUTO }
    }
    runtime_config { max_prompt_size: 1000 max_batch_size: 10 }
  )pb");

  auto mock_engine = std::make_shared<NiceMock<MockBatchedInferenceEngine>>();
  auto factory = CreateBatchedInferenceFnFactory(mock_engine, config);
  ASSERT_THAT(factory.status(), IsOk());
  auto fn = factory.value()->CreateFn();
  ASSERT_THAT(fn.status(), IsOk());
  MockContext mock_context;

  std::string input =
      testing::GetPrivateInferenceInputCheckpointForTest({"bark", "meow"});
  WriteRequest write_request;
  write_request.mutable_first_request_metadata()
      ->mutable_unencrypted()
      ->set_blob_id("blob1");
  *write_request.mutable_first_request_configuration() = Any();
  ASSERT_THAT(fn.value()->Write(write_request, input, mock_context).status(),
              IsOk());

  EXPECT_CALL(*mock_engine, DoBatchedInference(_))
      .WillOnce([](std::vector<std::string> prompts) {
        std::vector<absl::StatusOr<std::string>> results;
        for (const auto& p : prompts) {
          if (absl::StartsWith(p, "auto bark")) {
            results.push_back("```json\n{\"auto_col\": [\"bark_res\"]}\n```");
          } else if (absl::StartsWith(p, "auto meow")) {
            // Returns empty string, which fails PARSER_AUTO JSON parsing.
            results.push_back("");
          }
        }
        return results;
      });

  std::string expected = testing::GetCustomInferenceOutputCheckpointForTest(
      {{"transcript", {"bark", "meow"}}, {"auto_col", {"bark_res", ""}}});
  EXPECT_CALL(mock_context,
              EmitEncrypted(0, Field(&Session::KV::data, Eq(expected))))
      .WillOnce(Return(true));

  fcp::confidentialcompute::CommitRequest commit_request;
  EXPECT_THAT(fn.value()->Commit(commit_request, mock_context).status(),
              IsOk());
  EXPECT_EQ(
      mock_context
          .GetCounters()["BatchedInferenceContainer-empty-inference-response"],
      1);
  EXPECT_EQ(
      mock_context.GetCounters()
          ["BatchedInferenceContainer-inference-output-processing-failed"],
      1);
}

TEST_F(BatchedInferenceFnTest,
       InferenceCountsMultipleEmptyResponsesAccurately) {
  InferenceConfiguration config = PARSE_TEXT_PROTO(R"pb(
    inference_task {
      column_config {
        input_column_names: "transcript"
        output_column_name: "none_col"
      }
      prompt { prompt_template: "none {transcript}" }
    }
    runtime_config { max_prompt_size: 1000 max_batch_size: 10 }
  )pb");

  auto mock_engine = std::make_shared<NiceMock<MockBatchedInferenceEngine>>();
  auto factory = CreateBatchedInferenceFnFactory(mock_engine, config);
  ASSERT_THAT(factory.status(), IsOk());
  auto fn = factory.value()->CreateFn();
  ASSERT_THAT(fn.status(), IsOk());
  MockContext mock_context;

  std::string input = testing::GetPrivateInferenceInputCheckpointForTest(
      {"dog", "cat", "bird", "fish"});
  WriteRequest write_request;
  write_request.mutable_first_request_metadata()
      ->mutable_unencrypted()
      ->set_blob_id("blob1");
  *write_request.mutable_first_request_configuration() = Any();
  ASSERT_THAT(fn.value()->Write(write_request, input, mock_context).status(),
              IsOk());

  EXPECT_CALL(*mock_engine, DoBatchedInference(_))
      .WillOnce([](std::vector<std::string> prompts) {
        std::vector<absl::StatusOr<std::string>> results;
        for (const auto& p : prompts) {
          if (p == "none dog") {
            results.push_back("dog_res");
          } else if (p == "none cat") {
            results.push_back("");
          } else if (p == "none bird") {
            results.push_back("");
          } else if (p == "none fish") {
            results.push_back("fish_res");
          }
        }
        return results;
      });

  std::string expected = testing::GetCustomInferenceOutputCheckpointForTest(
      {{"transcript", {"dog", "cat", "bird", "fish"}},
       {"none_col", {"dog_res", "", "", "fish_res"}}});
  EXPECT_CALL(mock_context,
              EmitEncrypted(0, Field(&Session::KV::data, Eq(expected))))
      .WillOnce(Return(true));

  fcp::confidentialcompute::CommitRequest commit_request;
  EXPECT_THAT(fn.value()->Commit(commit_request, mock_context).status(),
              IsOk());
  EXPECT_EQ(
      mock_context
          .GetCounters()["BatchedInferenceContainer-empty-inference-response"],
      2);
}

TEST_F(BatchedInferenceFnTest, ErrorHandling_SkipsInvalidArgument) {
  InferenceConfiguration config = testing::GetInferenceConfigForTest();
  auto mock_engine = std::make_shared<NiceMock<MockBatchedInferenceEngine>>();
  auto factory = CreateBatchedInferenceFnFactory(mock_engine, config);
  ASSERT_THAT(factory.status(), IsOk());
  auto fn = factory.value()->CreateFn();
  ASSERT_THAT(fn.status(), IsOk());
  MockContext mock_context;

  std::string input =
      testing::GetPrivateInferenceInputCheckpointForTest({"bark"});
  WriteRequest write_request;
  write_request.mutable_first_request_metadata()
      ->mutable_unencrypted()
      ->set_blob_id("blob1");
  *write_request.mutable_first_request_configuration() = Any();
  ASSERT_THAT(fn.value()->Write(write_request, input, mock_context).status(),
              IsOk());

  EXPECT_CALL(*mock_engine, DoBatchedInference(_))
      .WillOnce([](std::vector<std::string> prompts) {
        return std::vector<absl::StatusOr<std::string>>{
            absl::InvalidArgumentError("Something is wrong")};
      });

  std::string expected =
      testing::GetPrivateInferenceOutputCheckpointForTest({"bark"}, {""});
  EXPECT_CALL(mock_context,
              EmitEncrypted(0, Field(&Session::KV::data, Eq(expected))))
      .WillOnce(Return(true));

  fcp::confidentialcompute::CommitRequest commit_request;
  EXPECT_THAT(fn.value()->Commit(commit_request, mock_context).status(),
              IsOk());
}

TEST_F(BatchedInferenceFnTest, ErrorHandling_FailsOnInternalError) {
  InferenceConfiguration config = testing::GetInferenceConfigForTest();
  auto mock_engine = std::make_shared<NiceMock<MockBatchedInferenceEngine>>();
  auto factory = CreateBatchedInferenceFnFactory(mock_engine, config);
  ASSERT_THAT(factory.status(), IsOk());
  auto fn = factory.value()->CreateFn();
  ASSERT_THAT(fn.status(), IsOk());
  MockContext mock_context;

  std::string input =
      testing::GetPrivateInferenceInputCheckpointForTest({"bark"});
  WriteRequest write_request;
  write_request.mutable_first_request_metadata()
      ->mutable_unencrypted()
      ->set_blob_id("blob1");
  *write_request.mutable_first_request_configuration() = Any();
  ASSERT_THAT(fn.value()->Write(write_request, input, mock_context).status(),
              IsOk());

  EXPECT_CALL(*mock_engine, DoBatchedInference(_))
      .WillOnce([](std::vector<std::string> prompts) {
        return std::vector<absl::StatusOr<std::string>>{
            absl::InternalError("Something is very wrong")};
      });

  fcp::confidentialcompute::CommitRequest commit_request;
  EXPECT_THAT(fn.value()->Commit(commit_request, mock_context).status(),
              StatusIs(absl::StatusCode::kInternal));
}

TEST_F(BatchedInferenceFnTest, PrivateLoggerMessageCheckpoint) {
  std::shared_ptr<MessageFactory> message_factory =
      CreateTestLogEntryMessageFactory();

  InferenceConfiguration config = testing::GetInferenceConfigForTest();
  auto mock_engine = std::make_shared<NiceMock<MockBatchedInferenceEngine>>();
  auto factory = CreateBatchedInferenceFnFactory(mock_engine, config,
                                                 message_factory, "my_query");
  ASSERT_THAT(factory.status(), IsOk());
  auto fn = factory.value()->CreateFn();
  ASSERT_THAT(fn.status(), IsOk());
  MockContext mock_context;

  // Create two serialized TestLogEntry messages ("bark" and "oink").
  auto msg1 = message_factory->NewMessage();
  msg1->GetReflection()->SetString(
      msg1.get(), msg1->GetDescriptor()->FindFieldByName("transcript"), "bark");
  auto msg2 = message_factory->NewMessage();
  msg2->GetReflection()->SetString(
      msg2.get(), msg2->GetDescriptor()->FindFieldByName("transcript"), "oink");

  tensorflow_federated::aggregation::FederatedComputeCheckpointBuilderFactory
      builder_factory;
  auto builder = builder_factory.Create();
  const std::string entry_col_name = absl::StrCat(
      "my_query/", fcp::confidential_compute::kPrivateLoggerEntryKey);
  const std::string time_col_name = absl::StrCat(
      "my_query/", fcp::confidential_compute::kEventTimeColumnName);
  tensorflow_federated::aggregation::Tensor entries_tensor(
      std::vector<std::string>{msg1->SerializeAsString(),
                               msg2->SerializeAsString()},
      entry_col_name);
  ASSERT_THAT(builder->Add(entry_col_name, std::move(entries_tensor)), IsOk());
  tensorflow_federated::aggregation::Tensor times_tensor(
      std::vector<std::string>{"2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z"},
      time_col_name);
  ASSERT_THAT(builder->Add(time_col_name, std::move(times_tensor)), IsOk());
  auto input_ckpt = builder->Build();
  ASSERT_THAT(input_ckpt.status(), IsOk());

  WriteRequest write_request;
  write_request.mutable_first_request_metadata()
      ->mutable_unencrypted()
      ->set_blob_id("pl_blob_1");
  *write_request.mutable_first_request_configuration() = Any();

  ASSERT_THAT(fn.value()
                  ->Write(write_request, std::string(input_ckpt->Flatten()),
                          mock_context)
                  .status(),
              IsOk());

  EXPECT_CALL(*mock_engine, DoBatchedInference(_))
      .WillOnce([](std::vector<std::string> prompts) {
        std::vector<absl::StatusOr<std::string>> results;
        for (const auto& p : prompts) {
          results.push_back("Processed: " + p);
        }
        return results;
      });

  std::string captured_output;
  EXPECT_CALL(mock_context, EmitEncrypted(0, _))
      .WillOnce(Invoke([&](int, Session::KV kv) {
        EXPECT_EQ(kv.blob_id, "pl_blob_1");
        captured_output = std::move(kv.data);
        return true;
      }));

  fcp::confidentialcompute::CommitRequest commit_request;
  EXPECT_THAT(fn.value()->Commit(commit_request, mock_context).status(),
              IsOk());

  // Verify the emitted checkpoint contains flattened proto columns,
  // confidential_compute_event_time, and inference output column ("topic").
  tensorflow_federated::aggregation::FederatedComputeCheckpointParserFactory
      parser_factory;
  auto parser = parser_factory.Create(absl::Cord(captured_output));
  ASSERT_THAT(parser.status(), IsOk());
  auto out_tensors = (*parser)->LoadAllTensors();
  ASSERT_THAT(out_tensors.status(), IsOk());

  ASSERT_TRUE(out_tensors->contains("transcript"));
  EXPECT_THAT(out_tensors->at("transcript").AsSpan<absl::string_view>(),
              ::testing::ElementsAre("bark", "oink"));
  ASSERT_TRUE(
      out_tensors->contains(fcp::confidential_compute::kEventTimeColumnName));
  EXPECT_THAT(
      out_tensors->at(fcp::confidential_compute::kEventTimeColumnName)
          .AsSpan<absl::string_view>(),
      ::testing::ElementsAre("2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z"));
  ASSERT_TRUE(out_tensors->contains("topic"));
  EXPECT_THAT(out_tensors->at("topic").AsSpan<absl::string_view>(),
              ::testing::ElementsAre("Processed: Hello, bark",
                                     "Processed: Hello, oink"));
}

TEST_F(BatchedInferenceFnTest, TensorCheckpointWithPrivacyId) {
  InferenceConfiguration config = testing::GetInferenceConfigForTest();
  auto mock_engine = std::make_shared<NiceMock<MockBatchedInferenceEngine>>();
  auto factory = CreateBatchedInferenceFnFactory(mock_engine, config);
  ASSERT_THAT(factory.status(), IsOk());
  auto fn = factory.value()->CreateFn();
  ASSERT_THAT(fn.status(), IsOk());
  MockContext mock_context;

  // Unlike the columns, the privacy ID is a scalar tensor.
  auto builder = FederatedComputeCheckpointBuilderFactory().Create();
  ASSERT_THAT(builder->Add("transcript",
                           Tensor(std::vector<std::string>{"bark", "oink"},
                                  "transcript")),
              IsOk());
  ASSERT_THAT(builder->Add(kPrivacyIdColumnName,
                           Tensor("the_privacy_id", kPrivacyIdColumnName)),
              IsOk());
  absl::StatusOr<absl::Cord> input = builder->Build();
  ASSERT_THAT(input.status(), IsOk());

  WriteRequest write_request;
  write_request.mutable_first_request_metadata()
      ->mutable_unencrypted()
      ->set_blob_id("blob1");
  *write_request.mutable_first_request_configuration() = Any();
  ASSERT_THAT(
      fn.value()
          ->Write(write_request, std::string(input->Flatten()), mock_context)
          .status(),
      IsOk());

  EXPECT_CALL(*mock_engine, DoBatchedInference(_))
      .WillOnce([](std::vector<std::string> prompts) {
        std::vector<absl::StatusOr<std::string>> results;
        for (const auto& p : prompts) {
          results.push_back("Processed: " + p);
        }
        return results;
      });

  // The privacy ID isn't an input column, so it isn't included in the output.
  std::string expected = testing::GetPrivateInferenceOutputCheckpointForTest(
      {"bark", "oink"}, {"Processed: Hello, bark", "Processed: Hello, oink"});
  EXPECT_CALL(mock_context,
              EmitEncrypted(0, Field(&Session::KV::data, Eq(expected))))
      .WillOnce(Return(true));

  fcp::confidentialcompute::CommitRequest commit_request;
  EXPECT_THAT(fn.value()->Commit(commit_request, mock_context).status(),
              IsOk());
}

TEST_F(BatchedInferenceFnTest, TensorCheckpointWithNonScalarPrivacyIdFails) {
  auto builder = FederatedComputeCheckpointBuilderFactory().Create();
  ASSERT_THAT(builder->Add("transcript",
                           Tensor(std::vector<std::string>{"bark", "oink"},
                                  "transcript")),
              IsOk());
  ASSERT_THAT(builder->Add(kPrivacyIdColumnName,
                           Tensor(std::vector<std::string>{"id1", "id2"},
                                  kPrivacyIdColumnName)),
              IsOk());
  absl::StatusOr<absl::Cord> input = builder->Build();
  ASSERT_THAT(input.status(), IsOk());

  EXPECT_THAT(WriteSingleBlob(std::string(input->Flatten())),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be a scalar")));
}

TEST_F(BatchedInferenceFnTest, TensorCheckpointWithDifferingRowCountsFails) {
  auto builder = FederatedComputeCheckpointBuilderFactory().Create();
  ASSERT_THAT(builder->Add("transcript",
                           Tensor(std::vector<std::string>{"bark", "oink"},
                                  "transcript")),
              IsOk());
  ASSERT_THAT(
      builder->Add("other", Tensor(std::vector<std::string>{"meow"}, "other")),
      IsOk());
  absl::StatusOr<absl::Cord> input = builder->Build();
  ASSERT_THAT(input.status(), IsOk());

  EXPECT_THAT(WriteSingleBlob(std::string(input->Flatten())),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("same number of rows")));
}

TEST_F(BatchedInferenceFnTest, InvalidCheckpointFails) {
  EXPECT_THAT(WriteSingleBlob("invalid checkpoint"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Failed to construct a checkpoint parser")));
}

TEST_F(BatchedInferenceFnTest, MessageCheckpointWithoutEntriesFails) {
  // A tensor-based checkpoint doesn't have the `my_query/entry` tensor.
  EXPECT_THAT(
      WriteSingleBlob(
          testing::GetPrivateInferenceInputCheckpointForTest({"bark"}),
          CreateTestLogEntryMessageFactory(), "my_query"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Failed to create input from message checkpoint")));
}

}  // namespace
}  // namespace confidential_federated_compute::inference
