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

#include "local_batched_inference_engine.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "containers/common/inference/batched_inference_fn.h"
#include "containers/common/inference/batched_inference_test_utils.h"
#include "containers/common/local_inference/inference_model.h"
#include "containers/common/local_inference/testing/mocks.h"
#include "containers/fns/fn_factory.h"
#include "containers/session.h"
#include "containers/testing/mocks.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/private_inference.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/any.pb.h"
#include "gtest/gtest.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::local_inference {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::confidential_federated_compute::MockContext;
using ::confidential_federated_compute::Session;
using ::fcp::confidentialcompute::InferenceConfiguration;
using ::fcp::confidentialcompute::InferenceInitializeConfiguration;
using ::fcp::confidentialcompute::WriteRequest;
using ::google::protobuf::Any;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Field;
using ::testing::HasSubstr;
using ::testing::Return;
using ::testing::UnorderedElementsAre;

using ::confidential_federated_compute::local_inference::testing::
    MockInferenceModel;

TEST(LocalBatchedInferenceEngineTest, EmptyPromptsReturnsEmpty) {
  auto mock_model = std::make_shared<MockInferenceModel>();
  EXPECT_CALL(*mock_model, GenerateBatch).Times(0);
  LocalBatchedInferenceEngine engine(mock_model);
  auto results = engine.DoBatchedInference({});
  EXPECT_TRUE(results.empty());
}

TEST(LocalBatchedInferenceEngineTest, NullModelFailsEveryPrompt) {
  LocalBatchedInferenceEngine engine(nullptr);
  auto results = engine.DoBatchedInference({"prompt1", "prompt2"});
  ASSERT_EQ(results.size(), 2);
  EXPECT_THAT(results[0], StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(results[1], StatusIs(absl::StatusCode::kFailedPrecondition));
}

// The engine must hand the already-rendered prompts to the model verbatim: no
// templating, no parsing, no row expansion.
TEST(LocalBatchedInferenceEngineTest, PassesPromptsThroughVerbatim) {
  auto mock_model = std::make_shared<MockInferenceModel>();
  EXPECT_CALL(*mock_model,
              GenerateBatch(ElementsAre("rendered 1", "rendered 2")))
      .WillOnce(Return(
          std::vector<absl::StatusOr<std::string>>{"output 1", "output 2"}));

  LocalBatchedInferenceEngine engine(mock_model);
  auto results = engine.DoBatchedInference({"rendered 1", "rendered 2"});
  ASSERT_EQ(results.size(), 2);
  ASSERT_THAT(results[0], IsOk());
  ASSERT_THAT(results[1], IsOk());
  EXPECT_EQ(results[0].value(), "output 1");
  EXPECT_EQ(results[1].value(), "output 2");
}

TEST(LocalBatchedInferenceEngineTest, PropagatesPerPromptFailures) {
  auto mock_model = std::make_shared<MockInferenceModel>();
  EXPECT_CALL(*mock_model, GenerateBatch)
      .WillOnce(Return(std::vector<absl::StatusOr<std::string>>{
          absl::InternalError("Model execution failed."), "output 2"}));

  LocalBatchedInferenceEngine engine(mock_model);
  auto results = engine.DoBatchedInference({"prompt 1", "prompt 2"});
  ASSERT_EQ(results.size(), 2);
  EXPECT_THAT(results[0], StatusIs(absl::StatusCode::kInternal));
  ASSERT_THAT(results[1], IsOk());
  EXPECT_EQ(results[1].value(), "output 2");
}

TEST(LocalBatchedInferenceEngineProviderTest, ModelPerSessionIsRejected) {
  LocalBatchedInferenceEngineProvider provider;

  InferenceInitializeConfiguration init_config = PARSE_TEXT_PROTO(R"pb(
    inference_config {
      runtime_config { model_mode: MODEL_MODE_MODEL_PER_SESSION }
    }
    gemma_init_config {}
  )pb");
  EXPECT_THAT(provider.GetEngineForInferenceConfig(init_config, {}),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(LocalBatchedInferenceEngineProviderTest,
     MissingModelInitConfigIsRejected) {
  LocalBatchedInferenceEngineProvider provider;

  InferenceInitializeConfiguration init_config;
  EXPECT_THAT(provider.GetEngineForInferenceConfig(init_config, {}),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

// The model weight blobs are streamed during StreamInitialize and looked up by
// configuration id; a missing blob must be reported rather than silently
// producing a model-less engine.
TEST(LocalBatchedInferenceEngineProviderTest,
     MissingGemmaConfigurationBlobIsRejected) {
  LocalBatchedInferenceEngineProvider provider;

  InferenceInitializeConfiguration init_config = PARSE_TEXT_PROTO(R"pb(
    gemma_init_config {
      tokenizer_configuration_id: "tokenizer_id"
      model_weight_configuration_id: "weight_id"
    }
  )pb");
  EXPECT_THAT(provider.GetEngineForInferenceConfig(init_config, {}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(LocalBatchedInferenceEngineProviderTest,
     MissingLlamaConfigurationBlobIsRejected) {
  LocalBatchedInferenceEngineProvider provider;

  InferenceInitializeConfiguration init_config = PARSE_TEXT_PROTO(R"pb(
    llama_cpp_init_config { model_weight_configuration_id: "weight_id" }
  )pb");
  EXPECT_THAT(provider.GetEngineForInferenceConfig(init_config, {}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

// The tests below run the real engine underneath the real BatchedInferenceFn,
// mocking only the model itself. Templating, parsing and row expansion all
// belong to the Fn, so the model must see each rendered prompt exactly once and
// its raw output must be parsed exactly once. A mocked engine cannot catch a
// regression where the engine re-applies either step, which is why these cases
// exist.
class LocalBatchedInferenceIntegrationTest : public ::testing::Test {
 protected:
  using Columns = std::vector<std::pair<std::string, std::vector<std::string>>>;

  // Builds the Fn on top of a real engine wrapping `mock_model_`, then writes a
  // single blob holding `columns`.
  void SetUpFnWithColumns(const InferenceConfiguration& config,
                          const Columns& columns) {
    engine_ = std::make_shared<LocalBatchedInferenceEngine>(mock_model_);
    auto factory = inference::CreateBatchedInferenceFnFactory(engine_, config);
    ASSERT_THAT(factory, IsOk());
    auto fn = (*factory)->CreateFn();
    ASSERT_THAT(fn, IsOk());
    fn_ = *std::move(fn);

    WriteRequest write_request;
    write_request.mutable_first_request_metadata()
        ->mutable_unencrypted()
        ->set_blob_id("blob1");
    *write_request.mutable_first_request_configuration() = Any();
    // Despite its name this helper just builds a checkpoint out of named string
    // columns, which serves equally well for an input blob.
    ASSERT_THAT(
        fn_->Write(
            write_request,
            inference::testing::GetCustomInferenceOutputCheckpointForTest(
                columns),
            mock_context_),
        IsOk());
  }

  void SetUpFnWithRows(const InferenceConfiguration& config,
                       std::vector<std::string> rows) {
    SetUpFnWithColumns(config, {{"transcript", std::move(rows)}});
  }

  void ExpectEmitted(const std::string& expected_checkpoint) {
    EXPECT_CALL(mock_context_, EmitEncrypted(0, Field(&Session::KV::data,
                                                      Eq(expected_checkpoint))))
        .WillOnce(Return(true));
  }

  absl::Status Commit() {
    fcp::confidentialcompute::CommitRequest commit_request;
    return fn_->Commit(commit_request, mock_context_).status();
  }

  std::shared_ptr<MockInferenceModel> mock_model_ =
      std::make_shared<MockInferenceModel>();
  std::shared_ptr<LocalBatchedInferenceEngine> engine_;
  std::unique_ptr<fns::Fn> fn_;
  MockContext mock_context_;
};

// Two tasks over two input columns: the model must see the four fully rendered
// prompts and nothing else.
TEST_F(LocalBatchedInferenceIntegrationTest, MultiTaskMultiColumnPrompts) {
  InferenceConfiguration config = PARSE_TEXT_PROTO(R"pb(
    inference_task {
      column_config {
        input_column_names: "transcript"
        input_column_names: "topic"
        output_column_name: "summary"
      }
      prompt {
        prompt_template: "summarize {transcript} about {topic}"
        parser: PARSER_NONE
      }
    }
    inference_task {
      column_config {
        input_column_names: "transcript"
        output_column_name: "sentiment"
      }
      prompt {
        prompt_template: "sentiment of {transcript}"
        parser: PARSER_NONE
      }
    }
    runtime_config { max_prompt_size: 1000 max_batch_size: 10 }
  )pb");
  SetUpFnWithColumns(
      config, {{"transcript", {"bark", "oink"}}, {"topic", {"dogs", "pigs"}}});

  EXPECT_CALL(*mock_model_,
              GenerateBatch(UnorderedElementsAre(
                  "summarize bark about dogs", "summarize oink about pigs",
                  "sentiment of bark", "sentiment of oink")))
      .WillOnce([](absl::Span<const std::string> prompts) {
        std::vector<absl::StatusOr<std::string>> results;
        for (const std::string& prompt : prompts) {
          results.push_back(absl::StrCat("out(", prompt, ")"));
        }
        return results;
      });

  ExpectEmitted(inference::testing::GetCustomInferenceOutputCheckpointForTest(
      {{"topic", {"dogs", "pigs"}},
       {"transcript", {"bark", "oink"}},
       {"summary",
        {"out(summarize bark about dogs)", "out(summarize oink about pigs)"}},
       {"sentiment", {"out(sentiment of bark)", "out(sentiment of oink)"}}}));
  EXPECT_THAT(Commit(), IsOk());
}

// The model returns one raw string per prompt; the Fn is what splits it into
// several output rows. Re-parsing inside the engine would drop or mangle these.
TEST_F(LocalBatchedInferenceIntegrationTest, DelimiterParsingExpandsRowsOnce) {
  InferenceConfiguration config = PARSE_TEXT_PROTO(R"pb(
    inference_task {
      column_config {
        input_column_names: "transcript"
        output_column_name: "topic"
      }
      prompt {
        prompt_template: "topics of {transcript}"
        parser: PARSER_DELIMITER
      }
    }
    runtime_config { max_prompt_size: 1000 max_batch_size: 10 }
  )pb");
  SetUpFnWithRows(config, {"bark"});

  EXPECT_CALL(*mock_model_, GenerateBatch(ElementsAre("topics of bark")))
      .WillOnce(
          Return(std::vector<absl::StatusOr<std::string>>{"dogs,pets,noise"}));

  ExpectEmitted(inference::testing::GetCustomInferenceOutputCheckpointForTest(
      {{"transcript", {"bark", "bark", "bark"}},
       {"topic", {"dogs", "pets", "noise"}}}));
  EXPECT_THAT(Commit(), IsOk());
}

// PARSER_AUTO appends system instructions to the prompt and expects a JSON
// reply. Both the appending and the JSON parse must happen exactly once.
TEST_F(LocalBatchedInferenceIntegrationTest, AutoParsingHappensExactlyOnce) {
  InferenceConfiguration config = PARSE_TEXT_PROTO(R"pb(
    inference_task {
      column_config {
        input_column_names: "transcript"
        output_column_name: "topic"
      }
      prompt { prompt_template: "topics of {transcript}" parser: PARSER_AUTO }
    }
    runtime_config { max_prompt_size: 1000 max_batch_size: 10 }
  )pb");
  SetUpFnWithRows(config, {"bark"});

  std::vector<std::string> seen_prompts;
  EXPECT_CALL(*mock_model_, GenerateBatch)
      .WillOnce([&seen_prompts](absl::Span<const std::string> prompts) {
        seen_prompts.assign(prompts.begin(), prompts.end());
        return std::vector<absl::StatusOr<std::string>>{
            R"({"topic": ["dogs", "pets"]})"};
      });

  ExpectEmitted(inference::testing::GetCustomInferenceOutputCheckpointForTest(
      {{"transcript", {"bark", "bark"}}, {"topic", {"dogs", "pets"}}}));
  EXPECT_THAT(Commit(), IsOk());

  // The rendered template must be present once, with the system instructions
  // PARSER_AUTO adds appended to it.
  ASSERT_EQ(seen_prompts.size(), 1);
  EXPECT_THAT(seen_prompts[0], HasSubstr("topics of bark"));
  EXPECT_GT(seen_prompts[0].size(), std::string("topics of bark").size());
}

// A non-internal model failure is skippable: the affected row is blanked out
// and the rest of the batch still goes through.
TEST_F(LocalBatchedInferenceIntegrationTest, SkippableFailureBlanksOutRow) {
  InferenceConfiguration config = PARSE_TEXT_PROTO(R"pb(
    inference_task {
      column_config {
        input_column_names: "transcript"
        output_column_name: "topic"
      }
      prompt { prompt_template: "topics of {transcript}" parser: PARSER_NONE }
    }
    runtime_config { max_prompt_size: 1000 max_batch_size: 10 }
  )pb");
  SetUpFnWithRows(config, {"bark", "oink"});

  EXPECT_CALL(*mock_model_, GenerateBatch)
      .WillOnce(Return(std::vector<absl::StatusOr<std::string>>{
          absl::UnavailableError("model busy"), "quacking"}));

  ExpectEmitted(inference::testing::GetCustomInferenceOutputCheckpointForTest(
      {{"transcript", {"bark", "oink"}}, {"topic", {"", "quacking"}}}));
  EXPECT_THAT(Commit(), IsOk());
}

// An internal error is treated as fatal and fails the commit.
TEST_F(LocalBatchedInferenceIntegrationTest, InternalFailureAbortsCommit) {
  InferenceConfiguration config = PARSE_TEXT_PROTO(R"pb(
    inference_task {
      column_config {
        input_column_names: "transcript"
        output_column_name: "topic"
      }
      prompt { prompt_template: "topics of {transcript}" parser: PARSER_NONE }
    }
    runtime_config { max_prompt_size: 1000 max_batch_size: 10 }
  )pb");
  SetUpFnWithRows(config, {"bark"});

  EXPECT_CALL(*mock_model_, GenerateBatch)
      .WillOnce(Return(std::vector<absl::StatusOr<std::string>>{
          absl::InternalError("model blew up")}));

  EXPECT_CALL(mock_context_, EmitEncrypted).Times(0);
  EXPECT_THAT(Commit(), StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace confidential_federated_compute::local_inference
