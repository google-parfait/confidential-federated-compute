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

#include "containers/fed_sql/inference_model_helper.h"

#include <string>

#include "absl/status/status_matchers.h"
#include "containers/sql/input.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"

namespace confidential_federated_compute::fed_sql {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::confidential_federated_compute::sql::Input;
using ::confidential_federated_compute::sql::RowView;
using ::fcp::confidentialcompute::Prompt;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::MutableVectorData;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;

class InferenceModelInternalTest : public ::testing::Test {
 protected:
  // Helper to build a string tensor for our test data.
  absl::StatusOr<Tensor> CreateStringTensor(
      std::initializer_list<absl::string_view> values,
      const std::string& name) {
    return Tensor::Create(
        DataType::DT_STRING, TensorShape({static_cast<int64_t>(values.size())}),
        std::make_unique<MutableVectorData<absl::string_view>>(values), name);
  }
};

TEST_F(InferenceModelInternalTest, ApplyRegexMatchExact) {
  Prompt prompt;
  // This regex looks for "Result:" followed by space, then extracts any
  // characters within <angled brackets>.
  prompt.set_regex("Result: <([^>]+)>");

  InferenceOutputProcessor output_processor;
  std::string text = "Result: <success>";
  EXPECT_THAT(output_processor.ApplyRegex(prompt, text),
              IsOkAndHolds("success"));
}

TEST_F(InferenceModelInternalTest, ApplyRegexNoMatchFullString) {
  Prompt prompt;
  // This regex looks for "Result:" followed by space, then extracts any
  // characters within <angled brackets>.
  prompt.set_regex("Result: <([^>]+)>");

  InferenceOutputProcessor output_processor;
  std::string text = "Some prefix Result: <success> some suffix";
  // Note that the entire text should match this regex in order for it to work.
  // Since the text in this case has "Some prefix" and "some suffix",  the regex
  // is not applied.
  EXPECT_THAT(output_processor.ApplyRegex(prompt, text), IsOkAndHolds(text));
}

TEST_F(InferenceModelInternalTest, PopulatePromptTemplateSuccess) {
  std::string prompt_template = "User: {user}, Message: {msg}, Session: {id}";

  absl::StatusOr<Tensor> user_tensor = CreateStringTensor({"John"}, "user");
  ASSERT_THAT(user_tensor, IsOk());
  absl::StatusOr<Tensor> msg_tensor =
      CreateStringTensor({"Are you okay?"}, "msg");
  ASSERT_THAT(msg_tensor, IsOk());
  absl::StatusOr<Tensor> id_tensor = CreateStringTensor({"42"}, "id");
  ASSERT_THAT(id_tensor, IsOk());

  std::vector<Tensor> columns;
  columns.push_back(std::move(*user_tensor));
  columns.push_back(std::move(*msg_tensor));
  columns.push_back(std::move(*id_tensor));

  fcp::confidentialcompute::BlobHeader blob_header;
  absl::StatusOr<Input> input =
      Input::CreateFromTensors(std::move(columns), blob_header);
  ASSERT_THAT(input, IsOk());
  absl::StatusOr<RowView> row = input->GetRow(0);
  ASSERT_THAT(row, IsOk());

  const std::string column_names[] = {"user", "msg", "id"};
  const size_t indices[] = {0, 1, 2};

  InferencePromptProcessor prompt_processor;

  EXPECT_THAT(prompt_processor.PopulatePromptTemplate(prompt_template, *row,
                                                      column_names, indices),
              IsOkAndHolds("User: John, Message: Are you okay?, Session: 42"));
}
}  // namespace
}  // namespace confidential_federated_compute::fed_sql