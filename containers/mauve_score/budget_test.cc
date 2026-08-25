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

#include "budget.h"

#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mauve_budget_state.pb.h"

namespace confidential_federated_compute::mauve_score {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

TEST(BudgetTest, FirstRunWithEmptyState) {
  auto budget = Budget::Create(/*initial_state=*/"", /*access_budget_times=*/5);
  ASSERT_THAT(budget, IsOk());

  EXPECT_EQ(budget->GetInitialState(), "");
  budget->DecrementBudget();
  // dst_state should have remaining_budget = 4 (5 - 1).
  MauveBudgetState dst;
  ASSERT_TRUE(dst.ParseFromString(budget->SerializeAsString()));
  EXPECT_EQ(dst.remaining_budget(), 4);
}

TEST(BudgetTest, SubsequentRunParsesState) {
  // Create a prior state with remaining_budget = 3.
  MauveBudgetState prior_state;
  prior_state.set_remaining_budget(3);
  std::string serialized = prior_state.SerializeAsString();

  auto budget = Budget::Create(/*initial_state=*/serialized,
                               /*access_budget_times=*/5);
  ASSERT_THAT(budget, IsOk());

  EXPECT_EQ(budget->GetInitialState(), serialized);

  budget->DecrementBudget();

  // dst_state should have remaining_budget = 2 (3 - 1).
  MauveBudgetState dst;
  ASSERT_TRUE(dst.ParseFromString(budget->SerializeAsString()));
  EXPECT_EQ(dst.remaining_budget(), 2);
}

TEST(BudgetTest, BudgetExhaustedOnFirstRun) {
  EXPECT_THAT(Budget::Create(/*initial_state=*/"", /*access_budget_times=*/0),
              StatusIs(absl::StatusCode::kResourceExhausted));
}

TEST(BudgetTest, BudgetExhaustedOnSubsequentRun) {
  MauveBudgetState prior_state;
  prior_state.set_remaining_budget(0);
  std::string serialized = prior_state.SerializeAsString();

  EXPECT_THAT(Budget::Create(/*initial_state=*/serialized,
                             /*access_budget_times=*/5),
              StatusIs(absl::StatusCode::kResourceExhausted));
}

TEST(BudgetTest, SingleBudgetDecrementsToZero) {
  auto budget = Budget::Create(/*initial_state=*/"", /*access_budget_times=*/1);
  ASSERT_THAT(budget, IsOk());

  budget->DecrementBudget();

  MauveBudgetState dst;
  ASSERT_TRUE(dst.ParseFromString(budget->SerializeAsString()));
  EXPECT_EQ(dst.remaining_budget(), 0);
}

TEST(BudgetTest, MultipleDecrementBudget) {
  MauveBudgetState prior_state;
  prior_state.set_remaining_budget(4);
  std::string serialized = prior_state.SerializeAsString();

  auto budget = Budget::Create(/*initial_state=*/serialized,
                               /*access_budget_times=*/5);
  ASSERT_THAT(budget, IsOk());

  budget->DecrementBudget();
  budget->DecrementBudget();

  MauveBudgetState dst;
  ASSERT_TRUE(dst.ParseFromString(budget->SerializeAsString()));
  EXPECT_EQ(dst.remaining_budget(), 2);
}

}  // namespace
}  // namespace confidential_federated_compute::mauve_score
