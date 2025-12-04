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

#include "program_executor_tee/private_state.h"

#include <memory>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "program_executor_tee/budget.pb.h"

namespace confidential_federated_compute::program_executor_tee {
namespace {

constexpr uint32_t kDefaultMaxRuns = 5;

TEST(PrivateStateTest, CreatePrivateState_NoInitialState) {
  absl::StatusOr<std::unique_ptr<PrivateState>> result =
      PrivateState::CreatePrivateState(std::nullopt, kDefaultMaxRuns);
  ASSERT_TRUE(result.ok());
  std::unique_ptr<PrivateState> state = std::move(result.value());

  // For the first release operation, the initial state should be empty and the
  // update state should show a decremented run count.
  EXPECT_FALSE(state->GetReleaseInitialState().has_value());
  BudgetState update_state;
  ASSERT_TRUE(update_state.ParseFromString(state->GetReleaseUpdateState()));
  EXPECT_EQ(update_state.num_runs_remaining(), kDefaultMaxRuns - 1);
}

TEST(PrivateStateTest, CreatePrivateState_InvalidInitialState) {
  absl::StatusOr<std::unique_ptr<PrivateState>> result =
      PrivateState::CreatePrivateState("invalid state", kDefaultMaxRuns);

  // Expect failure due to failed parsing.
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInternal);
  EXPECT_EQ(result.status().message(), "Failed to parse initial budget state.");
}

TEST(PrivateStateTest, CreatePrivateState_ExhaustedInitialState) {
  BudgetState initial_budget_state;
  initial_budget_state.set_num_runs_remaining(0);
  initial_budget_state.set_counter(100);

  absl::StatusOr<std::unique_ptr<PrivateState>> result =
      PrivateState::CreatePrivateState(initial_budget_state.SerializeAsString(),
                                       kDefaultMaxRuns);

  // Expect failure due to exhausted budget.
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(result.status().message(), "No budget remaining.");
}

TEST(PrivateStateTest, CreatePrivateState_ValidInitialState) {
  BudgetState initial_budget_state;
  initial_budget_state.set_num_runs_remaining(3);
  initial_budget_state.set_counter(100);

  absl::StatusOr<std::unique_ptr<PrivateState>> result =
      PrivateState::CreatePrivateState(initial_budget_state.SerializeAsString(),
                                       kDefaultMaxRuns);
  ASSERT_TRUE(result.ok());
  std::unique_ptr<PrivateState> state = std::move(result.value());

  // For the first release operation, the initial state should match and the
  // update state should show a decremented run count.
  EXPECT_EQ(state->GetReleaseInitialState().value(),
            initial_budget_state.SerializeAsString());
  BudgetState update_state;
  ASSERT_TRUE(update_state.ParseFromString(state->GetReleaseUpdateState()));
  EXPECT_EQ(update_state.num_runs_remaining(), 2);
  EXPECT_EQ(update_state.counter(), 101);
}

TEST(PrivateStateTest, CreatePrivateState_MultipleReleaseOperations) {
  BudgetState initial_budget_state;
  initial_budget_state.set_num_runs_remaining(3);
  initial_budget_state.set_counter(100);

  absl::StatusOr<std::unique_ptr<PrivateState>> result =
      PrivateState::CreatePrivateState(initial_budget_state.SerializeAsString(),
                                       kDefaultMaxRuns);
  ASSERT_TRUE(result.ok());
  std::unique_ptr<PrivateState> state = std::move(result.value());

  // Simulate the private state being updated after successful releases from
  // this program.
  std::string release_update = state->GetReleaseUpdateState();
  state->SetReleaseInitialState(release_update);

  // For the next release operation, the initial state should match the latest
  // update, the number of runs should have only been decremented by 1 compared
  // to the initial state at construction time, and the counter should have been
  // incremented twice.
  EXPECT_EQ(state->GetReleaseInitialState().value(), release_update);
  BudgetState update_state;
  ASSERT_TRUE(update_state.ParseFromString(state->GetReleaseUpdateState()));
  EXPECT_EQ(update_state.num_runs_remaining(), 2);
  EXPECT_EQ(update_state.counter(), 102);
}

}  // namespace
}  // namespace confidential_federated_compute::program_executor_tee