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

TEST(PrivateStateTest, CreatePrivateState_NoInitialState) {
  absl::StatusOr<std::unique_ptr<PrivateState>> result =
      PrivateState::CreatePrivateState(std::nullopt);
  ASSERT_TRUE(result.ok());
  std::unique_ptr<PrivateState> state = std::move(result.value());

  // For the first release operation, the initial state should be empty and the
  // update state should show a decremented run count.
  EXPECT_FALSE(state->GetReleaseInitialState().has_value());
  BudgetState update_state;
  ASSERT_TRUE(update_state.ParseFromString(state->GetReleaseUpdateState()));
  EXPECT_EQ(update_state.counter(), 1);
}

TEST(PrivateStateTest, CreatePrivateState_ExistingInitialState) {
  BudgetState initial_budget_state;
  initial_budget_state.set_counter(100);

  absl::StatusOr<std::unique_ptr<PrivateState>> result =
      PrivateState::CreatePrivateState(
          initial_budget_state.SerializeAsString());

  // Expect failure due to exhausted budget.
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(result.status().message(), "No budget remaining.");
}

TEST(PrivateStateTest, CreatePrivateState_MultipleReleaseOperations) {
  absl::StatusOr<std::unique_ptr<PrivateState>> result =
      PrivateState::CreatePrivateState(std::nullopt);
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
  EXPECT_EQ(update_state.counter(), 2);
}

}  // namespace
}  // namespace confidential_federated_compute::program_executor_tee