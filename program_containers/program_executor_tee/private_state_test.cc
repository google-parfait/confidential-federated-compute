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
  // update state should show a counter incremented by 1.
  EXPECT_FALSE(state->GetState().has_value());
  BudgetState update_state;
  ASSERT_TRUE(update_state.ParseFromString(state->CommitNewState()));
  EXPECT_EQ(update_state.counter(), 1);
}

TEST(PrivateStateTest, CreatePrivateState_ExistingInitialState) {
  BudgetState initial_budget_state;
  initial_budget_state.set_counter(100);
  initial_budget_state.set_recovery_blob_id("some_blob_id");

  absl::StatusOr<std::unique_ptr<PrivateState>> result =
      PrivateState::CreatePrivateState(
          initial_budget_state.SerializeAsString());

  // Should succeed by parsing the initial state as a BudgetState.
  ASSERT_TRUE(result.ok());
  std::unique_ptr<PrivateState> state = std::move(result.value());

  // The committed state should match the serialized initial state.
  ASSERT_TRUE(state->GetState().has_value());
  BudgetState parsed_state;
  ASSERT_TRUE(parsed_state.ParseFromString(state->GetState().value()));
  EXPECT_EQ(parsed_state.counter(), 100);

  // CommitNewState should increment the counter.
  BudgetState update_state;
  ASSERT_TRUE(update_state.ParseFromString(state->CommitNewState()));
  EXPECT_EQ(update_state.counter(), 101);
}

TEST(PrivateStateTest, CreatePrivateState_ExhaustedBudgetNoRecovery) {
  BudgetState initial_budget_state;
  initial_budget_state.set_counter(100);

  absl::StatusOr<std::unique_ptr<PrivateState>> result =
      PrivateState::CreatePrivateState(
          initial_budget_state.SerializeAsString());

  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(
      result.status().message(),
      "Programs may recover, but cannot run from scratch multiple times.");
}

TEST(PrivateStateTest, CreatePrivateState_InvalidInitialState) {
  absl::StatusOr<std::unique_ptr<PrivateState>> result =
      PrivateState::CreatePrivateState("not a valid proto");

  // Should fail because the initial state cannot be parsed as a BudgetState.
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(result.status().message(),
            "Failed to parse initial state into BudgetState.");
}

TEST(PrivateStateTest, CreatePrivateState_MultipleCommits) {
  absl::StatusOr<std::unique_ptr<PrivateState>> result =
      PrivateState::CreatePrivateState(std::nullopt);
  ASSERT_TRUE(result.ok());
  std::unique_ptr<PrivateState> state = std::move(result.value());

  // First commit: counter should be 1.
  std::string first_commit = state->CommitNewState();
  BudgetState first_state;
  ASSERT_TRUE(first_state.ParseFromString(first_commit));
  EXPECT_EQ(first_state.counter(), 1);

  // GetState should now return the committed state.
  ASSERT_TRUE(state->GetState().has_value());
  EXPECT_EQ(state->GetState().value(), first_commit);

  // Second commit: counter should be 2.
  BudgetState second_state;
  ASSERT_TRUE(second_state.ParseFromString(state->CommitNewState()));
  EXPECT_EQ(second_state.counter(), 2);
}

TEST(PrivateStateTest, AllowRecovery_MatchingRecoveryBlobId) {
  BudgetState initial_budget_state;
  initial_budget_state.set_recovery_blob_id("blob_123");

  absl::StatusOr<std::unique_ptr<PrivateState>> result =
      PrivateState::CreatePrivateState(
          initial_budget_state.SerializeAsString());
  ASSERT_TRUE(result.ok());
  std::unique_ptr<PrivateState> state = std::move(result.value());

  RecoveryInfo recovery_info;
  recovery_info.set_committed_blob_id("other_blob");

  // Recovery should be allowed because recovery_blob_id matches
  // the recovery blob id in the internal BudgetState.
  EXPECT_TRUE(state->AllowRecovery("blob_123", recovery_info));
}

TEST(PrivateStateTest, AllowRecovery_MatchingCommittedBlobId) {
  BudgetState initial_budget_state;
  initial_budget_state.set_recovery_blob_id("blob_123");

  absl::StatusOr<std::unique_ptr<PrivateState>> result =
      PrivateState::CreatePrivateState(
          initial_budget_state.SerializeAsString());
  ASSERT_TRUE(result.ok());
  std::unique_ptr<PrivateState> state = std::move(result.value());

  RecoveryInfo recovery_info;
  recovery_info.set_committed_blob_id("blob_123");

  // Recovery should be allowed because the committed blob id attached
  // to the recovery info matches the recovery blob id in the internal
  // BudgetState.
  EXPECT_TRUE(state->AllowRecovery("other_blob", recovery_info));
}

TEST(PrivateStateTest, AllowRecovery_NoMatch) {
  BudgetState initial_budget_state;
  initial_budget_state.set_recovery_blob_id("blob_123");

  absl::StatusOr<std::unique_ptr<PrivateState>> result =
      PrivateState::CreatePrivateState(
          initial_budget_state.SerializeAsString());
  ASSERT_TRUE(result.ok());
  std::unique_ptr<PrivateState> state = std::move(result.value());

  RecoveryInfo recovery_info;
  recovery_info.set_committed_blob_id("other_blob");

  // Recovery should be denied because neither the recovery_blob_id nor the
  // committed_blob_id matches.
  EXPECT_FALSE(state->AllowRecovery("different_blob", recovery_info));
}

TEST(PrivateStateTest, AllowRecovery_BothUnset) {
  // No recovery_blob_id in the initial state.
  absl::StatusOr<std::unique_ptr<PrivateState>> result =
      PrivateState::CreatePrivateState(std::nullopt);
  ASSERT_TRUE(result.ok());
  std::unique_ptr<PrivateState> state = std::move(result.value());

  // RecoveryInfo with no committed_blob_id set either.
  RecoveryInfo recovery_info;

  // Recovery should be allowed when both sides are in the initial state.
  EXPECT_TRUE(state->AllowRecovery("any_blob_id", recovery_info));
}

TEST(PrivateStateTest, AllowRecovery_OnlyInternalUnset) {
  absl::StatusOr<std::unique_ptr<PrivateState>> result =
      PrivateState::CreatePrivateState(std::nullopt);
  ASSERT_TRUE(result.ok());
  std::unique_ptr<PrivateState> state = std::move(result.value());

  RecoveryInfo recovery_info;
  recovery_info.set_committed_blob_id("some_blob");

  // Recovery should be denied: internal state has no recovery_blob_id but the
  // RecoveryInfo has a committed_blob_id, so they are out of sync.
  EXPECT_FALSE(state->AllowRecovery("any_blob_id", recovery_info));
}

TEST(PrivateStateTest, SetRecoveryBlobId) {
  absl::StatusOr<std::unique_ptr<PrivateState>> result =
      PrivateState::CreatePrivateState(std::nullopt);
  ASSERT_TRUE(result.ok());
  std::unique_ptr<PrivateState> state = std::move(result.value());

  state->SetRecoveryBlobId("new_blob_id");

  RecoveryInfo recovery_info;
  recovery_info.set_committed_blob_id("other_blob");

  // After setting the recovery blob ID, AllowRecovery should match on it.
  EXPECT_TRUE(state->AllowRecovery("new_blob_id", recovery_info));
  EXPECT_FALSE(state->AllowRecovery("wrong_blob_id", recovery_info));

  // CommitNewState should include the recovery blob ID.
  BudgetState committed;
  ASSERT_TRUE(committed.ParseFromString(state->CommitNewState()));
  EXPECT_EQ(committed.recovery_blob_id(), "new_blob_id");
}

TEST(PrivateStateTest, GetCommittedRecoveryBlobId_NoCommittedState) {
  absl::StatusOr<std::unique_ptr<PrivateState>> result =
      PrivateState::CreatePrivateState(std::nullopt);
  ASSERT_TRUE(result.ok());
  std::unique_ptr<PrivateState> state = std::move(result.value());

  // No committed state, so the committed recovery blob ID should be nullopt.
  EXPECT_EQ(state->GetCommittedRecoveryBlobId(), std::nullopt);
}

TEST(PrivateStateTest, GetCommittedRecoveryBlobId_FromInitialState) {
  BudgetState initial_budget_state;
  initial_budget_state.set_recovery_blob_id("initial_blob_id");

  absl::StatusOr<std::unique_ptr<PrivateState>> result =
      PrivateState::CreatePrivateState(
          initial_budget_state.SerializeAsString());
  ASSERT_TRUE(result.ok());
  std::unique_ptr<PrivateState> state = std::move(result.value());

  EXPECT_EQ(state->GetCommittedRecoveryBlobId(), "initial_blob_id");
}

TEST(PrivateStateTest, GetCommittedRecoveryBlobId_AfterCommit) {
  absl::StatusOr<std::unique_ptr<PrivateState>> result =
      PrivateState::CreatePrivateState(std::nullopt);
  ASSERT_TRUE(result.ok());
  std::unique_ptr<PrivateState> state = std::move(result.value());

  // No committed state yet.
  EXPECT_EQ(state->GetCommittedRecoveryBlobId(), std::nullopt);

  // Set a recovery blob ID and commit.
  state->SetRecoveryBlobId("new_blob_id");
  state->CommitNewState();

  // After committing, the committed recovery blob ID should be updated.
  EXPECT_EQ(state->GetCommittedRecoveryBlobId(), "new_blob_id");
}

TEST(PrivateStateTest, HasPriorSaveRecovery) {
  absl::StatusOr<std::unique_ptr<PrivateState>> result =
      PrivateState::CreatePrivateState(std::nullopt);
  ASSERT_TRUE(result.ok());
  std::unique_ptr<PrivateState> state = std::move(result.value());

  // Initially false for no recovery blob id.
  EXPECT_FALSE(state->HasPriorSaveRecovery());

  state->SetRecoveryBlobId("new_blob_id");
  // True after setting recovery blob id.
  EXPECT_TRUE(state->HasPriorSaveRecovery());

  BudgetState initial_budget_state;
  initial_budget_state.set_recovery_blob_id("initial_blob_id");
  absl::StatusOr<std::unique_ptr<PrivateState>> result2 =
      PrivateState::CreatePrivateState(
          initial_budget_state.SerializeAsString());
  ASSERT_TRUE(result2.ok());
  std::unique_ptr<PrivateState> state2 = std::move(result2.value());

  // True when initialized with state containing recovery blob id.
  EXPECT_TRUE(state2->HasPriorSaveRecovery());
}

}  // namespace
}  // namespace confidential_federated_compute::program_executor_tee