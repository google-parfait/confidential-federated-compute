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

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "containers/common/intervals/interval.h"
#include "containers/common/time_budget/budget.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute::mauve_score {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

TEST(BudgetTest, FirstRunWithEmptyState) {
  auto budget =
      Budget::Create(/*initial_state=*/std::nullopt, /*default_budget=*/5);
  ASSERT_THAT(budget, IsOk());

  EXPECT_EQ(budget->GetInitialState(), std::nullopt);
  EXPECT_THAT(budget->UpdatePerKeyBudget({"key1"}), IsOk());

  BudgetState dst;
  ASSERT_TRUE(dst.ParseFromString(budget->SerializeAsString()));
  ASSERT_EQ(dst.buckets_size(), 1);
  EXPECT_EQ(dst.buckets(0).key(), "key1");
  EXPECT_EQ(dst.buckets(0).budget(), 4);
}

TEST(BudgetTest, SubsequentRunParsesState) {
  BudgetState prior_state;
  auto* bucket = prior_state.add_buckets();
  bucket->set_key("key1");
  bucket->set_budget(3);
  std::string serialized = prior_state.SerializeAsString();

  auto budget =
      Budget::Create(/*initial_state=*/serialized, /*default_budget=*/5);
  ASSERT_THAT(budget, IsOk());

  EXPECT_EQ(budget->GetInitialState(), serialized);
  EXPECT_THAT(budget->UpdatePerKeyBudget({"key1"}), IsOk());

  BudgetState dst;
  ASSERT_TRUE(dst.ParseFromString(budget->SerializeAsString()));
  ASSERT_EQ(dst.buckets_size(), 1);
  EXPECT_EQ(dst.buckets(0).key(), "key1");
  EXPECT_EQ(dst.buckets(0).budget(), 2);
}

TEST(BudgetTest, ZeroDefaultBudgetRejected) {
  EXPECT_THAT(
      Budget::Create(/*initial_state=*/std::nullopt, /*default_budget=*/0),
      StatusIs(absl::StatusCode::kResourceExhausted));
}

TEST(BudgetTest, InvalidSerializedStateRejected) {
  EXPECT_THAT(Budget::Create(/*initial_state=*/"not_a_valid_proto_wire_format",
                             /*default_budget=*/5),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(BudgetTest, PerKeyBudgetExhaustion) {
  auto budget =
      Budget::Create(/*initial_state=*/std::nullopt, /*default_budget=*/1);
  ASSERT_THAT(budget, IsOk());

  EXPECT_THAT(budget->UpdatePerKeyBudget({"key1"}), IsOk());
  EXPECT_FALSE(budget->HasRemainingBudget("key1"));
  EXPECT_TRUE(budget->HasRemainingBudget("key2"));

  // Updating key1 again fails because its budget is 0.
  EXPECT_THAT(budget->UpdatePerKeyBudget({"key1"}),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // Updating key2 still succeeds.
  EXPECT_THAT(budget->UpdatePerKeyBudget({"key2"}), IsOk());
}

TEST(BudgetTest, MultipleKeysAtomicUpdate) {
  BudgetState prior_state;
  auto* b1 = prior_state.add_buckets();
  b1->set_key("key1");
  b1->set_budget(0);
  auto* b2 = prior_state.add_buckets();
  b2->set_key("key2");
  b2->set_budget(2);

  auto budget = Budget::Create(prior_state.SerializeAsString(),
                               /*default_budget=*/5);
  ASSERT_THAT(budget, IsOk());

  // Updating both together should fail without decrementing key2.
  EXPECT_THAT(budget->UpdatePerKeyBudget({"key1", "key2"}),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  BudgetState dst = budget->Serialize();
  ASSERT_EQ(dst.buckets_size(), 2);
  EXPECT_EQ(dst.buckets(1).key(), "key2");
  EXPECT_EQ(dst.buckets(1).budget(), 2);
}

TEST(BudgetTest, TimeBudgetFirstAndSubsequentRuns) {
  auto budget1 =
      Budget::Create(/*initial_state=*/std::nullopt, /*default_budget=*/2);
  ASSERT_THAT(budget1, IsOk());

  Interval<uint64_t> window1(0, 600);
  EXPECT_THAT(budget1->UpdateTimeBudget(window1), IsOk());

  auto budget2 = Budget::Create(budget1->SerializeAsString(),
                                /*default_budget=*/2);
  ASSERT_THAT(budget2, IsOk());

  Interval<uint64_t> window2(300, 900);
  EXPECT_THAT(budget2->UpdateTimeBudget(window2), IsOk());

  auto budget3 = Budget::Create(budget2->SerializeAsString(),
                                /*default_budget=*/2);
  ASSERT_THAT(budget3, IsOk());

  // [300, 600) has been consumed twice (budget 0 remaining).
  EXPECT_FALSE(budget3->HasRemainingBudget(Interval<uint64_t>(300, 600)));
  EXPECT_THAT(budget3->UpdateTimeBudget(Interval<uint64_t>(300, 600)),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // [600, 900) was only consumed once (budget 1 remaining).
  EXPECT_TRUE(budget3->HasRemainingBudget(Interval<uint64_t>(600, 900)));
  EXPECT_THAT(budget3->UpdateTimeBudget(Interval<uint64_t>(600, 900)), IsOk());
}

TEST(BudgetTest, PerKeyUpdateRejectedAfterTransitionToTimeBudget) {
  auto budget =
      Budget::Create(/*initial_state=*/std::nullopt, /*default_budget=*/2);
  ASSERT_THAT(budget, IsOk());

  EXPECT_THAT(budget->UpdateTimeBudget(Interval<uint64_t>(0, 600)), IsOk());

  // Subsequent per-key budget updates should be rejected.
  EXPECT_THAT(budget->UpdatePerKeyBudget({"key1"}),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

}  // namespace
}  // namespace confidential_federated_compute::mauve_score
