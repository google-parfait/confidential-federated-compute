// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "containers/session.h"

#include "gtest/gtest.h"

namespace confidential_federated_compute {
namespace {

TEST(SessionTest, AddSession) {
  SessionTracker session_tracker(1, 100);
  ASSERT_GT(session_tracker.AddSession(), 0);
}

TEST(SessionTest, MaximumSessionsReachedAddSession) {
  SessionTracker session_tracker(1, 100);
  ASSERT_GT(session_tracker.AddSession(), 0);
  ASSERT_EQ(session_tracker.AddSession(), 0);
}

TEST(SessionTest, MaximumSessionsReachedCanAddSessionAfterRemoveSession) {
  SessionTracker session_tracker(1, 100);
  ASSERT_GT(session_tracker.AddSession(), 0);
  ASSERT_EQ(session_tracker.AddSession(), 0);
  ASSERT_TRUE(session_tracker.RemoveSession().ok());
  ASSERT_GT(session_tracker.AddSession(), 0);
}

TEST(SessionTest, RemoveSessionWithoutAddSessionFails) {
  SessionTracker session_tracker(1, 100);
  ASSERT_EQ(session_tracker.RemoveSession().code(),
            absl::StatusCode::kFailedPrecondition);
}

}  // namespace
}  // namespace confidential_federated_compute
