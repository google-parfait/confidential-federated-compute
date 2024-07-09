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

#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "grpcpp/support/status.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute {
namespace {

using ::fcp::confidentialcompute::SessionResponse;

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

TEST(SessionTest, ErrorToSessionWriteFinishedResponseTest) {
  SessionResponse response = ToSessionWriteFinishedResponse(
      absl::InvalidArgumentError("invalid arg"), 42);
  ASSERT_TRUE(response.has_write());
  EXPECT_EQ(response.write().committed_size_bytes(), 0);
  EXPECT_EQ(response.write().write_capacity_bytes(), 42);
  EXPECT_EQ(response.write().status().code(),
            grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(response.write().status().message(), "invalid arg");
}

TEST(SessionTest, OkToSessionWriteFinishedResponseTest) {
  SessionResponse response =
      ToSessionWriteFinishedResponse(absl::OkStatus(), 42, 6);
  ASSERT_TRUE(response.has_write());
  EXPECT_EQ(response.write().committed_size_bytes(), 6);
  EXPECT_EQ(response.write().write_capacity_bytes(), 42);
  EXPECT_EQ(response.write().status().code(), grpc::StatusCode::OK);
  EXPECT_TRUE(response.write().status().message().empty());
}

}  // namespace
}  // namespace confidential_federated_compute
