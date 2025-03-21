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

#include <thread>

#include "absl/time/clock.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "gmock/gmock.h"
#include "grpcpp/support/status.h"
#include "gtest/gtest.h"
#include "testing/matchers.h"

namespace confidential_federated_compute {
namespace {

using ::fcp::confidentialcompute::SessionResponse;

TEST(SessionTest, AddSession) {
  SessionTracker session_tracker(1);
  EXPECT_THAT(session_tracker.AddSession(), IsOk());
}

TEST(SessionTest, MaximumSessionsReachedAddSession) {
  SessionTracker session_tracker(1);
  EXPECT_THAT(session_tracker.AddSession(), IsOk());
  EXPECT_THAT(session_tracker.AddSession(),
              IsCode(absl::StatusCode::kFailedPrecondition));
}

TEST(SessionTest, MaximumSessionsReachedCanAddSessionAfterRemoveSession) {
  SessionTracker session_tracker(1);
  EXPECT_THAT(session_tracker.AddSession(), IsOk());
  EXPECT_THAT(session_tracker.AddSession(),
              IsCode(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(session_tracker.RemoveSession(), IsOk());
  EXPECT_THAT(session_tracker.AddSession(), IsOk());
}

TEST(SessionTest, RemoveSessionWithoutAddSessionFails) {
  SessionTracker session_tracker(1);
  EXPECT_THAT(session_tracker.RemoveSession(),
              IsCode(absl::StatusCode::kFailedPrecondition));
}

TEST(SessionTest, ErrorToSessionWriteFinishedResponseTest) {
  SessionResponse response =
      ToSessionWriteFinishedResponse(absl::InvalidArgumentError("invalid arg"));
  ASSERT_TRUE(response.has_write());
  EXPECT_EQ(response.write().committed_size_bytes(), 0);
  EXPECT_EQ(response.write().status().code(),
            grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(response.write().status().message(), "invalid arg");
}

TEST(SessionTest, OkToSessionWriteFinishedResponseTest) {
  SessionResponse response =
      ToSessionWriteFinishedResponse(absl::OkStatus(), 6);
  ASSERT_TRUE(response.has_write());
  EXPECT_EQ(response.write().committed_size_bytes(), 6);
  EXPECT_EQ(response.write().status().code(), grpc::StatusCode::OK);
  EXPECT_TRUE(response.write().status().message().empty());
}

TEST(SessionTest, MaximumSessionsReachedConcurrentAddRemove) {
  SessionTracker session_tracker(1);
  EXPECT_THAT(session_tracker.AddSession(), IsOk());
  std::thread t([&]() {
    absl::SleepFor(absl::Milliseconds(10));
    EXPECT_THAT(session_tracker.RemoveSession(), IsOk());
  });
  EXPECT_THAT(session_tracker.AddSession(), IsOk());
  t.join();
}

}  // namespace
}  // namespace confidential_federated_compute
