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

#include "absl/status/status_matchers.h"
#include "absl/time/clock.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/any.pb.h"
#include "grpcpp/support/status.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::fcp::confidentialcompute::CommitResponse;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::WriteFinishedResponse;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::IsEmpty;
using ::testing::Property;

MATCHER_P2(RpcStatusIs, expected_code, message_matcher, "") {
  return ExplainMatchResult(
      AllOf(Property(&google::rpc::Status::code,
                     Eq(static_cast<int>(expected_code))),
            Property(&google::rpc::Status::message, message_matcher)),
      arg, result_listener);
}
TEST(SessionTest, AddSession) {
  SessionTracker session_tracker(1);
  EXPECT_THAT(session_tracker.AddSession(), IsOk());
}

TEST(SessionTest, MaximumSessionsReachedAddSession) {
  SessionTracker session_tracker(1);
  EXPECT_THAT(session_tracker.AddSession(), IsOk());
  EXPECT_THAT(session_tracker.AddSession(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(SessionTest, MaximumSessionsReachedCanAddSessionAfterRemoveSession) {
  SessionTracker session_tracker(1);
  EXPECT_THAT(session_tracker.AddSession(), IsOk());
  EXPECT_THAT(session_tracker.AddSession(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(session_tracker.RemoveSession(), IsOk());
  EXPECT_THAT(session_tracker.AddSession(), IsOk());
}

TEST(SessionTest, RemoveSessionWithoutAddSessionFails) {
  SessionTracker session_tracker(1);
  EXPECT_THAT(session_tracker.RemoveSession(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(SessionTest, ErrorToWriteFinishedResponseTest) {
  WriteFinishedResponse response =
      ToWriteFinishedResponse(absl::InvalidArgumentError("invalid arg"));
  EXPECT_EQ(response.committed_size_bytes(), 0);
  EXPECT_EQ(response.status().code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(response.status().message(), "invalid arg");
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

TEST(SessionTest, OkToWriteFinishedResponseTest) {
  WriteFinishedResponse response = ToWriteFinishedResponse(absl::OkStatus(), 6);
  EXPECT_EQ(response.committed_size_bytes(), 6);
  EXPECT_EQ(response.status().code(), grpc::StatusCode::OK);
  EXPECT_TRUE(response.status().message().empty());
}

TEST(SessionTest, OkToSessionWriteFinishedResponseTest) {
  SessionResponse response =
      ToSessionWriteFinishedResponse(absl::OkStatus(), 6);
  ASSERT_TRUE(response.has_write());
  EXPECT_EQ(response.write().committed_size_bytes(), 6);
  EXPECT_EQ(response.write().status().code(), grpc::StatusCode::OK);
  EXPECT_TRUE(response.write().status().message().empty());
}

TEST(SessionTest, ErrorToCommitResponseTest) {
  CommitResponse response = ToCommitResponse(
      /*status=*/absl::InvalidArgumentError("invalid arg"),
      /*num_inputs_committed*/ 0,
      /*ignored_errors=*/std::vector<absl::Status>{});
  EXPECT_THAT(
      response,
      AllOf(
          Property(
              &CommitResponse::status,
              RpcStatusIs(grpc::StatusCode::INVALID_ARGUMENT, "invalid arg")),
          Property(
              &CommitResponse::stats,
              AllOf(Property(&CommitResponse::CommitStats::num_inputs_committed,
                             Eq(0)),
                    Property(&CommitResponse::CommitStats::ignored_errors_size,
                             Eq(0))))));
}

TEST(SessionTest, OkToCommitResponseTest) {
  CommitResponse response = ToCommitResponse(
      /*status=*/absl::OkStatus(), /*num_inputs_committed=*/42,
      /*ignored_errors=*/
      std::vector<absl::Status>{absl::InvalidArgumentError("ignored")});
  EXPECT_THAT(
      response,
      AllOf(
          Property(&CommitResponse::status,
                   RpcStatusIs(grpc::StatusCode::OK, IsEmpty())),
          Property(
              &CommitResponse::stats,
              AllOf(Property(&CommitResponse::CommitStats::num_inputs_committed,
                             Eq(42)),
                    Property(&CommitResponse::CommitStats::ignored_errors_size,
                             Eq(1)),
                    Property(&CommitResponse::CommitStats::ignored_errors,
                             ElementsAre(
                                 RpcStatusIs(grpc::StatusCode::INVALID_ARGUMENT,
                                             "ignored")))))));
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

TEST(SessionKVTest, Construct) {
  google::protobuf::Any key;
  key.set_type_url("foo");
  auto kv = Session::KV(key, "v", "id");
  EXPECT_EQ(kv.key.type_url(), "foo");
  EXPECT_EQ(kv.data, "v");
  EXPECT_EQ(kv.blob_id, "id");
}

TEST(SessionKVTest, ConstructWithRandomBlobId) {
  Session::KV kv(google::protobuf::Any(), "bar");
  EXPECT_GT(kv.blob_id.size(), 0);
}

TEST(SessionKVTest, ConstructFromString) {
  Session::KV kv = "foobar";
  EXPECT_EQ(kv.data, "foobar");
}

}  // namespace
}  // namespace confidential_federated_compute
