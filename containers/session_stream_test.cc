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

// This file contains functions and classes for managing aggregation sessions
// of a ConfidentialTransform service.

#include "containers/session_stream.h"

#include <vector>

#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "gmock/gmock.h"
#include "grpcpp/client_context.h"
#include "grpcpp/test/mock_stream.h"
#include "gtest/gtest.h"
#include "testing/matchers.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SessionResponse;
using ::grpc::testing::MockServerReaderWriter;
using ::testing::_;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Return;

class SessionStreamTest : public ::testing::Test {
 protected:
  SessionStreamTest() : session_stream_(&mock_stream_, /*chunk_size=*/16) {
    ON_CALL(mock_stream_, Write).WillByDefault(Return(true));
  }

  void ExpectReads(std::vector<SessionRequest> requests) {
    struct ReadContext {
      int index = 0;
      std::vector<SessionRequest> requests;
    };
    auto context = std::make_shared<ReadContext>(ReadContext{
        .requests = std::move(requests),
    });
    EXPECT_CALL(mock_stream_, Read)
        .WillRepeatedly([context](SessionRequest* request) {
          if (context->index < context->requests.size()) {
            *request = std::move(context->requests[context->index++]);
            return true;
          }
          return false;
        });
  }

  void ExpectReads(std::vector<absl::string_view> requests) {
    std::vector<SessionRequest> parsed_requests;
    for (const auto& request : requests) {
      SessionRequest parsed_request = PARSE_TEXT_PROTO(request);
      parsed_requests.push_back(parsed_request);
    }
    ExpectReads(parsed_requests);
  }

  void ExpectRead(SessionRequest request) { ExpectReads({request}); }

  void ExpectRead(absl::string_view request) {
    SessionRequest parsed_request = PARSE_TEXT_PROTO(request);
    ExpectRead(parsed_request);
  }

  auto& ExpectWrite(SessionResponse response) {
    return EXPECT_CALL(mock_stream_, Write(EqualsProto(response), _));
  }

  auto& ExpectWrite(absl::string_view response) {
    SessionResponse parsed_response = PARSE_TEXT_PROTO(response);
    return ExpectWrite(parsed_response);
  }

  void ExpectWrites(std::vector<absl::string_view> responses) {
    for (const auto& response : responses) {
      ExpectWrite(response);
    }
  }

  NiceMock<MockServerReaderWriter<SessionResponse, SessionRequest>>
      mock_stream_;
  SessionStream session_stream_;
};

TEST_F(SessionStreamTest, SingleRead) {
  SessionRequest request = PARSE_TEXT_PROTO("finalize {}");
  ExpectRead(request);
  EXPECT_THAT(session_stream_.Read(), IsOkAndHolds(EqualsProto(request)));
}

TEST_F(SessionStreamTest, ConfigureRead) {
  SessionRequest request = PARSE_TEXT_PROTO("configure { chunk_size: 3 }");
  ExpectRead(request);
  EXPECT_THAT(session_stream_.Read(), IsOkAndHolds(EqualsProto(request)));
  EXPECT_EQ(session_stream_.chunk_size(), 3);
}

TEST_F(SessionStreamTest, ConfigureReadInvalidChunkSize) {
  SessionRequest request = PARSE_TEXT_PROTO("configure { chunk_size: 0 }");
  ExpectRead(request);
  EXPECT_THAT(session_stream_.Read(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid chunk size")));
}

TEST_F(SessionStreamTest, WriteRequestSingleRead) {
  SessionRequest request = PARSE_TEXT_PROTO(R"pb(
    write {
      first_request_metadata { unencrypted {} }
      first_request_configuration {}
      commit: true
      data: "data"
    }
  )pb");
  ExpectRead(request);
  EXPECT_THAT(session_stream_.Read(), IsOkAndHolds(EqualsProto(request)));
}

TEST_F(SessionStreamTest, WriteRequestChunkedRead) {
  ExpectReads({
      R"pb(
        write {
          first_request_metadata { unencrypted {} }
          first_request_configuration {}
          data: "quick brown fox "
        }
      )pb",
      R"pb(
        write { data: "jumps over the l" }
      )pb",
      R"pb(
        write { commit: true data: "azy dog" }
      )pb"});
  EXPECT_THAT(session_stream_.Read(),
              IsOkAndHolds(EqualsProto(
                  R"pb(
                    write {
                      first_request_metadata { unencrypted {} }
                      first_request_configuration {}
                      data: "quick brown fox jumps over the lazy dog"
                      commit: true
                    }
                  )pb")));
}

TEST_F(SessionStreamTest, MergeRequestSingleRead) {
  SessionRequest request = PARSE_TEXT_PROTO(R"pb(
    merge {
      first_request_metadata { unencrypted {} }
      first_request_configuration {}
      commit: true
      data: "data"
    }
  )pb");
  ExpectRead(request);
  EXPECT_THAT(session_stream_.Read(), IsOkAndHolds(EqualsProto(request)));
}

TEST_F(SessionStreamTest, MergeRequestChunkedRead) {
  ExpectReads({
      R"pb(
        merge {
          first_request_metadata { unencrypted {} }
          first_request_configuration {}
          data: "quick brown fox "
        }
      )pb",
      R"pb(
        merge { data: "jumps over the l" }
      )pb",
      R"pb(
        merge { commit: true data: "azy dog" }
      )pb"});
  EXPECT_THAT(session_stream_.Read(),
              IsOkAndHolds(EqualsProto(
                  R"pb(
                    merge {
                      first_request_metadata { unencrypted {} }
                      first_request_configuration {}
                      data: "quick brown fox jumps over the lazy dog"
                      commit: true
                    }
                  )pb")));
}

TEST_F(SessionStreamTest, SingleReadFailure) {
  EXPECT_CALL(mock_stream_, Read).WillOnce(Return(false));
  EXPECT_THAT(session_stream_.Read(), StatusIs(absl::StatusCode::kAborted,
                                               HasSubstr("failed to read")));
}

TEST_F(SessionStreamTest, UncommittedWriteRequestFollowedByNewWriteRequest) {
  ExpectReads({
      R"pb(
        write {
          first_request_metadata {}
          data: "foo"
        }
      )pb",
      R"pb(
        write {
          first_request_metadata {}
          data: "bar"
        }
      )pb"});
  EXPECT_THAT(session_stream_.Read(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("received another request")));
}

TEST_F(SessionStreamTest, UncommittedMergeRequestFollowedByNewMergeRequest) {
  ExpectReads({
      R"pb(
        merge {
          first_request_metadata {}
          data: "foo"
        }
      )pb",
      R"pb(
        merge {
          first_request_metadata {}
          data: "bar"
        }
      )pb"});
  EXPECT_THAT(session_stream_.Read(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("received another request")));
}

TEST_F(SessionStreamTest, UncommittedWriteRequestFollowedByMergeRequest) {
  ExpectReads({
      R"pb(
        write {
          first_request_metadata {}
          data: "foo"
        }
      )pb",
      R"pb(
        merge { data: "bar" }
      )pb"});
  EXPECT_THAT(session_stream_.Read(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("received another request")));
}

TEST_F(SessionStreamTest, UncommittedWriteFollowedByOtherRequest) {
  ExpectReads({
      R"pb(
        write {
          first_request_metadata {}
          data: "foo"
        }
      )pb",
      "finalize {}"});
  EXPECT_THAT(session_stream_.Read(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("received another request")));
}

TEST_F(SessionStreamTest, SingleWrite) {
  SessionResponse response = PARSE_TEXT_PROTO("configure {}");
  ExpectWrite(response);
  EXPECT_THAT(session_stream_.Write(response), IsOk());
}

TEST_F(SessionStreamTest, ReadResponseSingleWrite) {
  // Verify that finish_read is added
  ExpectWrite(R"pb(
    read {
      first_response_metadata {}
      data: "foo"
      finish_read: true
    }
  )pb");
  EXPECT_THAT(session_stream_.Write(PARSE_TEXT_PROTO(R"pb(
    read {
      first_response_metadata {}
      data: "foo"
    }
  )pb")),
              IsOk());
}

TEST_F(SessionStreamTest, ReadResponseChunkedWrite) {
  ExpectWrites({
      R"pb(
        read {
          first_response_metadata { unencrypted {} }
          first_response_configuration {}
          data: "quick brown fox "
        }
      )pb",
      R"pb(
        read { data: "jumps over the l" }
      )pb",
      R"pb(
        read { finish_read: true data: "azy dog" }
      )pb"});
  EXPECT_THAT(session_stream_.Write(PARSE_TEXT_PROTO(R"pb(
    read {
      first_response_metadata { unencrypted {} }
      first_response_configuration {}
      data: "quick brown fox jumps over the lazy dog"
    }
  )pb")),
              IsOk());
}

TEST_F(SessionStreamTest, SingleWriteFailure) {
  EXPECT_CALL(mock_stream_, Write).WillOnce(Return(false));
  EXPECT_THAT(
      session_stream_.Write(PARSE_TEXT_PROTO("finalize {}")),
      StatusIs(absl::StatusCode::kAborted, HasSubstr("failed to write")));
}

}  // namespace
}  // namespace confidential_federated_compute
