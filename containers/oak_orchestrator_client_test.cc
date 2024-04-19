// Copyright 2024 Google LLC.
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
#include "containers/oak_orchestrator_client.h"

#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "absl/status/status.h"
#include "gmock/gmock.h"
#include "google/protobuf/empty.pb.h"
#include "gtest/gtest.h"
#include "proto/containers/interfaces_mock.grpc.pb.h"

namespace confidential_federated_compute {

namespace {

using ::oak::containers::MockOrchestratorStub;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;

TEST(OakOrchestratorClientTest, NotifyAppReadySucceeds) {
  MockOrchestratorStub mock_orchestrator_stub;

  google::protobuf::Empty response;

  // Set expectation and action.
  EXPECT_CALL(mock_orchestrator_stub, NotifyAppReady(_, _, _))
      .WillOnce(DoAll(SetArgPointee<2>(response), Return(grpc::Status::OK)));

  // Create client and injecting mock stub.
  OakOrchestratorClient client(&mock_orchestrator_stub);
  // The client call should succeed.
  absl::Status status = client.NotifyAppReady();
  ASSERT_TRUE(status.ok()) << status;
}

TEST(OakOrchestratorClientTest, NotifyAppReadyFails) {
  MockOrchestratorStub mock_orchestrator_stub;

  google::protobuf::Empty response;

  // Set expectation and action.
  EXPECT_CALL(mock_orchestrator_stub, NotifyAppReady(_, _, _))
      .WillOnce(DoAll(SetArgPointee<2>(response),
                      Return(grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                                          "error message"))));

  // Create client and injecting mock stub.
  OakOrchestratorClient client(&mock_orchestrator_stub);
  // The client call should fail.
  absl::Status status = client.NotifyAppReady();
  ASSERT_EQ(status.code(), absl::StatusCode::kInternal);
}

}  // namespace

}  // namespace confidential_federated_compute
