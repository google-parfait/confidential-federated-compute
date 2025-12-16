// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may not obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "containers/fns/map_fn.h"

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "containers/testing/mocks.h"
#include "fcp/protos/confidentialcompute/tee_payload_metadata.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/any.pb.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::fns {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::ConfigureRequest;
using ::fcp::confidentialcompute::ConfigureResponse;
using ::fcp::confidentialcompute::FinalizeRequest;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::WriteFinishedResponse;
using ::fcp::confidentialcompute::WriteRequest;
using ::google::protobuf::Any;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SaveArg;
using ::testing::StrictMock;
using ::testing::Test;

class MockMapFn : public MapFn {
 public:
  MOCK_METHOD(absl::Status, InitializeReplica, (Any config, Context& context),
              (override));
  MOCK_METHOD(absl::StatusOr<KeyValue>, Map, (KeyValue input, Context& context),
              (override));
  MOCK_METHOD(absl::Status, FinalizeReplica, (Any config, Context& context),
              (override));
};

class MapFnTest : public Test {
 protected:
  void SetUp() override { session_ = std::make_unique<MockMapFn>(); }

  std::unique_ptr<MockMapFn> session_;
  StrictMock<MockContext> context_;
};

TEST_F(MapFnTest, ConfigureCallsInitialize) {
  Any config;
  config.set_type_url("foo");
  ConfigureRequest request;
  *request.mutable_configuration() = config;

  Any passed_config;
  EXPECT_CALL(*session_, InitializeReplica(_, _))
      .WillOnce(DoAll(SaveArg<0>(&passed_config), Return(absl::OkStatus())));

  EXPECT_THAT(session_->Configure(request, context_),
              IsOkAndHolds(testing::An<ConfigureResponse>()));
  EXPECT_EQ(passed_config.type_url(), "foo");
}

TEST_F(MapFnTest, WriteCallsMapAndEmit) {
  Any config;
  config.set_type_url("bar");
  BlobMetadata metadata;
  metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_GZIP);
  std::string data = "somedata";

  WriteRequest request;
  *request.mutable_first_request_configuration() = config;
  *request.mutable_first_request_metadata() = metadata;

  KeyValue output_key_value;
  output_key_value.value.data = "output_data";
  output_key_value.key.set_type_url("output_key");

  KeyValue passed_input;
  EXPECT_CALL(*session_, Map(_, _))
      .WillOnce(DoAll(SaveArg<0>(&passed_input), Return(output_key_value)));

  ReadResponse emitted_response;
  EXPECT_CALL(context_, Emit(_))
      .WillOnce(DoAll(SaveArg<0>(&emitted_response), Return(true)));

  absl::StatusOr<WriteFinishedResponse> result =
      session_->Write(request, data, context_);
  ASSERT_THAT(result, IsOk());
  EXPECT_EQ(result->committed_size_bytes(), data.size());

  EXPECT_EQ(passed_input.key.type_url(), "bar");
  EXPECT_EQ(passed_input.value.metadata.compression_type(),
            BlobMetadata::COMPRESSION_TYPE_GZIP);
  EXPECT_EQ(passed_input.value.data, "somedata");
  EXPECT_EQ(emitted_response.data(), "output_data");
  EXPECT_EQ(emitted_response.first_response_configuration().type_url(),
            "output_key");
}

TEST_F(MapFnTest, FinalizeCallsFinalizeReplica) {
  Any config;
  config.set_type_url("baz");
  FinalizeRequest request;
  *request.mutable_configuration() = config;
  BlobMetadata metadata;

  Any passed_config;
  EXPECT_CALL(*session_, FinalizeReplica(_, _))
      .WillOnce(DoAll(SaveArg<0>(&passed_config), Return(absl::OkStatus())));

  EXPECT_THAT(session_->Finalize(request, metadata, context_), IsOk());
  EXPECT_EQ(passed_config.type_url(), "baz");
}

TEST_F(MapFnTest, CommitIsNoOp) {
  fcp::confidentialcompute::CommitRequest request;
  EXPECT_THAT(session_->Commit(request, context_), IsOk());
}

}  // namespace
}  // namespace confidential_federated_compute::fns
