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

#include "transform/willow_transform_service.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "fcp/base/random_token.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "gmock/gmock.h"
#include "grpcpp/channel.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "gtest/gtest.h"
#include "transform/willow_messages.pb.h"
#include "willow/api/client.h"
#include "willow/api/server_accumulator.h"
#include "willow/input_encoding/codec.h"
#include "willow/input_encoding/codec_factory.h"
#include "willow/proto/willow/aggregation_config.pb.h"
#include "willow/testing_utils/shell_testing_decryptor.h"
#include "willow/testing_utils/testing_utils.h"

namespace confidential_federated_compute::willow {
namespace {

using ::absl_testing::IsOk;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::CommitRequest;
using ::fcp::confidentialcompute::ConfidentialTransform;
using ::fcp::confidentialcompute::FinalizeRequest;
using ::fcp::confidentialcompute::InitializeRequest;
using ::fcp::confidentialcompute::InitializeResponse;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::StreamInitializeRequest;
using ::fcp::confidentialcompute::WriteRequest;
using ::google::protobuf::Any;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::secure_aggregation::willow::AggregationConfigProto;
using ::secure_aggregation::willow::CodecFactory;
using ::secure_aggregation::willow::FinalizedAccumulatorResult;
using ::secure_aggregation::willow::FinalResultDecryptor;
using ::testing::ElementsAre;
using ::testing::Pair;
using ::testing::Test;
using ::testing::UnorderedElementsAre;

class WillowTransformServiceTest : public Test {
 public:
  WillowTransformServiceTest() {
    int port;
    const std::string server_address = "[::1]:";

    service_ = std::make_unique<WillowTransformService>();

    ServerBuilder builder;
    builder.AddListeningPort(server_address + "0",
                             grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(service_.get());
    server_ = builder.BuildAndStart();
    LOG(INFO) << "Server listening on " << server_address + std::to_string(port)
              << std::endl;
    stub_ = ConfidentialTransform::NewStub(
        grpc::CreateChannel(server_address + std::to_string(port),
                            grpc::InsecureChannelCredentials()));

    aggregation_config_ =
        secure_aggregation::willow::CreateTestAggregationConfigProto();
  }

  ~WillowTransformServiceTest() override { server_->Shutdown(); }

 protected:
  bool InitializeTransform() {
    grpc::ClientContext context;
    InitializeRequest request;
    InitializeResponse response;
    request.set_max_num_sessions(8);

    // Add Aggregation Configuration to the request.
    request.mutable_configuration()->PackFrom(aggregation_config_);

    auto init_stream = stub_->StreamInitialize(&context, &response);
    StreamInitializeRequest stream_request;
    *stream_request.mutable_initialize_request() = std::move(request);
    bool init_result =
        init_stream->Write(stream_request) && init_stream->WritesDone();
    auto init_status = init_stream->Finish();
    if (!init_status.ok()) {
      LOG(ERROR) << init_status.error_message();
    }
    return init_result && init_status.ok();
  }

  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
  StartSession(grpc::ClientContext* context) {
    SessionRequest session_request;
    SessionResponse session_response;
    // TODO: implement chunking
    session_request.mutable_configure()->set_chunk_size(1000);

    std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
        stream;
    stream = stub_->Session(context);
    CHECK(stream->Write(session_request));
    CHECK(stream->Read(&session_response));
    return stream;
  }

  SessionRequest CreateWriteRequest(std::string blob_id, WillowOp::Kind op_kind,
                                    std::string data) {
    SessionRequest request;
    WriteRequest* write_request = request.mutable_write();
    BlobMetadata metadata;
    metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_NONE);
    metadata.mutable_unencrypted()->set_blob_id(blob_id);
    metadata.set_total_size_bytes(data.size());
    *write_request->mutable_first_request_metadata() = metadata;
    WillowOp op;
    op.set_kind(op_kind);
    write_request->mutable_first_request_configuration()->PackFrom(op);
    write_request->set_commit(true);
    write_request->set_data(data);
    return request;
  }

  SessionRequest CreateCommitRequest(std::string range_start,
                                     std::string range_end) {
    SessionRequest request;
    CommitRequest* commit_request = request.mutable_commit();
    RangeProto range;
    range.set_start(range_start);
    range.set_end(range_end);
    commit_request->mutable_configuration()->PackFrom(range);
    return request;
  }

  SessionRequest CreateFinalizeRequest(WillowOp::Kind op_kind) {
    SessionRequest request;
    FinalizeRequest* finalize_request = request.mutable_finalize();
    WillowOp op;
    op.set_kind(op_kind);
    finalize_request->mutable_configuration()->PackFrom(op);
    return request;
  }

  std::unique_ptr<ConfidentialTransform::Stub> stub_;
  std::unique_ptr<WillowTransformService> service_;
  std::unique_ptr<Server> server_;
  AggregationConfigProto aggregation_config_;
};

// The simplest case with a single Willow encrypted contribution
// and immediate finalization without merging accumulators.
TEST_F(WillowTransformServiceTest, SingleInputNoMerge) {
  // Initialize decryptor and generate public key.
  auto decryptor = secure_aggregation::testing::ShellTestingDecryptor::Create(
      aggregation_config_);
  ASSERT_THAT(decryptor, IsOk());
  auto public_key = (*decryptor)->GeneratePublicKey();
  ASSERT_THAT(public_key, IsOk());

  // Create and encode input.
  auto metric_data = secure_aggregation::willow::CreateTestMetricData();
  auto group_by_data = secure_aggregation::willow::CreateTestGroupData();
  auto input_spec = secure_aggregation::willow::CreateTestInputSpecProto();
  auto encoder = CodecFactory::CreateExplicitCodec(input_spec);
  ASSERT_THAT(encoder, IsOk());
  auto encoded_data = (*encoder)->Encode(group_by_data, metric_data);
  ASSERT_THAT(encoded_data, IsOk());

  // Generate client contribution, encrypted towards public key with
  // server-provided nonce.
  std::string nonce = fcp::RandomToken::Generate().ToString();
  auto client_message = secure_aggregation::GenerateClientContribution(
      aggregation_config_, *encoded_data, *public_key, nonce);
  ASSERT_THAT(client_message, IsOk());

  ASSERT_TRUE(InitializeTransform());
  grpc::ClientContext context;
  auto stream = StartSession(&context);

  // Create and write the serialized client message to the session.
  SessionRequest write_request = CreateWriteRequest(
      nonce, WillowOp::ADD_INPUT, client_message->SerializeAsString());
  SessionResponse write_response;
  ASSERT_TRUE(stream->Write(write_request));
  ASSERT_TRUE(stream->Read(&write_response));
  ASSERT_TRUE(write_response.has_write());
  EXPECT_EQ(write_response.write().status().code(), grpc::OK);

  // Create and write the commit that covers the entire range.
  SessionRequest commit_request =
      CreateCommitRequest(std::string(16, 0x00), std::string(16, 0xFF));
  SessionResponse commit_response;
  ASSERT_TRUE(stream->Write(commit_request));
  ASSERT_TRUE(stream->Read(&commit_response));
  ASSERT_TRUE(commit_response.has_commit());
  EXPECT_EQ(commit_response.commit().status().code(), grpc::OK);

  // Finalize the session. An additional read response should be
  // produced containing the result of the finalization.
  SessionRequest finalize_request = CreateFinalizeRequest(WillowOp::FINALIZE);
  SessionResponse read_response, finalize_response;
  ASSERT_TRUE(stream->Write(finalize_request));
  ASSERT_TRUE(stream->Read(&read_response));
  ASSERT_TRUE(read_response.has_read());
  ASSERT_TRUE(read_response.read().finish_read());
  ASSERT_TRUE(stream->Read(&finalize_response));
  ASSERT_TRUE(finalize_response.has_finalize());

  // Decrypt finalized result
  FinalizedAccumulatorResult finalized_result;
  ASSERT_TRUE(finalized_result.ParseFromString(read_response.read().data()));

  auto final_result_decryptor = FinalResultDecryptor::CreateFromSerialized(
      std::move(*finalized_result.mutable_final_result_decryptor_state()));
  ASSERT_THAT(final_result_decryptor, IsOk());

  auto decryption_response = (*decryptor)
                                 ->GenerateSerializedPartialDecryptionResponse(
                                     finalized_result.decryption_request());
  ASSERT_THAT(decryption_response, IsOk());

  auto decrypted_encoded_result =
      (*final_result_decryptor)->Decrypt(std::move(*decryption_response));
  ASSERT_THAT(decrypted_encoded_result, IsOk());

  // Decode the final result
  auto decoded_result = (*encoder)->Decode(*decrypted_encoded_result);
  ASSERT_THAT(decoded_result, IsOk());

  // Verify the decoded result matches the original input.
  EXPECT_THAT(decoded_result->metric_data,
              UnorderedElementsAre(Pair("metric1", ElementsAre(20, 10, 5))));
  EXPECT_THAT(
      decoded_result->group_data,
      UnorderedElementsAre(Pair("country", ElementsAre("CA", "US", "US")),
                           Pair("lang", ElementsAre("es", "en", "es"))));
}

}  // namespace
}  // namespace confidential_federated_compute::willow
