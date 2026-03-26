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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/cord.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "gmock/gmock.h"
#include "grpcpp/channel.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "gtest/gtest.h"
#include "testing/parse_text_proto.h"
#include "transform/willow_messages.pb.h"
#include "willow/api/client.h"
#include "willow/api/server_accumulator.h"
#include "willow/input_encoding/codec.h"
#include "willow/input_encoding/codec_factory.h"
#include "willow/proto/willow/aggregation_config.pb.h"
#include "willow/testing_utils/shell_testing_decryptor.h"

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
using ::secure_aggregation::testing::ShellTestingDecryptor;
using ::secure_aggregation::willow::AggregationConfigProto;
using ::secure_aggregation::willow::Codec;
using ::secure_aggregation::willow::CodecFactory;
using ::secure_aggregation::willow::DecodedData;
using ::secure_aggregation::willow::FinalizedAccumulatorResult;
using ::secure_aggregation::willow::FinalResultDecryptor;
using ::secure_aggregation::willow::GroupData;
using ::secure_aggregation::willow::InputSpec;
using ::secure_aggregation::willow::MetricData;
using ::secure_aggregation::willow::ShellAhePublicKey;
using ::testing::ElementsAre;
using ::testing::Pair;
using ::testing::Test;
using ::testing::UnorderedElementsAre;

AggregationConfigProto CreateAggregationConfig() {
  return PARSE_TEXT_PROTO(R"pb(
    max_number_of_decryptors: 1
    max_number_of_clients: 10
    key_id: "test"
    vector_configs {
      key: "metric1"
      value { length: 8 bound: 100 }
    }
  )pb");
}

InputSpec CreateInputSpec() {
  return PARSE_TEXT_PROTO(R"pb(
    group_by_vector_specs {
      vector_name: "country"
      data_type: STRING
      domain_spec { string_values { values: [ "CA", "GB", "MX", "US" ] } }
    }
    group_by_vector_specs {
      vector_name: "lang"
      data_type: STRING
      domain_spec { string_values { values: [ "en", "es" ] } }
    }
    metric_vector_specs {
      vector_name: "metric1"
      data_type: INT64
      domain_spec { interval { min: 0 max: 100 } }
    }
  )pb");
}

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

    // Create Aggregation Config.
    aggregation_config_ = CreateAggregationConfig();

    // Create Codec.
    auto codec = CodecFactory::CreateExplicitCodec(CreateInputSpec());
    CHECK(codec.ok());
    codec_ = std::move(*codec);

    // Initialize decryptor and generate public key.
    auto decryptor = secure_aggregation::testing::ShellTestingDecryptor::Create(
        aggregation_config_);
    CHECK(decryptor.ok());
    decryptor_ = std::move(*decryptor);

    auto public_key = decryptor_->GeneratePublicKey();
    CHECK(public_key.ok());
    public_key_ = std::move(*public_key);
  }

  ~WillowTransformServiceTest() override {
    LOG(INFO) << "Shutting down the server";
    server_->Shutdown();
    server_->Wait();
  }

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
    if (init_status.ok()) {
      LOG(INFO) << "Transform initialized";
    } else {
      LOG(ERROR) << "Transform failed to initialize: "
                 << init_status.error_message();
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
    LOG(INFO) << "Session created";
    return stream;
  }

  SessionRequest CreateWriteRequest(std::string blob_id, WillowOp::Kind op_kind,
                                    absl::Cord data) {
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

  // Creates an encoded and encrypted write request for the specified
  // keys and metrics.
  absl::StatusOr<SessionRequest> CreateContributionWriteRequest(
      std::string nonce, std::vector<std::string> countries,
      std::vector<std::string> languages, std::vector<int64_t> metrics) {
    // Create and encode input.
    GroupData group_by_data;
    group_by_data["country"] = std::move(countries);
    group_by_data["lang"] = std::move(languages);
    MetricData metric_data;
    metric_data["metric1"] = std::move(metrics);

    FCP_ASSIGN_OR_RETURN(auto encoded_data,
                         codec_->Encode(group_by_data, metric_data));

    // Generate client contribution, encrypted towards public key with
    // the provided nonce.
    FCP_ASSIGN_OR_RETURN(
        auto client_message,
        secure_aggregation::GenerateClientContribution(
            aggregation_config_, encoded_data, public_key_, nonce));

    // Create write request with the serialized message.
    return CreateWriteRequest(nonce, WillowOp::ADD_INPUT,
                              client_message.SerializeAsCord());
  }

  absl::StatusOr<DecodedData> DecryptAndDecode(absl::Cord finalized_data) {
    // Decrypt finalized result
    FinalizedAccumulatorResult finalized_result;
    EXPECT_TRUE(finalized_result.ParseFromString(finalized_data));

    FCP_ASSIGN_OR_RETURN(
        auto final_result_decryptor,
        FinalResultDecryptor::CreateFromSerialized(std::move(
            *finalized_result.mutable_final_result_decryptor_state())));

    FCP_ASSIGN_OR_RETURN(
        auto decryption_response,
        decryptor_->GenerateSerializedPartialDecryptionResponse(
            finalized_result.decryption_request()));

    FCP_ASSIGN_OR_RETURN(
        auto decrypted_encoded_result,
        final_result_decryptor->Decrypt(std::move(decryption_response)));

    // Decode the final result
    return codec_->Decode(decrypted_encoded_result);
  }

  AggregationConfigProto aggregation_config_;
  std::unique_ptr<Codec> codec_;
  std::unique_ptr<ShellTestingDecryptor> decryptor_;
  ShellAhePublicKey public_key_;

  // The following order is important to ensure proper gRPC cleanup.
  std::unique_ptr<WillowTransformService> service_;
  std::unique_ptr<Server> server_;
  std::unique_ptr<ConfidentialTransform::Stub> stub_;
};

// The simplest case with a single Willow encrypted contribution
// and immediate finalization without merging accumulators.
TEST_F(WillowTransformServiceTest, SingleInputNoMerge) {
  ASSERT_TRUE(InitializeTransform());
  grpc::ClientContext context;
  auto stream = StartSession(&context);

  // Create and write the serialized client message to the session.
  absl::StatusOr<SessionRequest> write_request = CreateContributionWriteRequest(
      "nonce", {"US", "CA", "US"}, {"en", "es", "es"}, {10, 20, 5});
  LOG(INFO) << write_request.status();
  ASSERT_THAT(write_request, IsOk());

  SessionResponse write_response;
  ASSERT_TRUE(stream->Write(*write_request));
  ASSERT_TRUE(stream->Read(&write_response));
  ASSERT_TRUE(write_response.has_write());
  EXPECT_EQ(write_response.write().status().code(), grpc::OK);

  SessionRequest commit_request = CreateCommitRequest("aaaaa", "zzzzz");
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

  // Decrypt and decoded finalized result
  auto decoded_result = DecryptAndDecode(read_response.read().data());
  ASSERT_THAT(decoded_result, IsOk());

  // Verify the decoded result matches the original input.
  EXPECT_THAT(decoded_result->metric_data,
              UnorderedElementsAre(Pair("metric1", ElementsAre(20, 10, 5))));
  EXPECT_THAT(
      decoded_result->group_data,
      UnorderedElementsAre(Pair("country", ElementsAre("CA", "US", "US")),
                           Pair("lang", ElementsAre("es", "en", "es"))));

  stream->WritesDone();
  EXPECT_TRUE(stream->Finish().ok());
}

// More complex case with 3 batches of inputs over 2 sessions and a
// merge in the 3rd session.
TEST_F(WillowTransformServiceTest, MultipleInputsWithMerge) {
  ASSERT_TRUE(InitializeTransform());

  absl::StatusOr<SessionRequest> write_request;
  SessionRequest commit_request, compact_request, merge_request,
      finalize_request;
  SessionResponse write_response, read_response, commit_response,
      compact_response, merge_response, finalize_response;

  // First session - submit two contributions in one range.
  {
    grpc::ClientContext context;
    auto stream = StartSession(&context);
    write_request = CreateContributionWriteRequest("bbbbb", {"US", "CA"},
                                                   {"en", "en"}, {10, 20});
    ASSERT_THAT(write_request, IsOk());

    ASSERT_TRUE(stream->Write(*write_request));
    ASSERT_TRUE(stream->Read(&write_response));
    ASSERT_TRUE(write_response.has_write());
    EXPECT_EQ(write_response.write().status().code(), grpc::OK);

    write_request = CreateContributionWriteRequest(
        "ccccc", {"US", "CA", "MX"}, {"es", "en", "es"}, {30, 5, 10});
    ASSERT_THAT(write_request, IsOk());

    ASSERT_TRUE(stream->Write(*write_request));
    ASSERT_TRUE(stream->Read(&write_response));
    ASSERT_TRUE(write_response.has_write());
    EXPECT_EQ(write_response.write().status().code(), grpc::OK);

    commit_request = CreateCommitRequest("aaaaa", "ddddd");
    ASSERT_TRUE(stream->Write(commit_request));
    ASSERT_TRUE(stream->Read(&commit_response));
    ASSERT_TRUE(commit_response.has_commit());
    EXPECT_EQ(commit_response.commit().status().code(), grpc::OK);

    compact_request = CreateFinalizeRequest(WillowOp::COMPACT);
    ASSERT_TRUE(stream->Write(compact_request));
    ASSERT_TRUE(stream->Read(&read_response));
    ASSERT_TRUE(read_response.has_read());
    ASSERT_TRUE(read_response.read().finish_read());
    ASSERT_TRUE(stream->Read(&compact_response));
    ASSERT_TRUE(compact_response.has_finalize());

    stream->WritesDone();
    EXPECT_TRUE(stream->Finish().ok());
  }

  // Compacted result of the first session
  std::string compacted_blob_id1 =
      read_response.read().first_response_metadata().unencrypted().blob_id();
  absl::Cord compacted_data1 = read_response.read().data();

  // Second session - submit two contributions in two separate ranges.
  {
    grpc::ClientContext context;
    auto stream = StartSession(&context);
    write_request = CreateContributionWriteRequest("kkkkk", {"GB", "MX"},
                                                   {"en", "es"}, {15, 25});
    ASSERT_THAT(write_request, IsOk());

    ASSERT_TRUE(stream->Write(*write_request));
    ASSERT_TRUE(stream->Read(&write_response));
    ASSERT_TRUE(write_response.has_write());
    EXPECT_EQ(write_response.write().status().code(), grpc::OK);

    commit_request = CreateCommitRequest("ddddd", "mmmmm");
    ASSERT_TRUE(stream->Write(commit_request));
    ASSERT_TRUE(stream->Read(&commit_response));
    ASSERT_TRUE(commit_response.has_commit());
    EXPECT_EQ(commit_response.commit().status().code(), grpc::OK);

    write_request = CreateContributionWriteRequest(
        "ooooo", {"US", "CA", "GB"}, {"en", "en", "en"}, {10, 35, 10});
    ASSERT_THAT(write_request, IsOk());

    ASSERT_TRUE(stream->Write(*write_request));
    ASSERT_TRUE(stream->Read(&write_response));
    ASSERT_TRUE(write_response.has_write());
    EXPECT_EQ(write_response.write().status().code(), grpc::OK);

    commit_request = CreateCommitRequest("mmmmm", "zzzzz");
    ASSERT_TRUE(stream->Write(commit_request));
    ASSERT_TRUE(stream->Read(&commit_response));
    ASSERT_TRUE(commit_response.has_commit());
    EXPECT_EQ(commit_response.commit().status().code(), grpc::OK);

    compact_request = CreateFinalizeRequest(WillowOp::COMPACT);
    ASSERT_TRUE(stream->Write(compact_request));
    ASSERT_TRUE(stream->Read(&read_response));
    ASSERT_TRUE(read_response.has_read());
    ASSERT_TRUE(read_response.read().finish_read());
    ASSERT_TRUE(stream->Read(&compact_response));
    ASSERT_TRUE(compact_response.has_finalize());

    stream->WritesDone();
    EXPECT_TRUE(stream->Finish().ok());
  }

  // Compacted result of the second session
  std::string compacted_blob_id2 =
      read_response.read().first_response_metadata().unencrypted().blob_id();
  absl::Cord compacted_data2 = read_response.read().data();

  // Third session - merge the two compacted data results and finalize.
  {
    grpc::ClientContext context;
    auto stream = StartSession(&context);
    merge_request = CreateWriteRequest(compacted_blob_id1, WillowOp::MERGE,
                                       compacted_data1);
    ASSERT_TRUE(stream->Write(merge_request));
    ASSERT_TRUE(stream->Read(&merge_response));
    ASSERT_TRUE(merge_response.has_write());
    EXPECT_EQ(merge_response.write().status().code(), grpc::OK);

    merge_request = CreateWriteRequest(compacted_blob_id2, WillowOp::MERGE,
                                       compacted_data2);
    ASSERT_TRUE(stream->Write(merge_request));
    ASSERT_TRUE(stream->Read(&merge_response));
    ASSERT_TRUE(merge_response.has_write());
    EXPECT_EQ(merge_response.write().status().code(), grpc::OK);

    finalize_request = CreateFinalizeRequest(WillowOp::FINALIZE);
    ASSERT_TRUE(stream->Write(finalize_request));
    ASSERT_TRUE(stream->Read(&read_response));
    ASSERT_TRUE(read_response.has_read());
    ASSERT_TRUE(read_response.read().finish_read());
    ASSERT_TRUE(stream->Read(&finalize_response));
    ASSERT_TRUE(finalize_response.has_finalize());

    stream->WritesDone();
    EXPECT_TRUE(stream->Finish().ok());
  }

  // Decrypt and decoded finalized result
  auto decoded_result = DecryptAndDecode(read_response.read().data());
  ASSERT_THAT(decoded_result, IsOk());

  // Verify the decoded result matches the expected results.
  EXPECT_THAT(
      decoded_result->metric_data,
      UnorderedElementsAre(Pair("metric1", ElementsAre(60, 25, 35, 20, 30))));
  EXPECT_THAT(decoded_result->group_data,
              UnorderedElementsAre(
                  Pair("country", ElementsAre("CA", "GB", "MX", "US", "US")),
                  Pair("lang", ElementsAre("en", "en", "es", "en", "es"))));
}

}  // namespace
}  // namespace confidential_federated_compute::willow
