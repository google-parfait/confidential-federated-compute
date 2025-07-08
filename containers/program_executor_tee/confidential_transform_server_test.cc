// Copyright 2025 Google LLC.
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
#include "containers/program_executor_tee/confidential_transform_server.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "containers/crypto.h"
#include "containers/crypto_test_utils.h"
#include "containers/program_executor_tee/program_context/cc/fake_data_read_write_service.h"
#include "containers/program_executor_tee/program_context/cc/generate_checkpoint.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/data_read_write.grpc.pb.h"
#include "fcp/protos/confidentialcompute/data_read_write.pb.h"
#include "fcp/protos/confidentialcompute/program_executor_tee_config.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::program_executor_tee {

namespace {

using ::fcp::confidential_compute::NonceGenerator;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::ConfidentialTransform;
using ::fcp::confidentialcompute::InitializeRequest;
using ::fcp::confidentialcompute::InitializeResponse;
using ::fcp::confidentialcompute::ProgramExecutorTeeInitializeConfig;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::StreamInitializeRequest;
using ::fcp::confidentialcompute::WriteRequest;
using ::grpc::Server;
using ::grpc::ServerBuilder;
using ::grpc::StatusCode;
using ::testing::HasSubstr;
using ::testing::Test;

inline constexpr int kMaxNumSessions = 8;

class ProgramExecutorTeeTest : public Test {
 public:
  ProgramExecutorTeeTest() {
    const std::string localhost = "[::1]:";

    int confidential_transform_server_port;
    ServerBuilder builder;
    builder.AddListeningPort(localhost + "0", grpc::InsecureServerCredentials(),
                             &confidential_transform_server_port);
    builder.RegisterService(&service_);
    server_ = builder.BuildAndStart();
    LOG(INFO) << "ConfidentialTransform server listening on "
              << localhost + std::to_string(confidential_transform_server_port)
              << std::endl;
    stub_ = ConfidentialTransform::NewStub(grpc::CreateChannel(
        localhost + std::to_string(confidential_transform_server_port),
        grpc::InsecureChannelCredentials()));

    int data_read_write_service_port;
    ServerBuilder data_read_write_builder;
    data_read_write_builder.AddListeningPort(localhost + "0",
                                             grpc::InsecureServerCredentials(),
                                             &data_read_write_service_port);
    data_read_write_builder.RegisterService(&fake_data_read_write_service_);
    fake_data_read_write_server_ = data_read_write_builder.BuildAndStart();
    data_read_write_server_address_ =
        localhost + std::to_string(data_read_write_service_port);
    LOG(INFO) << "DataReadWrite server listening on "
              << data_read_write_server_address_ << std::endl;
  }

  ~ProgramExecutorTeeTest() override { server_->Shutdown(); }

 protected:
  ProgramExecutorTeeConfidentialTransform service_{std::make_unique<
      testing::NiceMock<crypto_test_utils::MockSigningKeyHandle>>()};
  std::unique_ptr<Server> server_;
  std::unique_ptr<ConfidentialTransform::Stub> stub_;

  std::string data_read_write_server_address_;
  FakeDataReadWriteService fake_data_read_write_service_;
  std::unique_ptr<Server> fake_data_read_write_server_;
};

TEST_F(ProgramExecutorTeeTest, InvalidStreamInitialize) {
  grpc::ClientContext context;
  InitializeResponse response;
  StreamInitializeRequest request;

  InitializeRequest* initialize_request = request.mutable_initialize_request();
  initialize_request->set_max_num_sessions(kMaxNumSessions);

  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&context, &response);
  ASSERT_TRUE(writer->Write(request));
  ASSERT_TRUE(writer->WritesDone());
  grpc::Status status = writer->Finish();
  ASSERT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(ProgramExecutorTeeTest, ValidStreamInitializeAndConfigure) {
  grpc::ClientContext context;
  InitializeResponse response;
  StreamInitializeRequest request;

  ProgramExecutorTeeInitializeConfig config;
  config.set_program("fake_program");
  config.set_outgoing_server_address(data_read_write_server_address_);

  InitializeRequest* initialize_request = request.mutable_initialize_request();
  initialize_request->set_max_num_sessions(kMaxNumSessions);
  initialize_request->mutable_configuration()->PackFrom(config);

  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      stub_->StreamInitialize(&context, &response);
  ASSERT_TRUE(writer->Write(request));
  ASSERT_TRUE(writer->WritesDone());
  ASSERT_TRUE(writer->Finish().ok());

  grpc::ClientContext session_context;
  SessionRequest session_request;
  SessionResponse session_response;
  session_request.mutable_configure()->set_chunk_size(1000);

  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream = stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(session_request));
  ASSERT_TRUE(stream->Read(&session_response));

  ASSERT_TRUE(session_response.has_configure());
  ASSERT_GT(session_response.configure().nonce().size(), 0);
}

// Note that there can only be one test in this file that exercises the
// pybind11::scoped_interpreter code in FinalizeSession, because all of the test
// cases in this file run in the same process and pybind11::scoped_interpreter
// is only allowed to be used once per process (third-party extension modules
// like numpy do not load correctly if it is used a second time). Currently the
// only test exercising the pybind11::scoped_interpreter code is
// ValidFinalizeSession.
class ProgramExecutorTeeSessionTest : public ProgramExecutorTeeTest {
 public:
  void CreateSession(std::string program,
                     std::vector<std::string> client_ids = {},
                     std::string client_data_dir = "") {
    grpc::ClientContext configure_context;
    InitializeResponse response;
    StreamInitializeRequest request;

    ProgramExecutorTeeInitializeConfig config;
    config.set_program(program);
    config.set_outgoing_server_address(data_read_write_server_address_);
    config.set_attester_id("fake_attester");
    config.set_client_data_dir(client_data_dir);
    for (const std::string& client_id : client_ids) {
      config.add_client_ids(client_id);
    }

    InitializeRequest* initialize_request =
        request.mutable_initialize_request();
    initialize_request->set_max_num_sessions(kMaxNumSessions);
    initialize_request->mutable_configuration()->PackFrom(config);

    std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
        stub_->StreamInitialize(&configure_context, &response);
    CHECK(writer->Write(request));
    CHECK(writer->WritesDone());
    CHECK(writer->Finish().ok());

    public_key_ = response.public_key();

    SessionRequest session_request;
    SessionResponse session_response;
    session_request.mutable_configure()->set_chunk_size(1000);

    stream_ = stub_->Session(&session_context_);
    CHECK(stream_->Write(session_request));
    CHECK(stream_->Read(&session_response));
    nonce_generator_ =
        std::make_unique<NonceGenerator>(session_response.configure().nonce());
  }

 protected:
  grpc::ClientContext session_context_;
  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream_;
  std::unique_ptr<NonceGenerator> nonce_generator_;
  std::string public_key_;
};

TEST_F(ProgramExecutorTeeSessionTest, SessionWriteFailsUnsupported) {
  CreateSession("unused program");
  SessionRequest session_request;
  SessionResponse session_response;
  BlobMetadata metadata = PARSE_TEXT_PROTO(R"pb(
    compression_type: COMPRESSION_TYPE_NONE
    unencrypted {}
  )pb");
  WriteRequest* write_request = session_request.mutable_write();
  *write_request->mutable_first_request_metadata() = metadata;
  write_request->set_commit(true);

  ASSERT_TRUE(stream_->Write(session_request));
  ASSERT_FALSE(stream_->Read(&session_response));

  grpc::Status status = stream_->Finish();
  ASSERT_EQ(status.error_code(), grpc::StatusCode::UNIMPLEMENTED);
  ASSERT_THAT(status.error_message(),
              HasSubstr("SessionWrite is not supported"));
}

TEST_F(ProgramExecutorTeeSessionTest, ValidFinalizeSession) {
  std::vector<std::string> client_ids = {"client1", "client2", "client3",
                                         "client4"};
  std::string client_data_dir = "data_dir";
  std::string tensor_name = "output_tensor_name";
  for (int i = 0; i < client_ids.size(); i++) {
    CHECK_OK(fake_data_read_write_service_.StorePlaintextMessage(
        client_data_dir + "/" + client_ids[i],
        BuildClientCheckpointFromInts({1 + i * 3, 2 + i * 3, 3 + i * 3},
                                      tensor_name)));
  }

  CreateSession(R"(
import federated_language
from federated_language.proto import computation_pb2
from federated_language.proto import data_type_pb2
import tensorflow_federated as tff
import tensorflow as tf
import numpy as np
from google.protobuf import any_pb2
from fcp.confidentialcompute.python import min_sep_data_source

async def trusted_program(input_provider, release_manager):

  data_source = min_sep_data_source.MinSepDataSource(
      min_sep=2,
      client_ids=input_provider.client_ids,
      client_data_directory=input_provider.client_data_directory,
      computation_type=computation_pb2.Type(
          tensor=computation_pb2.TensorType(
              dtype=data_type_pb2.DataType.DT_INT32,
              dims=[3],
          )
      ),
  )
  data_source_iterator = data_source.iterator()

  client_data_type = federated_language.FederatedType(
      federated_language.TensorType(np.int32, [3]),
      federated_language.CLIENTS
  )

  server_data_type = federated_language.FederatedType(
      federated_language.TensorType(np.int32, [3]),
      federated_language.SERVER
  )

  @tff.tensorflow.computation
  def add(x, y):
    return x + y

  @federated_language.federated_computation(server_data_type, client_data_type)
  def my_comp(server_state, client_data):
    summed_client_data = federated_language.federated_sum(client_data)
    return federated_language.federated_map(add, (server_state, summed_client_data))

  # Run four rounds, which will guarantee that each client is used exactly twice.
  server_state = [0,0,0]
  for _ in range(4):
    server_state = await my_comp(server_state, data_source_iterator.select(2))

  await release_manager.release(server_state, "result")
  )",
                client_ids, client_data_dir);

  SessionRequest session_request;
  SessionResponse session_response;
  session_request.mutable_finalize();

  ASSERT_TRUE(stream_->Write(session_request));
  ASSERT_TRUE(stream_->Read(&session_response));

  auto expected_request = fcp::confidentialcompute::outgoing::WriteRequest();
  expected_request.mutable_first_request_metadata()
      ->mutable_unencrypted()
      ->set_blob_id("result");
  expected_request.set_commit(true);

  auto write_call_args = fake_data_read_write_service_.GetWriteCallArgs();
  ASSERT_EQ(write_call_args.size(), 1);
  ASSERT_EQ(write_call_args[0].size(), 1);
  auto write_request = write_call_args[0][0];
  ASSERT_EQ(write_request.first_request_metadata().unencrypted().blob_id(),
            "result");
  ASSERT_TRUE(write_request.commit());
  tensorflow_federated::v0::Value released_value;
  released_value.ParseFromString(write_request.data());
  ASSERT_THAT(released_value.array().int32_list().value(),
              ::testing::ElementsAreArray({44, 52, 60}));

  ASSERT_TRUE(session_response.has_read());
  ASSERT_TRUE(session_response.read().finish_read());
}

}  // namespace

}  // namespace confidential_federated_compute::program_executor_tee
