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
#include <memory>

#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/program_executor_tee_config.pb.h"
#include "fcp/testing/parse_text_proto.h"
#include "gmock/gmock.h"
#include "grpcpp/client_context.h"
#include "program_executor_tee/testing_base.h"

namespace confidential_federated_compute::program_executor_tee {

namespace {

using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::InitializeRequest;
using ::fcp::confidentialcompute::InitializeResponse;
using ::fcp::confidentialcompute::ProgramExecutorTeeInitializeConfig;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::StreamInitializeRequest;
using ::fcp::confidentialcompute::WriteRequest;
using ::grpc::StatusCode;
using ::testing::HasSubstr;

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
  *config.mutable_reference_values() = PARSE_TEXT_PROTO(R"pb(
    oak_containers {
      root_layer {
        amd_sev { stage0 { skip {} } }
        insecure {}
      }
    }
  )pb");
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

}  // namespace

}  // namespace confidential_federated_compute::program_executor_tee
