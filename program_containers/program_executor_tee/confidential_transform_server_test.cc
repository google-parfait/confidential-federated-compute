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

#include "absl/status/status.h"
#include "fcp/base/status_converters.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/program_executor_tee_config.pb.h"
#include "fcp/testing/parse_text_proto.h"
#include "gmock/gmock.h"
#include "grpcpp/client_context.h"
#include "gtest/gtest.h"
#include "program_executor_tee/testing_base.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::program_executor_tee {

namespace {

using ::fcp::base::FromGrpcStatus;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::InitializeRequest;
using ::fcp::confidentialcompute::InitializeResponse;
using ::fcp::confidentialcompute::ProgramExecutorTeeInitializeConfig;
using ::fcp::confidentialcompute::SessionRequest;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::StreamInitializeRequest;
using ::fcp::confidentialcompute::WriteRequest;
using ::grpc::ClientWriter;
using ::grpc::StatusCode;
using ::testing::HasSubstr;

// Register the global python environment.
::testing::Environment* const python_env =
    ::testing::AddGlobalTestEnvironment(new PythonEnvironment());

// Write the InitializeRequest to the client stream and then close
// the stream, returning the status of Finish.
absl::Status WriteInitializeRequest(
    std::unique_ptr<ClientWriter<StreamInitializeRequest>> stream,
    InitializeRequest request) {
  StreamInitializeRequest stream_request;
  *stream_request.mutable_initialize_request() = std::move(request);
  if (!stream->Write(stream_request)) {
    return absl::AbortedError("Write to StreamInitialize failed.");
  }
  if (!stream->WritesDone()) {
    return absl::AbortedError("WritesDone to StreamInitialize failed.");
  }
  return FromGrpcStatus(stream->Finish());
}

TYPED_TEST_SUITE(ProgramExecutorTeeTest,
                 ::testing::Types<ProgramExecutorTeeConfidentialTransform>);

TYPED_TEST(ProgramExecutorTeeTest, InvalidStreamInitialize) {
  grpc::ClientContext context;
  InitializeResponse response;

  InitializeRequest request;
  request.set_max_num_sessions(kMaxNumSessions);

  absl::Status status = WriteInitializeRequest(
      this->stub_->StreamInitialize(&context, &response), std::move(request));
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(
      status.message(),
      HasSubstr("ProgramExecutorTeeInitializeConfig cannot be unpacked"));
}

TYPED_TEST(ProgramExecutorTeeTest,
           StreamInitializeWithKmsInvalidInitializeConfig) {
  grpc::ClientContext context;
  InitializeResponse response;

  InitializeRequest request;
  request.set_max_num_sessions(kMaxNumSessions);

  AuthorizeConfidentialTransformResponse::ProtectedResponse protected_response;
  *protected_response.add_result_encryption_keys() = "result_encryption_key";
  AuthorizeConfidentialTransformResponse::AssociatedData associated_data;
  associated_data.mutable_config_constraints()->PackFrom(
      CreateProgramExecutorTeeConfigConstraints("my_program"));
  associated_data.add_authorized_logical_pipeline_policies_hashes("hash_1");
  auto encrypted_request = this->oak_client_encryptor_
                               ->Encrypt(protected_response.SerializeAsString(),
                                         associated_data.SerializeAsString())
                               .value();
  *request.mutable_protected_response() = encrypted_request;

  absl::Status status = WriteInitializeRequest(
      this->stub_->StreamInitialize(&context, &response), std::move(request));
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(status.message(),
              HasSubstr("Cannot unpack ProgramExecutorTeeInitializeConfig"));
}

TYPED_TEST(ProgramExecutorTeeTest,
           StreamInitializeWithKmsInvalidConfigConstraints) {
  grpc::ClientContext context;
  InitializeResponse response;

  InitializeRequest request;
  request.mutable_configuration()->PackFrom(
      CreateProgramExecutorTeeInitializeConfig("my_program"));
  request.set_max_num_sessions(kMaxNumSessions);

  AuthorizeConfidentialTransformResponse::ProtectedResponse protected_response;
  *protected_response.add_result_encryption_keys() = "result_encryption_key";
  AuthorizeConfidentialTransformResponse::AssociatedData associated_data;
  associated_data.add_authorized_logical_pipeline_policies_hashes("hash_1");
  auto encrypted_request = this->oak_client_encryptor_
                               ->Encrypt(protected_response.SerializeAsString(),
                                         associated_data.SerializeAsString())
                               .value();
  *request.mutable_protected_response() = encrypted_request;

  absl::Status status = WriteInitializeRequest(
      this->stub_->StreamInitialize(&context, &response), std::move(request));
  ASSERT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(status.message(),
              HasSubstr("Cannot unpack ProgramExecutorTeeConfigConstraints"));
}

TYPED_TEST(ProgramExecutorTeeTest, StreamInitializeWithKmsMismatchingProgram) {
  grpc::ClientContext context;
  InitializeResponse response;

  InitializeRequest request;
  request.mutable_configuration()->PackFrom(
      CreateProgramExecutorTeeInitializeConfig("my_program"));
  request.set_max_num_sessions(kMaxNumSessions);

  AuthorizeConfidentialTransformResponse::ProtectedResponse protected_response;
  *protected_response.add_result_encryption_keys() = "result_encryption_key";
  AuthorizeConfidentialTransformResponse::AssociatedData associated_data;
  associated_data.mutable_config_constraints()->PackFrom(
      CreateProgramExecutorTeeConfigConstraints("mismatching_program"));
  associated_data.add_authorized_logical_pipeline_policies_hashes("hash_1");
  auto encrypted_request = this->oak_client_encryptor_
                               ->Encrypt(protected_response.SerializeAsString(),
                                         associated_data.SerializeAsString())
                               .value();
  *request.mutable_protected_response() = encrypted_request;

  absl::Status status = WriteInitializeRequest(
      this->stub_->StreamInitialize(&context, &response), std::move(request));
  ASSERT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  ASSERT_THAT(
      status.message(),
      HasSubstr(
          "Configured program must match the program specified in the policy"));
}

TYPED_TEST(ProgramExecutorTeeTest,
           StreamInitializeWithKmsMismatchingReferenceValues) {
  grpc::ClientContext context;
  InitializeResponse response;

  InitializeRequest request;
  request.mutable_configuration()->PackFrom(
      CreateProgramExecutorTeeInitializeConfig("my_program"));
  request.set_max_num_sessions(kMaxNumSessions);

  AuthorizeConfidentialTransformResponse::ProtectedResponse protected_response;
  *protected_response.add_result_encryption_keys() = "result_encryption_key";
  AuthorizeConfidentialTransformResponse::AssociatedData associated_data;
  auto config_constraints =
      CreateProgramExecutorTeeConfigConstraints("my_program");
  config_constraints.set_worker_reference_values(
      "mismatching_worker_reference_values");
  associated_data.mutable_config_constraints()->PackFrom(config_constraints);
  associated_data.add_authorized_logical_pipeline_policies_hashes("hash_1");
  auto encrypted_request = this->oak_client_encryptor_
                               ->Encrypt(protected_response.SerializeAsString(),
                                         associated_data.SerializeAsString())
                               .value();
  *request.mutable_protected_response() = encrypted_request;

  absl::Status status = WriteInitializeRequest(
      this->stub_->StreamInitialize(&context, &response), std::move(request));
  ASSERT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  ASSERT_THAT(status.message(),
              HasSubstr("Configured worker reference values must match the "
                        "ones specified in the policy"));
}

TYPED_TEST(ProgramExecutorTeeTest, ValidStreamInitializeAndConfigure) {
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
  config.set_outgoing_server_address(this->data_read_write_server_address_);

  InitializeRequest* initialize_request = request.mutable_initialize_request();
  initialize_request->set_max_num_sessions(kMaxNumSessions);
  initialize_request->mutable_configuration()->PackFrom(config);

  std::unique_ptr<::grpc::ClientWriter<StreamInitializeRequest>> writer =
      this->stub_->StreamInitialize(&context, &response);
  ASSERT_TRUE(writer->Write(request));
  ASSERT_TRUE(writer->WritesDone());
  ASSERT_TRUE(writer->Finish().ok());

  grpc::ClientContext session_context;
  SessionRequest session_request;
  SessionResponse session_response;
  session_request.mutable_configure()->set_chunk_size(1000);

  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream = this->stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(session_request));
  ASSERT_TRUE(stream->Read(&session_response));

  ASSERT_TRUE(session_response.has_configure());
  ASSERT_GT(session_response.configure().nonce().size(), 0);
}

TYPED_TEST(ProgramExecutorTeeTest,
           ValidStreamInitializeWithKmsNoReferenceValueConstraints) {
  grpc::ClientContext context;
  InitializeResponse response;

  InitializeRequest request;
  request.mutable_configuration()->PackFrom(
      CreateProgramExecutorTeeInitializeConfig(
          "my_program", /*client_ids=*/{},
          /*client_data_dir=*/"", /*outgoing_server_address=*/"",
          /*worker_reference_values_path=*/
          "program_executor_tee/testdata/test_reference_values.txtpb"));
  request.set_max_num_sessions(kMaxNumSessions);

  AuthorizeConfidentialTransformResponse::ProtectedResponse protected_response;
  *protected_response.add_result_encryption_keys() = "result_encryption_key";
  AuthorizeConfidentialTransformResponse::AssociatedData associated_data;
  associated_data.mutable_config_constraints()->PackFrom(
      CreateProgramExecutorTeeConfigConstraints("my_program"));
  associated_data.add_authorized_logical_pipeline_policies_hashes("hash_1");
  auto encrypted_request = this->oak_client_encryptor_
                               ->Encrypt(protected_response.SerializeAsString(),
                                         associated_data.SerializeAsString())
                               .value();
  *request.mutable_protected_response() = encrypted_request;

  ASSERT_TRUE(
      WriteInitializeRequest(this->stub_->StreamInitialize(&context, &response),
                             std::move(request))
          .ok());
}

TYPED_TEST(ProgramExecutorTeeTest, ValidStreamInitializeAndConfigureWithKms) {
  grpc::ClientContext context;
  InitializeResponse response;

  InitializeRequest request;
  request.mutable_configuration()->PackFrom(
      CreateProgramExecutorTeeInitializeConfig(
          "my_program", /*client_ids=*/{},
          /*client_data_dir=*/"", /*outgoing_server_address=*/"",
          /*worker_reference_values_path=*/
          "program_executor_tee/testdata/test_reference_values.txtpb"));
  request.set_max_num_sessions(kMaxNumSessions);

  AuthorizeConfidentialTransformResponse::ProtectedResponse protected_response;
  *protected_response.add_result_encryption_keys() = "result_encryption_key";
  AuthorizeConfidentialTransformResponse::AssociatedData associated_data;
  associated_data.mutable_config_constraints()->PackFrom(
      CreateProgramExecutorTeeConfigConstraints(
          "my_program", /*worker_reference_values_path=*/
          "program_executor_tee/testdata/test_reference_values.txtpb"));
  associated_data.add_authorized_logical_pipeline_policies_hashes("hash_1");
  auto encrypted_request = this->oak_client_encryptor_
                               ->Encrypt(protected_response.SerializeAsString(),
                                         associated_data.SerializeAsString())
                               .value();
  *request.mutable_protected_response() = encrypted_request;

  ASSERT_TRUE(
      WriteInitializeRequest(this->stub_->StreamInitialize(&context, &response),
                             std::move(request))
          .ok());

  grpc::ClientContext session_context;
  SessionRequest session_request;
  SessionResponse session_response;
  session_request.mutable_configure()->set_chunk_size(1000);

  std::unique_ptr<::grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream = this->stub_->Session(&session_context);
  ASSERT_TRUE(stream->Write(session_request));
  ASSERT_TRUE(stream->Read(&session_response));
  ASSERT_TRUE(session_response.has_configure());
}

class ProgramExecutorTeeConfidentialTransformSessionTest
    : public ProgramExecutorTeeSessionTest<
          ProgramExecutorTeeConfidentialTransform> {};

TEST_P(ProgramExecutorTeeConfidentialTransformSessionTest,
       SessionWriteFailsUnsupported) {
  this->CreateSession("unused program", UseKms());
  SessionRequest session_request;
  SessionResponse session_response;
  BlobMetadata metadata = PARSE_TEXT_PROTO(R"pb(
    compression_type: COMPRESSION_TYPE_NONE
    unencrypted {}
  )pb");
  WriteRequest* write_request = session_request.mutable_write();
  *write_request->mutable_first_request_metadata() = metadata;
  write_request->set_commit(true);

  ASSERT_TRUE(this->stream_->Write(session_request));
  ASSERT_FALSE(this->stream_->Read(&session_response));

  grpc::Status status = this->stream_->Finish();
  ASSERT_EQ(status.error_code(), grpc::StatusCode::UNIMPLEMENTED);
  ASSERT_THAT(
      status.error_message(),
      HasSubstr(
          "Writing to a session is not supported in program executor TEE"));
}

TEST_P(ProgramExecutorTeeConfidentialTransformSessionTest,
       ValidFinalizeSession) {
  this->CreateSession(R"(
def trusted_program(input_provider, external_service_handle):
  result = "a" + "b" + "c"
  external_service_handle.release_unencrypted(result.encode(), b"result")
  )",
                      UseKms());
  SessionRequest session_request;
  SessionResponse session_response;
  session_request.mutable_finalize();

  ASSERT_TRUE(this->stream_->Write(session_request));
  ASSERT_TRUE(this->stream_->Read(&session_response));

  auto released_data = this->fake_data_read_write_service_.GetReleasedData();
  ASSERT_EQ(released_data["result"], "abc");

  ASSERT_TRUE(session_response.has_finalize());
}

INSTANTIATE_TEST_SUITE_P(
    KmsParam, ProgramExecutorTeeConfidentialTransformSessionTest,
    ::testing::Bool(),  // Generates {false, true}
    ProgramExecutorTeeSessionTest<
        ProgramExecutorTeeConfidentialTransform>::TestNameSuffix);

}  // namespace

}  // namespace confidential_federated_compute::program_executor_tee
