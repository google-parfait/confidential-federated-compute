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
#include "fcp/confidentialcompute/private_state.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/program_executor_tee_config.pb.h"
#include "fcp/testing/parse_text_proto.h"
#include "gmock/gmock.h"
#include "grpcpp/client_context.h"
#include "gtest/gtest.h"
#include "program_executor_tee/program_context/cc/generate_checkpoint.h"
#include "program_executor_tee/testing_base.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::program_executor_tee {

namespace {

using ::fcp::base::FromGrpcStatus;
using ::fcp::confidential_compute::kPrivateStateConfigId;
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

bool WritePipelinePrivateState(ClientWriter<StreamInitializeRequest>* stream,
                               const std::string& state) {
  StreamInitializeRequest stream_request;
  auto* write_configuration = stream_request.mutable_write_configuration();
  write_configuration->set_commit(true);
  write_configuration->set_data(state);
  auto* metadata = write_configuration->mutable_first_request_metadata();
  metadata->set_configuration_id(kPrivateStateConfigId);
  metadata->set_total_size_bytes(state.size());
  return stream->Write(stream_request);
}

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

  auto writer = this->stub_->StreamInitialize(&context, &response);
  EXPECT_TRUE(WritePipelinePrivateState(writer.get(), /*state=*/""));
  absl::Status status =
      WriteInitializeRequest(std::move(writer), std::move(request));
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

  auto writer = this->stub_->StreamInitialize(&context, &response);
  EXPECT_TRUE(WritePipelinePrivateState(writer.get(), /*state=*/""));
  absl::Status status =
      WriteInitializeRequest(std::move(writer), std::move(request));
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

  auto writer = this->stub_->StreamInitialize(&context, &response);
  EXPECT_TRUE(WritePipelinePrivateState(writer.get(), /*state=*/""));
  absl::Status status =
      WriteInitializeRequest(std::move(writer), std::move(request));
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

  auto writer = this->stub_->StreamInitialize(&context, &response);
  EXPECT_TRUE(WritePipelinePrivateState(writer.get(), /*state=*/""));
  absl::Status status =
      WriteInitializeRequest(std::move(writer), std::move(request));
  ASSERT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  ASSERT_THAT(status.message(),
              HasSubstr("Configured worker reference values must match the "
                        "ones specified in the policy"));
}

TYPED_TEST(ProgramExecutorTeeTest, StreamInitializeWithKmsExhaustedBudget) {
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
      CreateProgramExecutorTeeConfigConstraints("my_program"));
  associated_data.add_authorized_logical_pipeline_policies_hashes("hash_1");
  auto encrypted_request = this->oak_client_encryptor_
                               ->Encrypt(protected_response.SerializeAsString(),
                                         associated_data.SerializeAsString())
                               .value();
  *request.mutable_protected_response() = encrypted_request;

  BudgetState initial_budget;
  initial_budget.set_num_runs_remaining(0);
  initial_budget.set_counter(100);
  auto writer = this->stub_->StreamInitialize(&context, &response);
  EXPECT_TRUE(WritePipelinePrivateState(
      writer.get(), /*state=*/initial_budget.SerializeAsString()));
  absl::Status status =
      WriteInitializeRequest(std::move(writer), std::move(request));
  ASSERT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  ASSERT_THAT(status.message(), HasSubstr("No budget remaining"));
}

TYPED_TEST(ProgramExecutorTeeTest, StreamInitializeWithKmsInvalidBudget) {
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
      CreateProgramExecutorTeeConfigConstraints("my_program"));
  associated_data.add_authorized_logical_pipeline_policies_hashes("hash_1");
  auto encrypted_request = this->oak_client_encryptor_
                               ->Encrypt(protected_response.SerializeAsString(),
                                         associated_data.SerializeAsString())
                               .value();
  *request.mutable_protected_response() = encrypted_request;

  auto writer = this->stub_->StreamInitialize(&context, &response);
  EXPECT_TRUE(WritePipelinePrivateState(writer.get(),
                                        /*state=*/"invalid_budget_proto"));
  absl::Status status =
      WriteInitializeRequest(std::move(writer), std::move(request));
  ASSERT_EQ(status.code(), absl::StatusCode::kInternal);
  ASSERT_THAT(status.message(),
              HasSubstr("Failed to parse initial budget state"));
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

  auto writer = this->stub_->StreamInitialize(&context, &response);
  EXPECT_TRUE(WritePipelinePrivateState(writer.get(), /*state=*/""));
  ASSERT_TRUE(
      WriteInitializeRequest(std::move(writer), std::move(request)).ok());
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

  auto writer = this->stub_->StreamInitialize(&context, &response);
  EXPECT_TRUE(WritePipelinePrivateState(writer.get(), /*state=*/""));
  ASSERT_TRUE(
      WriteInitializeRequest(std::move(writer), std::move(request)).ok());

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
  this->CreateSession("unused program");
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
       ValidFinalizeSessionWithoutStartingState) {
  this->CreateSession(R"(
def trusted_program(input_provider, external_service_handle):
  result_1 = "a" + "b" + "c"
  result_2 = "d" + "e" + "f"
  external_service_handle.release_unencrypted(result_1.encode(), b"result_1")
  external_service_handle.release_unencrypted(result_2.encode(), b"result_2")
  )");
  SessionRequest session_request;
  SessionResponse session_response;
  session_request.mutable_finalize();

  ASSERT_TRUE(this->stream_->Write(session_request));
  ASSERT_TRUE(this->stream_->Read(&session_response));

  auto released_data = this->fake_data_read_write_service_.GetReleasedData();
  ASSERT_EQ(released_data["result_1"], "abc");
  ASSERT_EQ(released_data["result_2"], "def");

  // State changes should only be checked in the KMS case.
  if (UseKms()) {
    auto released_state_changes =
        this->fake_data_read_write_service_.GetReleasedStateChanges();
    // There is no initial state.
    ASSERT_FALSE(released_state_changes["result_1"].first.value().has_value());
    // The first release operation triggers a state change that should decrease
    // the number of remaining runs and increment the counter.
    BudgetState expected_first_release_budget;
    expected_first_release_budget.set_num_runs_remaining(kMaxNumRuns - 1);
    expected_first_release_budget.set_counter(1);
    ASSERT_EQ(released_state_changes["result_1"].second.value(),
              expected_first_release_budget.SerializeAsString());
    ASSERT_EQ(released_state_changes["result_2"].first.value().value(),
              expected_first_release_budget.SerializeAsString());
    // The second release operations triggers a state change that should keep
    // the number of remaining runs the same and further increment the counter.
    BudgetState expected_second_release_budget;
    expected_second_release_budget.set_num_runs_remaining(kMaxNumRuns - 1);
    expected_second_release_budget.set_counter(2);
    ASSERT_EQ(released_state_changes["result_2"].second.value(),
              expected_second_release_budget.SerializeAsString());
  }

  ASSERT_TRUE(session_response.has_finalize());
}

TEST_P(ProgramExecutorTeeConfidentialTransformSessionTest,
       ValidFinalizeSessionWithStartingState) {
  int initial_num_runs = 3;
  int initial_counter = 10;
  BudgetState initial_budget;
  initial_budget.set_num_runs_remaining(initial_num_runs);
  initial_budget.set_counter(initial_counter);

  this->CreateSession(R"(
def trusted_program(input_provider, external_service_handle):
  result_1 = "a" + "b" + "c"
  result_2 = "d" + "e" + "f"
  external_service_handle.release_unencrypted(result_1.encode(), b"result_1")
  external_service_handle.release_unencrypted(result_2.encode(), b"result_2")
  )",
                      initial_budget.SerializeAsString());
  SessionRequest session_request;
  SessionResponse session_response;
  session_request.mutable_finalize();

  ASSERT_TRUE(this->stream_->Write(session_request));
  ASSERT_TRUE(this->stream_->Read(&session_response));

  auto released_data = this->fake_data_read_write_service_.GetReleasedData();
  ASSERT_EQ(released_data["result_1"], "abc");
  ASSERT_EQ(released_data["result_2"], "def");

  // State changes should only be checked in the KMS case.
  if (UseKms()) {
    auto released_state_changes =
        this->fake_data_read_write_service_.GetReleasedStateChanges();
    ASSERT_EQ(released_state_changes["result_1"].first.value().value(),
              initial_budget.SerializeAsString());
    // The first release operation triggers a state change that should decrease
    // the number of remaining runs and increment the counter.
    BudgetState expected_first_release_budget;
    expected_first_release_budget.set_num_runs_remaining(initial_num_runs - 1);
    expected_first_release_budget.set_counter(initial_counter + 1);
    ASSERT_EQ(released_state_changes["result_1"].second.value(),
              expected_first_release_budget.SerializeAsString());
    ASSERT_EQ(released_state_changes["result_2"].first.value().value(),
              expected_first_release_budget.SerializeAsString());
    // The second release operations triggers a state change that should keep
    // the number of remaining runs the same and further increment the counter.
    BudgetState expected_second_release_budget;
    expected_second_release_budget.set_num_runs_remaining(initial_num_runs - 1);
    expected_second_release_budget.set_counter(initial_counter + 2);
    ASSERT_EQ(released_state_changes["result_2"].second.value(),
              expected_second_release_budget.SerializeAsString());
  }

  ASSERT_TRUE(session_response.has_finalize());
}

TEST_P(ProgramExecutorTeeConfidentialTransformSessionTest,
       ValidFinalizeSessionWithInputs) {
  using InputData = std::tuple<std::string, std::string>;
  InputData input_1{"client_1",
                    BuildClientCheckpointFromInts({10, 20}, "my_key_name")};
  InputData input_2{"client_2",
                    BuildClientCheckpointFromInts({30, 40}, "my_key_name")};

  for (auto [client_id, data] : {input_1, input_2}) {
    if (UseKms()) {
      CHECK_OK(this->fake_data_read_write_service_.StoreEncryptedMessageForKms(
          client_id, data));
    } else {
      CHECK_OK(this->fake_data_read_write_service_.StorePlaintextMessage(
          client_id, data));
    }
  }

  this->CreateSession(R"(
import struct

# The ints are packed in little endian format into the tensor.
FORMAT_STRING = '<i' # '<' for little-endian, 'i' for 32-bit signed integer
BYTE_COUNT = 4

def trusted_program(input_provider, external_service_handle):
  sum = 0
  for client_id in input_provider.client_ids:
    tensor = input_provider.resolve_uri_to_tensor(client_id, "my_key_name")

    for unpacked_tuple in struct.iter_unpack(FORMAT_STRING, tensor.content):
      # The tuple will contain a single element (the integer)
      sum += unpacked_tuple[0]

  external_service_handle.release_unencrypted(struct.pack(FORMAT_STRING, sum), b"result")
  )",
                      /*kms_private_state=*/"",
                      /*client_ids=*/{"client_1", "client_2"});

  SessionRequest session_request;
  SessionResponse session_response;
  session_request.mutable_finalize();

  ASSERT_TRUE(this->stream_->Write(session_request));
  ASSERT_TRUE(this->stream_->Read(&session_response));

  auto released_data = this->fake_data_read_write_service_.GetReleasedData();

  int32_t result;
  std::memcpy(&result, released_data["result"].data(), sizeof(int32_t));
  ASSERT_EQ(result, 100);
}

INSTANTIATE_TEST_SUITE_P(
    KmsParam, ProgramExecutorTeeConfidentialTransformSessionTest,
    ::testing::Bool(),  // Generates {false, true}
    ProgramExecutorTeeSessionTest<
        ProgramExecutorTeeConfidentialTransform>::TestNameSuffix);

}  // namespace

}  // namespace confidential_federated_compute::program_executor_tee
