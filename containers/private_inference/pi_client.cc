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

#include "pi_client.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "cc/oak_session/client_session.h"
#include "cc/oak_session/config.h"
#include "cc/oak_session/server_session.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/security/credentials.h"
#include "oak_session_utils.h"
#include "proto/services/session_v1_service.grpc.pb.h"
#include "src/com/google/android/as/oss/privateinference/api/private_aratea_service.pb.h"
#include "src/com/google/android/as/oss/privateinference/library/oakutil/public_keys.pb.h"
#include "src/com/google/android/as/oss/privateinference/service/api/private_inference.pb.h"

// Forward declaration of the Rust function.
extern "C" {
::oak::session::bindings::SessionConfigBuilder*
update_peer_unidirectional_session_config(
    ::oak::session::bindings::SessionConfigBuilder* builder,
    const char* tink_serialized_public_keyset_data,
    int tink_serialized_public_keyset_len);
}

ABSL_FLAG(std::string, feature_name, "FEATURE_NAME_PSI_MEMORY_GENERATION",
          "Feature name for the Pi Server");

constexpr absl::string_view kProdVerificationKeysPath =
    "private_inference/public_keys_prod_proto.binarypb";

namespace confidential_federated_compute::private_inference {

namespace {

using ::com::google::android::as::oss::privateinference::library::oakutil::
    VerificationKeys;
using ::com::google::android::as::oss::privateinference::service::api::
    PcsPrivateInferenceFeatureName;
using ::mdi::privatearatea::PcsGenerateContentRequest;
using ::mdi::privatearatea::PcsPrivateArateaRequest;
using ::mdi::privatearatea::PcsPrivateArateaResponse;
using ::oak::session::ClientSession;
using ::oak::session::v1::SessionRequest;
using ::oak::session::v1::SessionResponse;

class PiClientImpl : public PiClient {
 public:
  explicit PiClientImpl(std::string server_address,
                        PcsPrivateInferenceFeatureName feature_name)
      : server_address_(std::move(server_address)),
        feature_name_(feature_name) {}

  absl::Status Initialize() {
    // 1. Establish channel
    LOG(INFO) << "Creating channel to PI server: " << server_address_;
    channel_ = grpc::CreateChannel(server_address_,
                                   grpc::InsecureChannelCredentials());
    stub_ = oak::services::OakSessionV1Service::NewStub(channel_);

    stream_ = stub_->Stream(&context_);

    // 2. Configure session (Attestation)
    VerificationKeys verification_keys;
    std::ifstream input(std::string(kProdVerificationKeysPath),
                        std::ios::binary);
    if (!input) {
      return absl::InternalError(absl::StrCat(
          "PiClientImpl::Initialize: Failed to open verification keys file: ",
          kProdVerificationKeysPath));
    }
    if (!verification_keys.ParseFromIstream(&input)) {
      return absl::InternalError(
          "PiClientImpl::Initialize: Failed to parse verification keys.");
    }
    auto builder = ::oak::session::SessionConfigBuilder(
        oak::session::AttestationType::kPeerUnidirectional,
        oak::session::HandshakeType::kNoiseNN);

    auto status = builder.UpdateRaw(
        [keys = verification_keys.tink_serialized_public_keyset()](
            ::oak::session::SessionConfigBuilderHolder raw_builder) {
          auto new_builder = update_peer_unidirectional_session_config(
              raw_builder.release(), keys.data(), keys.length());
          return ::oak::session::SessionConfigBuilderHolder(new_builder);
        });

    if (!status.ok()) {
      return absl::Status(
          status.code(),
          absl::StrCat(
              "PiClientImpl::Initialize: Failed to build session config: ",
              status.message()));
    }
    LOG(INFO) << "Session config built.";
    auto session_config = builder.Build();
    auto session_or = ::oak::session::ClientSession::Create(session_config);
    if (!session_or.ok()) {
      return absl::Status(
          session_or.status().code(),
          absl::StrCat(
              "PiClientImpl::Initialize: ClientSession::Create failed: ",
              session_or.status().message()));
    }
    session_ = std::move(session_or.value());

    auto handshake_status =
        ExchangeHandshakeMessages(session_.get(), stream_.get());
    if (!handshake_status.ok()) {
      return absl::Status(
          handshake_status.code(),
          absl::StrCat(
              "PiClientImpl::Initialize: ExchangeHandshakeMessages failed: ",
              handshake_status.message()));
    }
    LOG(INFO) << "Handshake completed.";
    return absl::OkStatus();
  }

  absl::StatusOr<std::string> Generate(const std::string& prompt) override {
    // Session is already established and handshake completed during
    // initialization.

    // 3. Send PcsGenerateContentRequest wrapped in a
    // PcsPrivateArateaRequest.
    PcsPrivateArateaRequest request;
    request.set_feature_name(feature_name_);

    PcsGenerateContentRequest generate_request;
    generate_request.add_contents()->add_parts()->set_text(prompt);
    *request.mutable_generate_content_request() = generate_request;

    auto write_status = session_->Write(request.SerializeAsString());
    if (!write_status.ok()) {
      return absl::Status(
          write_status.code(),
          absl::StrCat("PiClientImpl::Generate: session::Write failed: ",
                       write_status.message()));
    }

    auto pump_status = PumpOutgoingMessages(session_.get(), stream_.get());
    if (!pump_status.ok()) {
      return absl::Status(
          pump_status.status().code(),
          absl::StrCat("PiClientImpl::Generate: PumpOutgoingMessages failed: ",
                       pump_status.status().message()));
    }

    while (true) {
      oak::session::v1::SessionResponse response;
      if (!stream_->Read(&response)) {
        return absl::InternalError(
            "PiClientImpl::Generate: Server closed stream while waiting for "
            "application reply.");
      }

      absl::Status put_status = session_->PutIncomingMessage(response);
      if (!put_status.ok()) {
        return absl::InternalError(
            absl::StrCat("PiClientImpl::Generate: PutIncomingMessage failed:  ",
                         put_status.ToString()));
      }

      auto decrypted_message = session_->ReadToRustBytes();
      if (!decrypted_message.ok()) {
        return absl::InternalError(absl::StrCat(
            "PiClientImpl::Generate: Failed to read from session: ",
            decrypted_message.status().ToString()));
      }

      if (decrypted_message->has_value()) {
        std::string payload =
            static_cast<std::string>(decrypted_message->value());

        PcsPrivateArateaResponse response;
        if (!response.ParseFromString(payload)) {
          return absl::InternalError(
              "PiClientImpl::Generate: Failed to parse "
              "PcsPrivateArateaResponse");
        }

        if (response.has_generate_content_response()) {
          const auto& generate_content_response =
              response.generate_content_response();
          if (generate_content_response.candidates_size() > 0) {
            const auto& candidate = generate_content_response.candidates(0);
            if (candidate.finish_reason() !=
                ::mdi::privatearatea::PcsCandidate::STOP) {
              LOG(WARNING)
                  << "Finish reason is not STOP: "
                  << ::mdi::privatearatea::PcsCandidate::FinishReason_Name(
                         candidate.finish_reason());
            }
            if (candidate.content().parts_size() > 0) {
              std::string result;
              for (const auto& part : candidate.content().parts()) {
                absl::StrAppend(&result, part.text());
              }
              return result;
            }
          }
        }

        return absl::InternalError(
            "PiClientImpl::Generate: No text found in GenerateContentResponse");
      }
      auto pump_status2 = PumpOutgoingMessages(session_.get(), stream_.get());
      if (!pump_status2.ok()) {
        return absl::Status(
            pump_status2.status().code(),
            absl::StrCat("PiClientImpl::Generate: PumpOutgoingMessages (read "
                         "loop) failed: ",
                         pump_status2.status().message()));
      }
    }
  }

 private:
  std::string server_address_;
  PcsPrivateInferenceFeatureName feature_name_;
  grpc::ClientContext context_;
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<oak::services::OakSessionV1Service::Stub> stub_;
  std::unique_ptr<grpc::ClientReaderWriter<SessionRequest, SessionResponse>>
      stream_;
  std::unique_ptr<ClientSession> session_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<PiClient>> CreatePiClient(
    std::string server_address) {
  std::string feature_name_str = absl::GetFlag(FLAGS_feature_name);
  PcsPrivateInferenceFeatureName feature_name;
  if (!::com::google::android::as::oss::privateinference::service::api::
          PcsPrivateInferenceFeatureName_Parse(feature_name_str,
                                               &feature_name)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse feature name: ", feature_name_str));
  }

  auto client =
      std::make_unique<PiClientImpl>(std::move(server_address), feature_name);
  absl::Status status = client->Initialize();
  if (!status.ok()) {
    return status;
  }
  return client;
}

}  // namespace confidential_federated_compute::private_inference
