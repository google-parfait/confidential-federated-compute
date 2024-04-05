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
#include "containers/test_concat/pipeline_transform_server.h"

#include <optional>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "containers/crypto.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/status_converters.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"
#include "grpcpp/server_context.h"

namespace confidential_federated_compute::test_concat {

using ::fcp::base::ToGrpcStatus;
using ::fcp::confidentialcompute::ConfigureAndAttestRequest;
using ::fcp::confidentialcompute::ConfigureAndAttestResponse;
using ::fcp::confidentialcompute::PipelineTransform;
using ::fcp::confidentialcompute::Record;
using ::fcp::confidentialcompute::TransformRequest;
using ::fcp::confidentialcompute::TransformResponse;
using ::grpc::ServerContext;

absl::Status TestConcatPipelineTransform::ConfigureAndAttest(
    const ConfigureAndAttestRequest* request,
    ConfigureAndAttestResponse* response) {
  const RecordDecryptor* record_decryptor;
  {
    absl::MutexLock l(&mutex_);
    if (record_decryptor_ != std::nullopt) {
      return absl::FailedPreconditionError(
          "ConfigureAndAttest can only be called once.");
    }
    record_decryptor_.emplace(request->configuration(), crypto_stub_);

    // Since record_decryptor_ is set once in ConfigureAndAttest and never
    // modified, and the underlying object is threadsafe, it is safe to store a
    // local pointer to it and access the object without a lock after we check
    // under the mutex that a value has been set for the std::optional wrapper.
    record_decryptor = &*record_decryptor_;
  }

  FCP_ASSIGN_OR_RETURN(*response->mutable_public_key(),
                       record_decryptor->GetPublicKey());
  return absl::OkStatus();
}

absl::Status TestConcatPipelineTransform::GenerateNonces(
    const fcp::confidentialcompute::GenerateNoncesRequest* request,
    fcp::confidentialcompute::GenerateNoncesResponse* response) {
  RecordDecryptor* record_decryptor;
  {
    absl::MutexLock l(&mutex_);
    if (record_decryptor_ == std::nullopt) {
      return absl::FailedPreconditionError(
          "ConfigureAndAttest must be called before GenerateNonces.");
    }
    // Since record_decryptor_ is set once in ConfigureAndAttest and never
    // modified, and the underlying object is threadsafe, it is safe to store a
    // local pointer to it and access the object without a lock after we check
    // under the mutex that a value has been set for the std::optional wrapper.
    record_decryptor = &*record_decryptor_;
  }
  for (int i = 0; i < request->nonces_count(); ++i) {
    FCP_ASSIGN_OR_RETURN(std::string nonce, record_decryptor->GenerateNonce());
    response->add_nonces(std::move(nonce));
  }
  return absl::OkStatus();
}

absl::Status TestConcatPipelineTransform::Transform(
    const TransformRequest* request, TransformResponse* response) {
  RecordDecryptor* record_decryptor;
  {
    absl::MutexLock l(&mutex_);
    if (record_decryptor_ == std::nullopt) {
      return absl::FailedPreconditionError(
          "ConfigureAndAttest must be called before Transform.");
    }

    // Since record_decryptor_ is set once in ConfigureAndAttest and never
    // modified, and the underlying object is threadsafe, it is safe to store a
    // local pointer to it and access the object without a lock after we check
    // under the mutex that a value has been set for the std::optional wrapper.
    record_decryptor = &*record_decryptor_;
  }

  std::string result = "";
  for (const fcp::confidentialcompute::Record& record : request->inputs()) {
    FCP_ASSIGN_OR_RETURN(std::string unencrypted_data,
                         record_decryptor->DecryptRecord(record));
    absl::StrAppend(&result, unencrypted_data);
  }

  Record* output = response->add_outputs();
  output->set_unencrypted_data(std::move(result));
  output->set_compression_type(Record::COMPRESSION_TYPE_NONE);

  return absl::OkStatus();
}

grpc::Status TestConcatPipelineTransform::ConfigureAndAttest(
    ServerContext* context, const ConfigureAndAttestRequest* request,
    ConfigureAndAttestResponse* response) {
  return ToGrpcStatus(ConfigureAndAttest(request, response));
}

grpc::Status TestConcatPipelineTransform::GenerateNonces(
    grpc::ServerContext* context,
    const fcp::confidentialcompute::GenerateNoncesRequest* request,
    fcp::confidentialcompute::GenerateNoncesResponse* response) {
  return ToGrpcStatus(GenerateNonces(request, response));
}

grpc::Status TestConcatPipelineTransform::Transform(
    ServerContext* context, const TransformRequest* request,
    TransformResponse* response) {
  return ToGrpcStatus(Transform(request, response));
}

}  // namespace confidential_federated_compute::test_concat
