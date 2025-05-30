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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_DATA_URI_RESOLVER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_DATA_URI_RESOLVER_H_

#include <functional>
#include <future>
#include <map>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "containers/crypto.h"
#include "fcp/protos/confidentialcompute/data_read_write.grpc.pb.h"
#include "federated_language/proto/computation.pb.h"
#include "grpcpp/channel.h"
#include "grpcpp/create_channel.h"
#include "tensorflow_federated/cc/core/impl/executors/data_backend.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::program_executor_tee {

// Implementation of DataBackend that uses a DataReadWrite service to resolve
// pointers to data.
class DataUriResolver final : public tensorflow_federated::DataBackend {
 public:
  DataUriResolver(BlobDecryptor* blob_decryptor,
                  std::function<std::string()> nonce_generator, int port)
      : blob_decryptor_(blob_decryptor),
        nonce_generator_(nonce_generator),
        stub_(fcp::confidentialcompute::outgoing::DataReadWrite::NewStub(
            grpc::CreateChannel(absl::StrCat("[::1]:", port),
                                grpc::InsecureChannelCredentials()))) {}

  absl::Status ResolveToValue(
      const federated_language::Data& data_reference,
      const federated_language::Type& data_type,
      tensorflow_federated::v0::Value& value_out) override;

 private:
  // Resolve a uri into a federated compute checkpoint by sending a ReadRequest
  // and decrypting the result returned in the ReadResponse.
  absl::StatusOr<std::string> ResolveUriToCheckpoint(absl::string_view uri);

  // Pointer to BlobDecryptor created upon instantiation of this container.
  BlobDecryptor* blob_decryptor_;

  // Function that can be used to generate a nonce for each ReadRequest. The
  // nonce allows us to check that the ReadResponse (which will be
  // cryptographically bound to the response) was produced following a new
  // interaction with the ledger. This is necessary for proper data access
  // accounting.
  std::function<std::string()> nonce_generator_;

  // DataReadWrite service stub.
  std::unique_ptr<fcp::confidentialcompute::outgoing::DataReadWrite::Stub>
      stub_;

  // Map of data uris to futures representing decrypted federated compute
  // checkpoints. If a uri exists but the corresponding future does not yet
  // have a value, this indicates that we have sent a ReadRequest and are
  // waiting for a ReadResponse. It would be inadvisable to send another
  // ReadRequest in this situation given it would be counted as a separate
  // data access attempt by the ledger.
  std::map<std::string, std::shared_future<std::string>> uri_to_value_cache_;

  // Mutex that we use to ensure that we do not send two ReadRequests for
  // the same data uri.
  absl::Mutex mutex_;
};

}  // namespace confidential_federated_compute::program_executor_tee

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_DATA_URI_RESOLVER_H_