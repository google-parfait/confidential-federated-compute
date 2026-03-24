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

// This file contains functions and classes for managing aggregation sessions
// of a ConfidentialTransform service.

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SESSION_STREAM_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SESSION_STREAM_H_

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "grpcpp/support/sync_stream.h"

namespace confidential_federated_compute {

// SessionStream encapsulates a gRPC stream to the host of a Trusted Container
// session and provides a higher-level API for receiving requests and sending
// responses with seamless chunking support.
class SessionStream {
 public:
  SessionStream(grpc::ServerReaderWriterInterface<
                    fcp::confidentialcompute::SessionResponse,
                    fcp::confidentialcompute::SessionRequest>* stream,
                uint32_t default_chunk_size = 1024 * 1024 /* 1MB */)
      : stream_(stream), chunk_size_(default_chunk_size) {}

  absl::StatusOr<fcp::confidentialcompute::SessionRequest> Read();
  absl::Status Write(fcp::confidentialcompute::SessionResponse response);

  uint32_t chunk_size() const { return chunk_size_; }

 private:
  absl::Status WriteImpl(
      const fcp::confidentialcompute::SessionResponse& response);

  grpc::ServerReaderWriterInterface<fcp::confidentialcompute::SessionResponse,
                                    fcp::confidentialcompute::SessionRequest>*
      stream_;
  uint32_t chunk_size_;
};

}  // namespace confidential_federated_compute

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SESSION_STREAM_H_