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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_FN_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_FN_H_

#include <string>

#include "absl/status/status.h"
#include "containers/session.h"
#include "fcp/base/monitoring.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "google/protobuf/any.pb.h"

namespace confidential_federated_compute::fns {

struct Value {
  // The serialized input or output data.
  std::string data;
  // Metadata associated with the data descriving how the data was compressed
  // and encrypted.
  // TODO: consider hiding the metadata and providing it automatically.
  fcp::confidentialcompute::BlobMetadata metadata;
};

struct KeyValue {
  google::protobuf::Any key;
  Value value;
};

// Common base class for functions
class Fn : public confidential_federated_compute::Session {
  // Does any setup work needed for this Fn replica.
  //
  // Invoked exactly once on each Fn replica (one replica per chunk of
  // work) before all function invocations.
  //
  // By default, does nothing.
  virtual absl::Status InitializeReplica(google::protobuf::Any config,
                                         Context& context) {
    return absl::OkStatus();
  }

  // Does any shutdown work needed for this Fn replica.
  //
  // Invoked exactly once on each Fn replica (one replica per chunk of work)
  // after all function invocations.
  //
  // By default, does nothing.
  virtual absl::Status FinalizeReplica(google::protobuf::Any config,
                                       Context& context) {
    return absl::OkStatus();
  }

 public:
  absl::StatusOr<fcp::confidentialcompute::ConfigureResponse> Configure(
      fcp::confidentialcompute::ConfigureRequest configure_request,
      Context& context) override final {
    FCP_RETURN_IF_ERROR(
        InitializeReplica(configure_request.configuration(), context));
    return fcp::confidentialcompute::ConfigureResponse();
  }

  absl::StatusOr<fcp::confidentialcompute::FinalizeResponse> Finalize(
      fcp::confidentialcompute::FinalizeRequest request,
      fcp::confidentialcompute::BlobMetadata input_metadata,
      Context& context) override final {
    FCP_RETURN_IF_ERROR(FinalizeReplica(request.configuration(), context));
    // TODO: Add support for releasing the results (if needed).
    return fcp::confidentialcompute::FinalizeResponse();
  }
};

}  // namespace confidential_federated_compute::fns

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_FN_H_