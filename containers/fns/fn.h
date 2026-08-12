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

#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/string_view.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "google/protobuf/any.pb.h"

namespace confidential_federated_compute::fns {

// Common base class for functions
class Fn : public confidential_federated_compute::Session {
 public:
  // A per-input invocation context for Fn operations.
  //
  // FnContext captures the input's metadata and automatically propagates it to
  // emitted outputs.
  class FnContext {
   public:
    explicit FnContext(Context& session_context,
                       fcp::confidentialcompute::AssociatedMetadata metadata);

    // Unpacks the metadata entry matching T's type_url into `message`.
    // Returns true if a matching entry was found and successfully unpacked.
    // Modeled after google::protobuf::Any::UnpackTo.
    template <typename T>
    bool UnpackMetadata(T* message) const {
      for (const auto& entry : metadata_.metadata()) {
        if (entry.Is<T>()) {
          return entry.UnpackTo(message);
        }
      }
      return false;
    }

    // Packs the given message into the metadata, replacing any existing entry
    // with the same type_url. Modeled after google::protobuf::Any::PackFrom.
    template <typename T>
    void PackMetadata(const T& message) {
      for (auto& entry : *metadata_.mutable_metadata()) {
        if (entry.Is<T>()) {
          entry.PackFrom(message);
          return;
        }
      }
      // Not found — add new entry.
      metadata_.add_metadata()->PackFrom(message);
    }

    // Removes the metadata entry matching T's type_url, if present.
    template <typename T>
    void RemoveMetadata(const T& = T{}) {
      auto* entries = metadata_.mutable_metadata();
      entries->erase(std::remove_if(entries->begin(), entries->end(),
                                    [](const google::protobuf::Any& e) {
                                      return e.Is<T>();
                                    }),
                     entries->end());
    }

    // Removes all metadata entries.
    void ClearMetadata() { metadata_.clear_metadata(); }

    // Emit methods (delegates to session_context, auto-attaches metadata).
    bool Emit(fcp::confidentialcompute::ReadResponse read_response);
    bool EmitUnencrypted(Session::KV kv);
    bool EmitEncrypted(int reencryption_key_index, Session::KV kv);
    bool EmitReleasable(int reencryption_key_index, Session::KV kv,
                        std::optional<absl::string_view> src_state,
                        absl::string_view dst_state,
                        std::string& release_token);
    Counters& GetCounters();

   private:
    Context& session_context_;
    fcp::confidentialcompute::AssociatedMetadata metadata_;
  };

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

  absl::StatusOr<fcp::confidentialcompute::ConfigureResponse> Configure(
      fcp::confidentialcompute::ConfigureRequest configure_request,
      Context& context) override final {
    ABSL_RETURN_IF_ERROR(
        InitializeReplica(configure_request.configuration(), context));
    return fcp::confidentialcompute::ConfigureResponse();
  }

  absl::StatusOr<fcp::confidentialcompute::FinalizeResponse> Finalize(
      fcp::confidentialcompute::FinalizeRequest request,
      fcp::confidentialcompute::BlobMetadata input_metadata,
      Context& context) override final {
    ABSL_RETURN_IF_ERROR(FinalizeReplica(request.configuration(), context));
    // TODO: Add support for releasing the results (if needed).
    return fcp::confidentialcompute::FinalizeResponse();
  }
};

}  // namespace confidential_federated_compute::fns

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_FN_H_