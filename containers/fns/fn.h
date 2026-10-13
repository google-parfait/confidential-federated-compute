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

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/string_view.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "google/protobuf/any.pb.h"

namespace confidential_federated_compute::fns {

// Common base class for functions.
//
// Fn is not intended to be subclassed directly by consumers. Instead, use one
// of the concrete base classes: DoFn, MapFn, BatchDoFn etc.
class Fn : public confidential_federated_compute::Session {
 public:
  ~Fn() override = default;

  // A per-input invocation context for Fn operations.
  //
  // FnContext captures the input's associated metadata and protected metadata
  // and automatically propagates them to emitted outputs (see Emit methods).
  //
  // Both associated metadata and protected metadata are collections of
  // arbitrary messages, each identified by a unique `type_url`.
  //
  // Associated metadata is not encrypted and is visible outside of the TEE,
  // but it is cryptographically bound to the encrypted payload as AEAD
  // associated data. Protected metadata is encrypted together with the payload
  // and is only visible inside the TEE.
  class FnContext {
   public:
    // A lightweight view over one of the context's metadata collections that
    // allows unpacking, packing and removing entries. Obtained via
    // associated_metadata() or protected_metadata(); must not outlive the
    // FnContext.
    template <typename MetadataProto>
    class MetadataView {
     public:
      // Unpacks the entry matching T's type_url into `message`. Returns true if
      // a matching entry was found and successfully unpacked. Modeled after
      // google::protobuf::Any::UnpackTo.
      template <typename T>
      bool Unpack(T* message) const {
        for (const auto& entry : metadata_.metadata()) {
          if (entry.template Is<T>()) {
            return entry.UnpackTo(message);
          }
        }
        return false;
      }

      // Packs the given message, replacing any existing entry with the same
      // type_url. Modeled after google::protobuf::Any::PackFrom.
      template <typename T>
      void Pack(const T& message) {
        for (auto& entry : *metadata_.mutable_metadata()) {
          if (entry.template Is<T>()) {
            entry.PackFrom(message);
            return;
          }
        }
        metadata_.add_metadata()->PackFrom(message);
      }

      // Removes the entry matching T's type_url, if present.
      template <typename T>
      void Remove(const T& = T{}) {
        auto& entries = *metadata_.mutable_metadata();
        entries.erase(std::remove_if(entries.begin(), entries.end(),
                                     [](const google::protobuf::Any& e) {
                                       return e.template Is<T>();
                                     }),
                      entries.end());
      }

      // Removes all entries.
      void Clear() { metadata_.clear_metadata(); }

      // Replaces all entries.
      void Set(MetadataProto metadata) { metadata_ = std::move(metadata); }

     private:
      friend class FnContext;
      explicit MetadataView(MetadataProto& metadata) : metadata_(metadata) {}

      MetadataProto& metadata_;
    };

    explicit FnContext(
        Context& session_context,
        fcp::confidentialcompute::AssociatedMetadata associated_metadata,
        fcp::confidentialcompute::ProtectedMetadata protected_metadata = {});

    MetadataView<fcp::confidentialcompute::AssociatedMetadata>
    associated_metadata() {
      return MetadataView<fcp::confidentialcompute::AssociatedMetadata>(
          associated_metadata_);
    }

    MetadataView<fcp::confidentialcompute::ProtectedMetadata>
    protected_metadata() {
      return MetadataView<fcp::confidentialcompute::ProtectedMetadata>(
          protected_metadata_);
    }

    // Emit methods (delegates to session_context, auto-attaches metadata).
    bool Emit(fcp::confidentialcompute::ReadResponse read_response);
    bool EmitUnencrypted(Session::KV kv);
    bool EmitEncrypted(int reencryption_key_index, Session::KV kv);
    bool EmitReleasable(int reencryption_key_index, Session::KV kv,
                        std::optional<absl::string_view> src_state,
                        absl::string_view dst_state);

    // Increments the named counter by 1.
    void IncrementCounter(absl::string_view name);
    // Increments the named counter by the given amount.
    void IncrementCounterBy(absl::string_view name, int64_t amount);

   private:
    friend class Fn;

    // Returns the release token saved by EmitReleasable, or empty if
    // EmitReleasable was not called.
    const std::string& GetReleaseToken() const { return release_token_; }

    Context& session_context_;
    fcp::confidentialcompute::AssociatedMetadata associated_metadata_;
    fcp::confidentialcompute::ProtectedMetadata protected_metadata_;
    std::string release_token_;
  };

  // A context for the InitializeReplica() method that intentionally hides
  // EmitReleasable. This ensures that releasable blobs can only be emitted
  // during FinalizeReplica(), not during InitializeReplica().
  //
  // All other FnContext methods (Emit, EmitUnencrypted, EmitEncrypted,
  // metadata, counters) remain accessible.
  class ConfigureContext : public FnContext {
   public:
    explicit ConfigureContext(Context& session_context)
        : FnContext(session_context,
                    fcp::confidentialcompute::AssociatedMetadata()) {}

   private:
    // Hide EmitReleasable from InitializeReplica callers.
    using FnContext::EmitReleasable;
  };

  // Does any setup work needed for this Fn replica.
  //
  // Invoked exactly once on each Fn replica (one replica per chunk of
  // work) before all function invocations.
  //
  // By default, does nothing.
  virtual absl::Status InitializeReplica(google::protobuf::Any config,
                                         ConfigureContext& context) {
    return absl::OkStatus();
  }

  // Does any shutdown work needed for this Fn replica.
  //
  // Invoked exactly once on each Fn replica (one replica per chunk of work)
  // after all function invocations.
  //
  // By default, does nothing.
  virtual absl::Status FinalizeReplica(google::protobuf::Any config,
                                       FnContext& context) {
    return absl::OkStatus();
  }

  absl::StatusOr<fcp::confidentialcompute::ConfigureResponse> Configure(
      fcp::confidentialcompute::ConfigureRequest configure_request,
      Context& context) override final {
    ConfigureContext configure_context(context);
    ABSL_RETURN_IF_ERROR(InitializeReplica(configure_request.configuration(),
                                           configure_context));
    return fcp::confidentialcompute::ConfigureResponse();
  }

  // Unbundles the protected metadata from encrypted inputs that were emitted
  // with it (see FnContext::EmitEncrypted) and forwards it to the overload
  // below. Inputs that aren't protected metadata bundles (unencrypted inputs,
  // legacy blobs, or bundles of other message types) are forwarded unchanged
  // with empty protected metadata.
  absl::StatusOr<fcp::confidentialcompute::WriteFinishedResponse> Write(
      fcp::confidentialcompute::WriteRequest write_request,
      std::string unencrypted_data, Context& context) override final;

  // Implemented by the concrete Fn base classes, which make the protected
  // metadata available via FnContext.
  virtual absl::StatusOr<fcp::confidentialcompute::WriteFinishedResponse> Write(
      fcp::confidentialcompute::WriteRequest write_request,
      std::string unencrypted_data,
      fcp::confidentialcompute::ProtectedMetadata protected_metadata,
      Context& context) = 0;

  absl::StatusOr<fcp::confidentialcompute::FinalizeResponse> Finalize(
      fcp::confidentialcompute::FinalizeRequest request,
      fcp::confidentialcompute::BlobMetadata input_metadata,
      Context& context) override final {
    FnContext fn_context(context,
                         fcp::confidentialcompute::AssociatedMetadata());
    ABSL_RETURN_IF_ERROR(FinalizeReplica(request.configuration(), fn_context));
    fcp::confidentialcompute::FinalizeResponse response;
    if (!fn_context.GetReleaseToken().empty()) {
      *response.mutable_release_token() = fn_context.GetReleaseToken();
    }
    return response;
  }

 private:
  Fn() = default;

  // Only the following derived classes may inherit from Fn.
  friend class DoFn;
  friend class MapFn;
  friend class BatchDoFn;
  friend class MockFn;  // Test-only: allows MockFn to subclass Fn.
};

}  // namespace confidential_federated_compute::fns

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_FN_H_