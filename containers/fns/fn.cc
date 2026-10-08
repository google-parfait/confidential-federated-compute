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
#include "containers/fns/fn.h"

#include <optional>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "containers/common/io/any_bundle.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"

namespace confidential_federated_compute::fns {

using ::fcp::confidentialcompute::ProtectedMetadata;
using ::fcp::confidentialcompute::WriteFinishedResponse;
using ::fcp::confidentialcompute::WriteRequest;

Fn::FnContext::FnContext(
    Context& session_context,
    fcp::confidentialcompute::AssociatedMetadata associated_metadata,
    fcp::confidentialcompute::ProtectedMetadata protected_metadata)
    : session_context_(session_context),
      associated_metadata_(std::move(associated_metadata)),
      protected_metadata_(std::move(protected_metadata)) {}

bool Fn::FnContext::Emit(fcp::confidentialcompute::ReadResponse read_response) {
  return session_context_.Emit(std::move(read_response));
}

bool Fn::FnContext::EmitUnencrypted(Session::KV kv) {
  return session_context_.EmitUnencrypted(std::move(kv));
}

bool Fn::FnContext::EmitEncrypted(int reencryption_key_index, Session::KV kv) {
  // Bundle the context's protected metadata together with the data so that
  // it is encrypted along with it. Outputs without protected metadata keep
  // the legacy (unbundled) format.
  if (protected_metadata_.metadata_size() > 0) {
    kv.data = std::string(
        BundleAny(protected_metadata_, absl::Cord(std::move(kv.data))));
  }
  if (associated_metadata_.metadata_size() > 0) {
    google::protobuf::Any associated_metadata;
    associated_metadata.PackFrom(associated_metadata_);
    return session_context_.EmitEncrypted(reencryption_key_index, std::move(kv),
                                          std::move(associated_metadata));
  }
  return session_context_.EmitEncrypted(reencryption_key_index, std::move(kv));
}

bool Fn::FnContext::EmitReleasable(int reencryption_key_index, Session::KV kv,
                                   std::optional<absl::string_view> src_state,
                                   absl::string_view dst_state) {
  if (!release_token_.empty()) {
    LOG(WARNING) << "Release token can be set only once per Fn.";
    return false;
  }
  return session_context_.EmitReleasable(reencryption_key_index, std::move(kv),
                                         src_state, dst_state, release_token_);
}

absl::StatusOr<WriteFinishedResponse> Fn::Write(WriteRequest write_request,
                                                std::string unencrypted_data,
                                                Context& context) {
  // Protected metadata is only ever produced by FnContext::EmitEncrypted, so
  // it is only looked for in encrypted inputs; unencrypted inputs come from
  // the untrusted side and are always passed through unchanged.
  ProtectedMetadata protected_metadata;
  if (!write_request.first_request_metadata().has_hpke_plus_aead_data() ||
      !UnbundleAny(protected_metadata, unencrypted_data)) {
    protected_metadata.Clear();
  }
  return Write(std::move(write_request), std::move(unencrypted_data),
               std::move(protected_metadata), context);
}

void Fn::FnContext::IncrementCounter(absl::string_view name) {
  IncrementCounterBy(name, 1);
}

void Fn::FnContext::IncrementCounterBy(absl::string_view name, int64_t amount) {
  session_context_.GetCounters()[std::string(name)] += amount;
}

}  // namespace confidential_federated_compute::fns
