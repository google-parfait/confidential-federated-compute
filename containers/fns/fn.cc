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

#include "absl/strings/string_view.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"

namespace confidential_federated_compute::fns {

namespace {

void MaybeAttachMetadata(
    Session::KV& kv,
    const fcp::confidentialcompute::AssociatedMetadata& metadata) {
  if (!kv.associated_metadata.has_value() && metadata.metadata_size() > 0) {
    kv.associated_metadata = metadata;
  }
}

}  // namespace

Fn::FnContext::FnContext(Context& session_context,
                         fcp::confidentialcompute::AssociatedMetadata metadata)
    : session_context_(session_context), metadata_(std::move(metadata)) {}

bool Fn::FnContext::Emit(fcp::confidentialcompute::ReadResponse read_response) {
  return session_context_.Emit(std::move(read_response));
}

bool Fn::FnContext::EmitUnencrypted(Session::KV kv) {
  MaybeAttachMetadata(kv, metadata_);
  return session_context_.EmitUnencrypted(std::move(kv));
}

bool Fn::FnContext::EmitEncrypted(int reencryption_key_index, Session::KV kv) {
  MaybeAttachMetadata(kv, metadata_);
  return session_context_.EmitEncrypted(reencryption_key_index, std::move(kv));
}

bool Fn::FnContext::EmitReleasable(int reencryption_key_index, Session::KV kv,
                                   std::optional<absl::string_view> src_state,
                                   absl::string_view dst_state,
                                   std::string& release_token) {
  MaybeAttachMetadata(kv, metadata_);
  return session_context_.EmitReleasable(reencryption_key_index, std::move(kv),
                                         src_state, dst_state, release_token);
}

Counters& Fn::FnContext::GetCounters() {
  return session_context_.GetCounters();
}

}  // namespace confidential_federated_compute::fns
