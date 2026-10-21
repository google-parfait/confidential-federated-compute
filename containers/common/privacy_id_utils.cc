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

#include "containers/common/privacy_id_utils.h"

#include <optional>
#include <string>

#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/construct_user_session.pb.h"
#include "google/protobuf/any.pb.h"

namespace confidential_federated_compute {

using ::fcp::confidentialcompute::PrivacyIdMetadata;
using ::fcp::confidentialcompute::ProtectedMetadata;

std::optional<std::string> GetPrivacyIdFromProtectedMetadata(
    const ProtectedMetadata& protected_metadata) {
  for (const google::protobuf::Any& entry : protected_metadata.metadata()) {
    PrivacyIdMetadata privacy_id_metadata;
    if (entry.UnpackTo(&privacy_id_metadata)) {
      return privacy_id_metadata.privacy_id();
    }
  }
  return std::nullopt;
}

}  // namespace confidential_federated_compute
