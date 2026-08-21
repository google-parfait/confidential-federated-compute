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

#include "containers/common/utf8_utils.h"

#include <cstddef>

#include "absl/strings/string_view.h"
// Note: Depending on protobuf's internal third_party/utf8_range may be fragile
// as it is an internal dependency that protobuf happens to export, but it is
// the easiest option available for now without introducing major new deps.
#include "third_party/utf8_range/utf8_validity.h"

namespace confidential_federated_compute {

bool IsValidUtf8(absl::string_view s) {
  return utf8_range::IsStructurallyValid(s);
}

absl::string_view TruncateUtf8(absl::string_view s, size_t max_bytes) {
  if (s.size() <= max_bytes) {
    return s;
  }
  // Step backwards past any UTF-8 continuation bytes (10xxxxxx).
  size_t truncate_pos = max_bytes;
  while (truncate_pos > 0 &&
         (static_cast<unsigned char>(s[truncate_pos]) & 0xC0) == 0x80) {
    --truncate_pos;
  }
  return s.substr(0, truncate_pos);
}

}  // namespace confidential_federated_compute
