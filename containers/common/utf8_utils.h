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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_UTF8_UTILS_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_UTF8_UTILS_H_

#include <cstddef>

#include "absl/strings/string_view.h"

namespace confidential_federated_compute {

// Returns true if the given string is valid UTF-8 according to RFC 3629.
bool IsValidUtf8(absl::string_view s);

// Truncates `s` to at most `max_bytes` without splitting multi-byte UTF-8
// code points. Returns a subview of `s`.
//
// This will produce valid UTF-8 strings, even though it may end up truncating
// in the middle of a grapheme cluster. That's OK for our purposes, since taking
// those into account would require a more complex solution using a proper UTF
// library with UTF tables.
absl::string_view TruncateUtf8(absl::string_view s, size_t max_bytes);

}  // namespace confidential_federated_compute

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_UTF8_UTILS_H_
