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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_BIG_ENDIAN_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_BIG_ENDIAN_H_

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>

#include "absl/base/config.h"
#include "absl/numeric/int128.h"

namespace confidential_federated_compute {

uint64_t swap(uint64_t x) {
#if ABSL_HAVE_BUILTIN(__builtin_bswap64) || defined(__GNUC__)
  return __builtin_bswap64(x);
#else
  return (((x & uint64_t{0xFF}) << 56) | ((x & uint64_t{0xFF00}) << 40) |
          ((x & uint64_t{0xFF0000}) << 24) | ((x & uint64_t{0xFF000000}) << 8) |
          ((x & uint64_t{0xFF00000000}) >> 8) |
          ((x & uint64_t{0xFF0000000000}) >> 24) |
          ((x & uint64_t{0xFF000000000000}) >> 40) |
          ((x & uint64_t{0xFF00000000000000}) >> 56));
#endif
}

absl::uint128 swap(absl::uint128 x) {
  return absl::MakeUint128(swap(absl::Uint128Low64(x)),
                           swap(absl::Uint128High64(x)));
}

template <typename T>
T LoadBigEndian(const void* p, size_t size) {
  T value = {};
  std::memcpy(&value, p, std::min(sizeof(T), size));
#ifdef ABSL_IS_LITTLE_ENDIAN
  value = swap(value);
#endif
  return value;
}

template <typename T>
T LoadBigEndian(const std::string& s) {
  return LoadBigEndian<T>(s.data(), s.size());
}

template <typename T>
std::string StoreBigEndian(T value) {
#ifdef ABSL_IS_LITTLE_ENDIAN
  value = swap(value);
#endif
  std::string ret(sizeof(T), '\0');
  std::memcpy(ret.data(), &value, sizeof(T));
  return ret;
}

}  // namespace confidential_federated_compute

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_BIG_ENDIAN_H_
