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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_IO_ANY_BUNDLE_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_IO_ANY_BUNDLE_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/any.pb.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

namespace confidential_federated_compute {

// This functions provide a way to bundle and unbundle a message with a
// "payload" data into a single Cord, which can be stored and transmitted.
// The combined data can then be encrypted together.
//
// The bundle format is:
// [length of message in bytes] (Varint64)
// [message serialized as Any]
// [payload size in bytes] (Varint64)
// [payload data]

// Bundles the given message and payload data into a single Cord.
template <typename T>
absl::Cord BundleAny(T message, absl::Cord data) {
  google::protobuf::Any any;
  any.PackFrom(message);
  std::string any_serialized = any.SerializeAsString();

  std::string prefix;
  {
    google::protobuf::io::StringOutputStream stream(&prefix);
    google::protobuf::io::CodedOutputStream coded_stream(&stream);
    coded_stream.WriteVarint64(any_serialized.size());
    coded_stream.WriteString(any_serialized);
    coded_stream.WriteVarint64(data.size());
  }

  absl::Cord result(std::move(prefix));
  result.Append(std::move(data));
  return result;
}

namespace internal {

// Parses the bundle header (the delimited Any followed by the payload size)
// from the front of `input` and unpacks the message into `result`. On
// success, returns true and advances `input` past the header, so that it
// holds exactly the payload. Returns false without modifying `input` if it is
// not a valid bundle of a message of type T (`result` may be modified).
template <typename T>
bool ParseDelimitedAny(T& result, absl::string_view& input) {
  google::protobuf::io::ArrayInputStream stream(input.data(), input.size());
  google::protobuf::io::CodedInputStream coded_stream(&stream);

  uint64_t any_size;
  if (!coded_stream.ReadVarint64(&any_size)) {
    return false;
  }

  std::string any_serialized;
  if (!coded_stream.ReadString(&any_serialized, any_size)) {
    return false;
  }

  google::protobuf::Any any;
  if (!any.ParseFromString(any_serialized)) {
    return false;
  }
  if (!any.UnpackTo(&result)) {
    return false;
  }

  uint64_t payload_size;
  if (!coded_stream.ReadVarint64(&payload_size)) {
    return false;
  }

  size_t pos = coded_stream.CurrentPosition();
  if (pos > input.size() || payload_size != input.size() - pos) {
    return false;
  }
  input.remove_prefix(pos);
  return true;
}

}  // namespace internal

// Unbundles the given Cord into the message and payload data, returning true
// if the unbundling is successful. The unbundled message is stored in the
// `result` parameter and the unbundled payload data is stored in the `data`
// parameter overriding the original bundle data. If unbundling fails, `data`
// is left unchanged.
template <typename T>
bool UnbundleAny(T& result, absl::Cord& data) {
  absl::string_view input = data.Flatten();
  if (!internal::ParseDelimitedAny(result, input)) {
    return false;
  }
  data.RemovePrefix(data.size() - input.size());
  return true;
}

// Same as above, but for data held in a std::string. The payload is moved to
// the front of `data` in place, so no additional copy of the payload is made.
// If unbundling fails, `data` is left unchanged.
template <typename T>
bool UnbundleAny(T& result, std::string& data) {
  absl::string_view input = data;
  if (!internal::ParseDelimitedAny(result, input)) {
    return false;
  }
  data.erase(0, data.size() - input.size());
  return true;
}

}  // namespace confidential_federated_compute

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_IO_ANY_BUNDLE_H_
