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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_ANY_BUNDLE_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_ANY_BUNDLE_H_

#include <string>

#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/any.pb.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

namespace confidential_federated_compute::fed_sql {

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

// Unbundles the given Cord into the message and payload data, returning true
// if the unbundling is successful. The unbundled message is stored in the
// `result` parameter and the unbundled payload data is stored in the `data`
// parameter overriding the original bundle data.
template <typename T>
bool UnbundleAny(T& result, absl::Cord& data) {
  absl::string_view flattened = data.Flatten();
  google::protobuf::io::ArrayInputStream stream(flattened.data(),
                                                flattened.size());
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

  int pos = coded_stream.CurrentPosition();
  if (pos + payload_size != flattened.size()) {
    return false;
  }

  data.RemovePrefix(pos);
  return true;
}

}  // namespace confidential_federated_compute::fed_sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FED_SQL_ANY_BUNDLE_H_
