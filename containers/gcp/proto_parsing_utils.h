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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GCP_PROTO_PARSING_UTILS_H
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GCP_PROTO_PARSING_UTILS_H

#include <string>

#include "absl/log/log.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"

namespace confidential_federated_compute::gcp {

// Reads a file at the given path and parses it as a textproto into the provided
// message. Crashes with LOG(FATAL) if the file cannot be read or parsed.
void ReadTextProtoOrDie(const std::string& path,
                        google::protobuf::Message* message);

}  // namespace confidential_federated_compute::gcp

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GCP_PROTO_PARSING_UTILS_H
