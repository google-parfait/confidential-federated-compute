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

#include "proto_parsing_utils.h"

#include <fstream>
#include <sstream>

namespace confidential_federated_compute::gcp {

void ReadTextProtoOrDie(const std::string& path,
                        google::protobuf::Message* message) {
  std::ifstream file(path);
  if (!file.is_open()) {
    LOG(FATAL) << "Failed to open proto file: " << path;
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  if (!google::protobuf::TextFormat::ParseFromString(buffer.str(), message)) {
    LOG(FATAL) << "Failed to parse textproto file: " << path;
  }
}

}  // namespace confidential_federated_compute::gcp
