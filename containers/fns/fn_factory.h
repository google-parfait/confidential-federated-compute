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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_FN_FACTORY_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_FN_FACTORY_H_

#include "absl/status/statusor.h"
#include "containers/session.h"

namespace confidential_federated_compute::fns {

// Interface for the factory that creates Fns.
class FnFactory {
 public:
  // A Fn instance is a session.
  using Fn = confidential_federated_compute::Session;

  FnFactory() = default;
  virtual ~FnFactory() = default;
  FnFactory(const FnFactory&) = delete;
  FnFactory& operator=(const FnFactory&) = delete;
  FnFactory(FnFactory&&) = delete;
  FnFactory& operator=(FnFactory&&) = delete;

  // Creates a new Fn instance.
  // This method must be thread-safe.
  virtual absl::StatusOr<std::unique_ptr<Fn>> CreateFn() const = 0;
};

}  // namespace confidential_federated_compute::fns

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_FN_FACTORY_H_
