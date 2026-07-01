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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CONSTRUCT_USER_SESSION_CONSTRUCT_USER_SESSION_FN_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CONSTRUCT_USER_SESSION_CONSTRUCT_USER_SESSION_FN_H_

#include <memory>

#include "absl/status/statusor.h"
#include "containers/fns/fn_factory.h"
#include "google/protobuf/any.pb.h"

namespace confidential_federated_compute::construct_user_session {

// Validates the `configuration` and creates a FnFactory for
// ConstructUserSessionFn instances.
absl::StatusOr<std::unique_ptr<fns::FnFactory>>
CreateConstructUserSessionFnFactoryProvider(
    const google::protobuf::Any& configuration,
    const google::protobuf::Any& config_constraints,
    const fns::WriteConfigurationMap& write_configuration_map);

}  // namespace confidential_federated_compute::construct_user_session

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_CONSTRUCT_USER_SESSION_CONSTRUCT_USER_SESSION_FN_H_
