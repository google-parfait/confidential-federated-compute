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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_MAUVE_SCORE_MAUVE_SCORE_FN_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_MAUVE_SCORE_MAUVE_SCORE_FN_H_

#include <memory>

#include "absl/status/statusor.h"
#include "containers/fns/confidential_transform_server.h"
#include "containers/fns/fn_factory.h"

namespace confidential_federated_compute::mauve_score {

// Returns an FnFactoryProvider suitable for FnConfidentialTransform.
fns::FnFactoryProvider CreateMauveScoreFnFactoryProvider();

}  // namespace confidential_federated_compute::mauve_score

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_MAUVE_SCORE_MAUVE_SCORE_FN_H_
