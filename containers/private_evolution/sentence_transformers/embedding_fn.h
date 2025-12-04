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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_EVOLUTION_SENTENCE_TRANSFORMERS_EMBEDDING_FN_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_EVOLUTION_SENTENCE_TRANSFORMERS_EMBEDDING_FN_H_

#include <memory>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "containers/fns/fn_factory.h"
#include "model_delegate.h"

namespace confidential_federated_compute::sentence_transformers {

constexpr absl::string_view kDataTensorName = "data";

using EmbeddingFnFactoryProvider = absl::AnyInvocable<absl::StatusOr<
    std::unique_ptr<confidential_federated_compute::fns::FnFactory>>(
    const google::protobuf::Any& configuration,
    const google::protobuf::Any& config_constraints,
    const confidential_federated_compute::fns::WriteConfigurationMap&
        write_configuration_map)>;

EmbeddingFnFactoryProvider CreateEmbeddingFnFactoryProvider(
    ModelDelegateFactory& delegate_factory, absl::string_view tmp_dir);

}  // namespace confidential_federated_compute::sentence_transformers

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_EVOLUTION_SENTENCE_TRANSFORMERS_EMBEDDING_FN_H_
