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

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SQL_DATA_INGRESS_SQL_DATA_INGRESS_FN_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SQL_DATA_INGRESS_SQL_DATA_INGRESS_FN_H_

#include <memory>
#include <optional>

#include "absl/status/statusor.h"
#include "containers/fns/fn_factory.h"
#include "google/protobuf/any.h"

namespace confidential_federated_compute::sql_data_ingress {

constexpr char kOutputTensorName[] = "data";

absl::StatusOr<std::unique_ptr<fns::FnFactory>> ProvideSqlDataIngressFnFactory(
    const google::protobuf::Any& configuration,
    const google::protobuf::Any& config_constraints,
    const confidential_federated_compute::fns::WriteConfigurationMap&
        write_configuration_map);

}  // namespace confidential_federated_compute::sql_data_ingress

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SQL_DATA_INGRESS_SQL_DATA_INGRESS_FN_H_
