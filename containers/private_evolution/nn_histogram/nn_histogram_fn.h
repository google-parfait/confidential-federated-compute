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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_EVOLUTION_NN_HISTOGRAM_NN_HISTOGRAM_FN_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_EVOLUTION_NN_HISTOGRAM_NN_HISTOGRAM_FN_H_

#include <memory>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "containers/fns/fn_factory.h"
#include "fcp/protos/confidentialcompute/sentence_transformers_config.pb.h"
#include "google/protobuf/any.h"
#include "utils.h"

namespace confidential_federated_compute::nn_histogram {

using ReadRecordFn = absl::AnyInvocable<absl::StatusOr<
    std::vector<fcp::confidentialcompute::Embedding>>(absl::string_view)>;
using NearestNeighborFn = absl::AnyInvocable<absl::StatusOr<int32_t>(
    absl::Span<const float>,
    const std::vector<fcp::confidentialcompute::Embedding>&)>;

constexpr absl::string_view kDataTensorName = "data";
constexpr absl::string_view kIndexTensorName = "index";

// Creates a FnFactory for NN Histogram MapFn instances.
absl::StatusOr<std::unique_ptr<fns::FnFactory>> ProvideNNHistogramFnFactory(
    const google::protobuf::Any& configuration,
    const google::protobuf::Any& config_constraints,
    const confidential_federated_compute::fns::WriteConfigurationMap&
        write_configuration_map,
    ReadRecordFn read_record_fn = ReadRecords,
    NearestNeighborFn nn_fn = FindNearestNeighbor);

}  // namespace confidential_federated_compute::nn_histogram

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_EVOLUTION_NN_HISTOGRAM_NN_HISTOGRAM_FN_H_
