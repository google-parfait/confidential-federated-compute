// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GVISOR_GVISOR_DO_FN_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GVISOR_GVISOR_DO_FN_H_

#include <memory>

#include "absl/status/statusor.h"
#include "containers/fns/fn_factory.h"
#include "google/protobuf/any.pb.h"

namespace confidential_federated_compute::gvisor {

// Configuration ID of the WriteConfigurationRequest blob that carries the
// customer rootfs layer (a tar).
inline constexpr char kRootfsLayerConfigId[] = "rootfs_layer";

// Creates the factory for the gVisor DoFn.
//
// At initialization, unpacks the rootfs layer passed as the `rootfs_layer`
// configuration blob into an OCI bundle. Each input is a Federated Compute
// checkpoint whose string tensor `data` holds records. Do() runs the bundle's
// entrypoint once per record, each time in a fresh gVisor sandbox (runsc),
// with the record as stdin. It emits one checkpoint per input, encrypted with
// the first reencryption key so that only the pipeline's next node can read
// it. The checkpoint's string tensor `data` holds the stdout of every run that
// exited with status 0 before the deadline and wrote at most 1 MiB.
//
// The container never parses the records or the outputs: what the bytes mean
// is up to the program and the stages around this one. `configuration` must
// be empty for now.
absl::StatusOr<std::unique_ptr<fns::FnFactory>> ProvideGvisorDoFnFactory(
    const google::protobuf::Any& configuration,
    const google::protobuf::Any& config_constraints,
    const fns::WriteConfigurationMap& write_configuration_map);

}  // namespace confidential_federated_compute::gvisor

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_GVISOR_GVISOR_DO_FN_H_
