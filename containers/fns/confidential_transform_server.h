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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_CONFIDENTIAL_TRANSFORM_SERVER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_CONFIDENTIAL_TRANSFORM_SERVER_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "cc/crypto/signing_key.h"
#include "containers/confidential_transform_server_base.h"
#include "containers/fns/fn_factory.h"
#include "containers/session.h"
#include "google/protobuf/any.pb.h"

namespace confidential_federated_compute::fns {
// FnFactoryProvider creates a FnFactory once when the container is
// initialized. It must validate any config constraints and pass any necessary
// state to the FnFactory. The FnFactory is used repeatedly to create Fn
// instances.
using FnFactoryProvider =
    absl::AnyInvocable<absl::StatusOr<std::unique_ptr<FnFactory>>(
        const google::protobuf::Any& configuration,
        const google::protobuf::Any& config_constraints,
        const WriteConfigurationMap& write_configuration_map)>;

// ConfidentialTransform service for customer-defined Fns.
//
// FnFactory holds any state that Fns need and creates Fns.
class FnConfidentialTransform final
    : public confidential_federated_compute::ConfidentialTransformBase {
 public:
  FnConfidentialTransform(
      std::unique_ptr<oak::crypto::SigningKeyHandle> signing_key_handle,
      FnFactoryProvider fn_factory_provider,
      std::unique_ptr<oak::crypto::EncryptionKeyHandle> encryption_key_handle)
      : confidential_federated_compute::ConfidentialTransformBase(
            std::move(signing_key_handle), std::move(encryption_key_handle)),
        fn_factory_provider_(std::move(fn_factory_provider)) {};

 private:
  absl::Status StreamInitializeTransformWithKms(
      const google::protobuf::Any& configuration,
      const google::protobuf::Any& config_constraints,
      std::vector<std::string> reencryption_keys,
      absl::string_view reencryption_policy_hash) override;

  absl::StatusOr<std::unique_ptr<confidential_federated_compute::Session>>
  CreateSession() override;

  absl::StatusOr<std::string> GetKeyId(
      const fcp::confidentialcompute::BlobMetadata& metadata) override;

  absl::Status ReadWriteConfigurationRequest(
      const fcp::confidentialcompute::WriteConfigurationRequest&
          write_configuration) override;

  absl::StatusOr<google::protobuf::Struct> StreamInitializeTransform(
      const fcp::confidentialcompute::InitializeRequest* request) override {
    return absl::FailedPreconditionError(
        "Fn container must be initialized with KMS.");
  }

  // Track the configuration ID of the current data blob passed to container
  // through `ReadWriteConfigurationRequest`.
  std::string current_configuration_id_;
  // Tracking data passed into the container through WriteConfigurationRequest.
  struct WriteConfigurationMetadata {
    std::string file_path;
    uint64_t total_size_bytes;
    bool commit;
  };
  absl::flat_hash_map<std::string, WriteConfigurationMetadata>
      write_configuration_map_;
  FnFactoryProvider fn_factory_provider_;
  absl::Mutex fn_factory_mutex_;
  std::optional<std::unique_ptr<FnFactory>> fn_factory_
      ABSL_GUARDED_BY(fn_factory_mutex_);
};

}  // namespace confidential_federated_compute::fns

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_FNS_CONFIDENTIAL_TRANSFORM_SERVER_H_
