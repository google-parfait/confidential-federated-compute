// Copyright 2024 Google LLC.
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
#include "containers/fed_sql/confidential_transform_server.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "containers/fed_sql/kms_session.h"
#include "containers/fed_sql/ledger_session.h"
#include "containers/fed_sql/private_state.h"
#include "containers/session.h"
#include "containers/sql/sqlite_adapter.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/private_state.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "fcp/protos/confidentialcompute/fed_sql_container_config.pb.h"
#include "openssl/rand.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/dp_fedsql_constants.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/config_converter.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"

namespace confidential_federated_compute::fed_sql {

namespace {

using ::fcp::confidential_compute::kPrivateStateConfigId;
using ::fcp::confidentialcompute::AccessBudget;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::FedSqlContainerConfigConstraints;
using ::fcp::confidentialcompute::FedSqlContainerInitializeConfiguration;
using ::fcp::confidentialcompute::GemmaInitializeConfiguration;
using ::fcp::confidentialcompute::InferenceInitializeConfiguration;
using ::fcp::confidentialcompute::InitializeRequest;
using ::google::protobuf::Struct;
using ::tensorflow_federated::aggregation::CheckpointAggregator;
using ::tensorflow_federated::aggregation::CheckpointBuilder;
using ::tensorflow_federated::aggregation::CheckpointParser;
using ::tensorflow_federated::aggregation::DT_DOUBLE;
using ::tensorflow_federated::aggregation::Intrinsic;
using ::tensorflow_federated::aggregation::kDeltaIndex;
using ::tensorflow_federated::aggregation::kEpsilonIndex;

absl::Status ValidateFedSqlOuterDpParameters(const Intrinsic& intrinsic) {
  if (intrinsic.parameters.size() < 2) {
    return absl::InvalidArgumentError(
        "Outer DP IntrinsicConfig must have at least two "
        "parameters.");
  }
  if (intrinsic.parameters.at(kEpsilonIndex).dtype() != DT_DOUBLE ||
      intrinsic.parameters.at(kDeltaIndex).dtype() != DT_DOUBLE) {
    return absl::InvalidArgumentError(
        "Epsilon and delta parameters for outer DP IntrinsicConfig must both "
        "have type DT_DOUBLE.");
  }
  if (intrinsic.parameters.at(kEpsilonIndex).num_elements() != 1 ||
      intrinsic.parameters.at(kDeltaIndex).num_elements() != 1) {
    return absl::InvalidArgumentError(
        "Epsilon and delta parameters for outer DP IntrinsicConfig must each "
        "have exactly one value.");
  }
  return absl::OkStatus();
}

absl::Status ValidateTopLevelIntrinsics(
    const std::vector<Intrinsic>& intrinsics) {
  if (intrinsics.size() != 1) {
    return absl::InvalidArgumentError(
        "Configuration must have exactly one IntrinsicConfig.");
  }
  return absl::OkStatus();
}

std::string CreateTempFilePath(std::string directory,
                               std::string basename_prefix,
                               uint32_t basename_id) {
  // Created temp file will be in <directory>/<basename_prefix>_<basename_id>.
  return absl::StrCat(directory, "/", basename_prefix, "_",
                      std::to_string(basename_id));
}

absl::Status AppendBytesToTempFile(std::string& file_path,
                                   std::ios_base::openmode mode,
                                   const char* data,
                                   std::streamsize data_size) {
  // Write or append binary content to file depending on mode.
  std::ofstream temp_file(file_path, mode);
  if (!temp_file.is_open()) {
    return absl::DataLossError(
        absl::StrCat("Failed to open temp file for writing: ", file_path));
  }
  temp_file.write(data, data_size);
  temp_file.close();
  return absl::OkStatus();
}

}  // namespace

FedSqlConfidentialTransform::FedSqlConfidentialTransform(
    std::unique_ptr<oak::crypto::SigningKeyHandle> signing_key_handle,
    std::unique_ptr<oak::crypto::EncryptionKeyHandle> encryption_key_handle)
    : ConfidentialTransformBase(std::move(signing_key_handle),
                                std::move(encryption_key_handle)) {
  CHECK_OK(confidential_federated_compute::sql::SqliteAdapter::Initialize());
  std::string key(32, '\0');
  // Generate a random key using BoringSSL. BoringSSL documentation says
  // RAND_bytes always returns 1, so we don't check the return value.
  RAND_bytes(reinterpret_cast<unsigned char*>(key.data()), key.size());
  sensitive_values_key_ = std::move(key);
};

absl::Status FedSqlConfidentialTransform::SetAndValidateIntrinsics(
    const FedSqlContainerInitializeConfiguration& config) {
  absl::MutexLock l(&mutex_);
  if (intrinsics_ != std::nullopt) {
    return absl::FailedPreconditionError(
        "SetIntrinsics can only be called once.");
  }
  FCP_RETURN_IF_ERROR(
      CheckpointAggregator::ValidateConfig(config.agg_configuration()));

  FCP_ASSIGN_OR_RETURN(std::vector<Intrinsic> intrinsics,
                       tensorflow_federated::aggregation::ParseFromConfig(
                           config.agg_configuration()));
  FCP_RETURN_IF_ERROR(ValidateTopLevelIntrinsics(intrinsics));
  const Intrinsic& fedsql_intrinsic = intrinsics.at(0);
  if (fedsql_intrinsic.uri ==
          tensorflow_federated::aggregation::kDPGroupByUri ||
      fedsql_intrinsic.uri ==
          tensorflow_federated::aggregation::kDPTensorAggregatorBundleUri) {
    FCP_RETURN_IF_ERROR(ValidateFedSqlOuterDpParameters(fedsql_intrinsic));
  }

  intrinsics_.emplace(std::move(intrinsics));
  return absl::OkStatus();
}

absl::StatusOr<Struct> FedSqlConfidentialTransform::GetConfigConstraints(
    const FedSqlContainerInitializeConfiguration& config) {
  const std::vector<Intrinsic>* intrinsics;
  {
    absl::MutexLock l(&mutex_);
    if (intrinsics_ == std::nullopt) {
      return absl::FailedPreconditionError(
          "Intrinsics have not been initialized.");
    }
    intrinsics = &*intrinsics_;
  }

  const Intrinsic& fedsql_intrinsic = intrinsics->at(0);
  Struct config_properties;
  (*config_properties.mutable_fields())["intrinsic_uri"].set_string_value(
      fedsql_intrinsic.uri);
  if (fedsql_intrinsic.uri ==
          tensorflow_federated::aggregation::kDPGroupByUri ||
      fedsql_intrinsic.uri ==
          tensorflow_federated::aggregation::kDPTensorAggregatorBundleUri) {
    double epsilon =
        fedsql_intrinsic.parameters.at(kEpsilonIndex).CastToScalar<double>();
    double delta =
        fedsql_intrinsic.parameters.at(kDeltaIndex).CastToScalar<double>();
    (*config_properties.mutable_fields())["epsilon"].set_number_value(epsilon);
    (*config_properties.mutable_fields())["delta"].set_number_value(delta);
  }
  if (config.serialize_output_access_policy_node_id() > 0) {
    (*config_properties.mutable_fields())["serialize_dest"].set_number_value(
        config.serialize_output_access_policy_node_id());
    serialize_output_access_policy_node_id_.emplace(
        config.serialize_output_access_policy_node_id());
  }
  if (config.report_output_access_policy_node_id() > 0) {
    (*config_properties.mutable_fields())["report_dest"].set_number_value(
        config.report_output_access_policy_node_id());
    report_output_access_policy_node_id_.emplace(
        config.report_output_access_policy_node_id());
  }
  return config_properties;
}

absl::Status FedSqlConfidentialTransform::ValidateConfigConstraints(
    const FedSqlContainerConfigConstraints& config_constraints) {
  const std::vector<Intrinsic>* intrinsics;
  {
    absl::MutexLock l(&mutex_);
    if (intrinsics_ == std::nullopt) {
      return absl::FailedPreconditionError(
          "Intrinsics have not been initialized.");
    }
    intrinsics = &*intrinsics_;
  }

  const Intrinsic& fedsql_intrinsic = intrinsics->at(0);
  if (fedsql_intrinsic.uri != config_constraints.intrinsic_uri()) {
    return absl::FailedPreconditionError(
        "Invalid intrinsic URI for DP configuration.");
  }

  if (fedsql_intrinsic.uri ==
          tensorflow_federated::aggregation::kDPGroupByUri ||
      fedsql_intrinsic.uri ==
          tensorflow_federated::aggregation::kDPTensorAggregatorBundleUri) {
    double epsilon =
        fedsql_intrinsic.parameters.at(kEpsilonIndex).CastToScalar<double>();
    if (config_constraints.epsilon() != epsilon) {
      return absl::FailedPreconditionError(
          "Epsilon value does not match the expected value.");
    }

    double delta =
        fedsql_intrinsic.parameters.at(kDeltaIndex).CastToScalar<double>();
    if (config_constraints.delta() != delta) {
      return absl::FailedPreconditionError(
          "Delta value does not match the expected value.");
    }
  }
  return absl::OkStatus();
}

absl::Status
FedSqlConfidentialTransform::InitializeSessionInferenceConfiguration(
    const InferenceInitializeConfiguration& inference_init_config) {
  // Check that all data blobs passed in through WriteConfigurationRequest
  // are committed.
  for (const auto& [config_id, config_metadata] : write_configuration_map_) {
    if (!config_metadata.commit) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Data blob with configuration_id ", config_id, " is not committed."));
    }
  }

  if (inference_init_config.model_init_config_case() ==
      InferenceInitializeConfiguration::MODEL_INIT_CONFIG_NOT_SET) {
    return absl::FailedPreconditionError(
        "When FedSqlContainerInitializeConfiguration.inference_init_config is "
        "set, InferenceInitializeConfiguration.model_init_config must be set.");
  }
  if (inference_init_config.inference_config().model_config_case() ==
      fcp::confidentialcompute::InferenceConfiguration::MODEL_CONFIG_NOT_SET) {
    return absl::FailedPreconditionError(
        "When FedSqlContainerInitializeConfiguration.inference_init_config is "
        "set, InferenceConfiguration.model_config must be set.");
  }
  for (const fcp::confidentialcompute::InferenceTask& inference_task :
       inference_init_config.inference_config().inference_task()) {
    if (inference_task.inference_logic_case() ==
        fcp::confidentialcompute::InferenceTask::INFERENCE_LOGIC_NOT_SET) {
      return absl::FailedPreconditionError(
          "When FedSqlContainerInitializeConfiguration.inference_init_config "
          "is set, InferenceConfiguration.inference_task.inference_logic must "
          "be set for all inference tasks.");
    }
  }

  // Populate inference_configuration_.initialize_configuration.
  inference_configuration_.emplace();
  inference_configuration_->initialize_configuration = inference_init_config;

  switch (inference_configuration_->initialize_configuration
              .model_init_config_case()) {
    case InferenceInitializeConfiguration::kGemmaInitConfig: {
      const GemmaInitializeConfiguration& gemma_init_config =
          inference_configuration_->initialize_configuration
              .gemma_init_config();
      if (write_configuration_map_.find(
              gemma_init_config.tokenizer_configuration_id()) ==
          write_configuration_map_.end()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Expected Gemma tokenizer configuration id ",
                         gemma_init_config.tokenizer_configuration_id(),
                         " is missing in WriteConfigurationRequest."));
      }
      if (write_configuration_map_.find(
              gemma_init_config.model_weight_configuration_id()) ==
          write_configuration_map_.end()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Expected Gemma model weight configuration id ",
                         gemma_init_config.model_weight_configuration_id(),
                         " is missing in WriteConfigurationRequest."));
      }
      // Populate inference_configuration_.gemma_configuration.
      SessionGemmaConfiguration gemma_config;
      gemma_config.tokenizer_path =
          write_configuration_map_[gemma_init_config
                                       .tokenizer_configuration_id()]
              .file_path;
      gemma_config.model_weight_path =
          write_configuration_map_[gemma_init_config
                                       .model_weight_configuration_id()]
              .file_path;
      inference_configuration_->gemma_configuration = std::move(gemma_config);
      break;
    }
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unsupported model_init_config_case: ",
                       inference_configuration_->initialize_configuration
                           .model_init_config_case()));
  }
  return absl::OkStatus();
}

absl::Status FedSqlConfidentialTransform::StreamInitializeTransformWithKms(
    const ::google::protobuf::Any& configuration,
    const ::google::protobuf::Any& config_constraints,
    std::vector<std::string> reencryption_keys,
    absl::string_view reencryption_policy_hash) {
  FedSqlContainerInitializeConfiguration fed_sql_config;
  if (!configuration.UnpackTo(&fed_sql_config)) {
    return absl::InvalidArgumentError(
        "FedSqlContainerInitializeConfiguration cannot be unpacked.");
  }
  FedSqlContainerConfigConstraints fed_sql_config_constraints;
  if (!config_constraints.UnpackTo(&fed_sql_config_constraints)) {
    return absl::InvalidArgumentError(
        "FedSqlContainerConfigConstraints cannot be unpacked.");
  }
  FCP_RETURN_IF_ERROR(SetAndValidateIntrinsics(fed_sql_config));
  FCP_RETURN_IF_ERROR(ValidateConfigConstraints(fed_sql_config_constraints));

  FCP_RETURN_IF_ERROR(
      InitializePrivateState(fed_sql_config_constraints.access_budget()));

  if (fed_sql_config.has_inference_init_config()) {
    FCP_RETURN_IF_ERROR(InitializeSessionInferenceConfiguration(
        fed_sql_config.inference_init_config()));
  }
  reencryption_keys_ = std::move(reencryption_keys);
  reencryption_policy_hash_ = reencryption_policy_hash;
  return absl::OkStatus();
}

absl::StatusOr<Struct> FedSqlConfidentialTransform::StreamInitializeTransform(
    const InitializeRequest* request) {
  FedSqlContainerInitializeConfiguration config;
  if (!request->configuration().UnpackTo(&config)) {
    return absl::InvalidArgumentError(
        "FedSqlContainerInitializeConfiguration cannot be unpacked.");
  }
  FCP_RETURN_IF_ERROR(SetAndValidateIntrinsics(config));
  FCP_ASSIGN_OR_RETURN(Struct config_properties, GetConfigConstraints(config));
  if (config.has_inference_init_config()) {
    FCP_RETURN_IF_ERROR(InitializeSessionInferenceConfiguration(
        config.inference_init_config()));
  }
  return config_properties;
}

absl::Status FedSqlConfidentialTransform::ReadWriteConfigurationRequest(
    const fcp::confidentialcompute::WriteConfigurationRequest&
        write_configuration) {
  std::ios_base::openmode file_open_mode;
  // First request metadata is set for the first WriteConfigurationRequest of a
  // new data blob.
  if (write_configuration.has_first_request_metadata()) {
    // Create a new file.
    file_open_mode = std::ios::binary;
    current_configuration_id_ =
        write_configuration.first_request_metadata().configuration_id();
    if (write_configuration_map_.find(current_configuration_id_) !=
        write_configuration_map_.end()) {
      return absl::InvalidArgumentError(
          "Duplicated configuration_id found in WriteConfigurationRequest.");
    }
    // Create a new temp files. Temp files are saved as
    // /tmp/write_configuration_1, /tmp/write_configuration_2, etc. Use
    // `write_configuration_map_.size() + 1` to distinguish different temp file
    // names.
    std::string temp_file_path = CreateTempFilePath(
        "/tmp", "write_configuration", write_configuration_map_.size() + 1);

    LOG(INFO) << "Start writing bytes for configuration_id: "
              << current_configuration_id_ << " to " << temp_file_path;

    write_configuration_map_[current_configuration_id_] =
        WriteConfigurationMetadata{
            .file_path = std::move(temp_file_path),
            .total_size_bytes = static_cast<uint64_t>(
                write_configuration.first_request_metadata()
                    .total_size_bytes()),
            .commit = write_configuration.commit()};
  } else {
    // If the current write_configuration is not the first
    // WriteConfigurationRequest of a data blob, append to existing file.
    file_open_mode = std::ios::binary | std::ios::app;
  }

  auto& [current_file_path, current_total_size_bytes, commit] =
      write_configuration_map_[current_configuration_id_];
  FCP_RETURN_IF_ERROR(AppendBytesToTempFile(current_file_path, file_open_mode,
                                            write_configuration.data().data(),
                                            write_configuration.data().size()));
  // Update the commit status of the data blob in write_configuration_map_.
  commit = write_configuration.commit();

  // When it's the last WriteConfigurationRequest of a blob, check the size of
  // the file matches the expectation.
  if (commit) {
    if (std::filesystem::file_size(current_file_path) !=
        current_total_size_bytes) {
      return absl::InvalidArgumentError(
          absl::StrCat("The total size of the data blob does not match "
                       "expected size. Expecting ",
                       current_total_size_bytes, ", got ",
                       std::filesystem::file_size(current_file_path)));
    }
    LOG(INFO) << "Successfully wrote all " << current_total_size_bytes
              << " bytes to " << current_file_path;
  }

  return absl::OkStatus();
}

absl::Status FedSqlConfidentialTransform::InitializePrivateState(
    const AccessBudget& access_budget) {
  auto it = write_configuration_map_.find(kPrivateStateConfigId);
  if (it == write_configuration_map_.end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected '", kPrivateStateConfigId,
                     "' configuration id is not found."));
  }
  const auto& file_path = it->second.file_path;
  std::ifstream file(file_path);
  if (!file.is_open()) {
    return absl::DataLossError(
        absl::StrCat("Failed to open file for reading: ", file_path));
  }
  std::optional<uint32_t> num_access_times =
      access_budget.has_times() ? std::optional<uint32_t>(access_budget.times())
                                : std::nullopt;

  auto size = std::filesystem::file_size(file_path);
  if (size > 0) {
    std::string private_state(size, '\0');
    file.read(private_state.data(), size);
    private_state_ = std::make_shared<PrivateState>(std::move(private_state),
                                                    num_access_times);
    return private_state_->budget.Parse(*private_state_->initial_state);
  } else {
    private_state_ =
        std::make_shared<PrivateState>(std::nullopt, num_access_times);
    return absl::OkStatus();
  }
}

absl::StatusOr<std::unique_ptr<confidential_federated_compute::Session>>
FedSqlConfidentialTransform::CreateSession() {
  std::unique_ptr<CheckpointAggregator> aggregator;
  const std::vector<Intrinsic>* intrinsics;
  {
    absl::MutexLock l(&mutex_);
    if (intrinsics_ == std::nullopt) {
      return absl::FailedPreconditionError(
          "Initialize must be called before Session.");
    }

    // Since intrinsics_ is set once in Initialize and never
    // modified, and the underlying object is threadsafe, it is safe to store a
    // local pointer to it and access the object without a lock after we check
    // under the mutex that values have been set for the std::optional wrappers.
    intrinsics = &*intrinsics_;
  }

  FCP_ASSIGN_OR_RETURN(aggregator, CheckpointAggregator::Create(intrinsics));
  if (KmsEnabled()) {
    CHECK(reencryption_keys_.has_value())
        << "Reencryption keys must be set when KMS is enabled.";
    CHECK(reencryption_policy_hash_.has_value())
        << "Reencryption policy hash must be set when KMS is enabled.";
    return std::make_unique<KmsFedSqlSession>(
        std::move(aggregator), *intrinsics, inference_configuration_,
        sensitive_values_key_, reencryption_keys_.value(),
        reencryption_policy_hash_.value(), private_state_,
        GetOakSigningKeyHandle());
  } else {
    return std::make_unique<FedSqlSession>(
        std::move(aggregator), *intrinsics, inference_configuration_,
        serialize_output_access_policy_node_id_,
        report_output_access_policy_node_id_, sensitive_values_key_);
  }
}

absl::StatusOr<std::string> FedSqlConfidentialTransform::GetKeyId(
    const fcp::confidentialcompute::BlobMetadata& metadata) {
  // For legacy ledger or unencrypted payloads, the key_id is not used, so we
  // return an empty string.
  if (!KmsEnabled() || metadata.has_unencrypted()) {
    return "";
  }

  if (!metadata.hpke_plus_aead_data().has_kms_symmetric_key_associated_data()) {
    return absl::InvalidArgumentError(
        "kms_symmetric_key_associated_data is not present.");
  }

  // Parse the BlobHeader to get the access policy hash and key ID.
  BlobHeader blob_header;
  if (!blob_header.ParseFromString(metadata.hpke_plus_aead_data()
                                       .kms_symmetric_key_associated_data()
                                       .record_header())) {
    return absl::InvalidArgumentError(
        "kms_symmetric_key_associated_data.record_header() cannot be "
        "parsed to BlobHeader.");
  }

  // Verify that the access policy hash matches one of the authorized
  // logical pipeline policy hashes returned by KMS before returning the key ID.
  if (!GetAuthorizedLogicalPipelinePoliciesHashes().contains(
          blob_header.access_policy_sha256())) {
    return absl::InvalidArgumentError(
        "BlobHeader.access_policy_sha256 does not match any "
        "authorized_logical_pipeline_policies_hashes returned by KMS.");
  }
  return blob_header.key_id();
}

}  // namespace confidential_federated_compute::fed_sql
