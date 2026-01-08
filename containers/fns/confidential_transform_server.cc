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
#include "containers/fns/confidential_transform_server.h"

#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "containers/confidential_transform_server_base.h"
#include "containers/fns/fn_factory.h"
#include "containers/session.h"
#include "fcp/base/monitoring.h"

namespace confidential_federated_compute::fns {
namespace {

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

}  // anonymous namespace

absl::Status FnConfidentialTransform::StreamInitializeTransform(
    const google::protobuf::Any& configuration,
    const google::protobuf::Any& config_constraints) {
  absl::WriterMutexLock l(&fn_factory_mutex_);
  if (fn_factory_.has_value()) {
    return absl::FailedPreconditionError("Fn container already initialized.");
  }
  absl::flat_hash_map<std::string, std::string> write_configuration_map;
  for (auto& [key, value] : write_configuration_map_) {
    if (value.commit) {
      write_configuration_map[key] = value.file_path;
    } else {
      return absl::InvalidArgumentError(
          "Malformed configuration file is found.");
    }
  }
  FCP_ASSIGN_OR_RETURN(std::unique_ptr<FnFactory> fn_factory,
                       fn_factory_provider_(configuration, config_constraints,
                                            write_configuration_map));
  fn_factory_.emplace(std::move(fn_factory));
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<confidential_federated_compute::Session>>
FnConfidentialTransform::CreateSession() {
  absl::ReaderMutexLock l(&fn_factory_mutex_);
  if (!fn_factory_.has_value()) {
    return absl::FailedPreconditionError(
        "Fn container must be initialized before creating Fns.");
  }
  return (*fn_factory_)->CreateFn();
}

absl::StatusOr<std::string> FnConfidentialTransform::GetKeyId(
    const fcp::confidentialcompute::BlobMetadata& metadata) {
  if (!metadata.hpke_plus_aead_data().has_kms_symmetric_key_associated_data()) {
    return absl::InvalidArgumentError(
        "kms_symmetric_key_associated_data is not present.");
  }

  // Parse the BlobHeader to get the access policy hash and key ID.
  fcp::confidentialcompute::BlobHeader blob_header;
  if (!blob_header.ParseFromString(metadata.hpke_plus_aead_data()
                                       .kms_symmetric_key_associated_data()
                                       .record_header())) {
    return absl::InvalidArgumentError(
        "kms_symmetric_key_associated_data.record_header() cannot be "
        "parsed to BlobHeader.");
  }

  // Verify that the access policy hash matches one of the authorized
  // logical pipeline policy hashes returned by KMS before returning the key
  // ID.
  if (!GetAuthorizedLogicalPipelinePoliciesHashes().contains(
          blob_header.access_policy_sha256())) {
    return absl::InvalidArgumentError(
        "BlobHeader.access_policy_sha256 does not match any "
        "authorized_logical_pipeline_policies_hashes returned by KMS.");
  }
  return blob_header.key_id();
}

absl::Status FnConfidentialTransform::ReadWriteConfigurationRequest(
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

  auto& [current_file_path, expected_total_size_bytes, commit] =
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
        expected_total_size_bytes) {
      return absl::InvalidArgumentError(
          absl::StrCat("The total size of the data blob does not match "
                       "expected size. Expecting ",
                       expected_total_size_bytes, ", got ",
                       std::filesystem::file_size(current_file_path)));
    }
    LOG(INFO) << "Successfully wrote all " << expected_total_size_bytes
              << " bytes to " << current_file_path;
  }

  return absl::OkStatus();
}

}  // namespace confidential_federated_compute::fns
