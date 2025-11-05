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
#include "program_executor_tee/confidential_transform_server.h"

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <execution>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "containers/blob_metadata.h"
#include "containers/crypto.h"
#include "containers/session.h"
#include "fcp/base/status_converters.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/confidential_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/data_read_write.pb.h"
#include "fcp/protos/confidentialcompute/program_executor_tee_config.pb.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "grpcpp/support/status.h"
#include "program_executor_tee/program_context/cc/data_parser.h"
#include "pybind11_protobuf/native_proto_caster.h"

namespace confidential_federated_compute::program_executor_tee {
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::FinalizeRequest;
using ::fcp::confidentialcompute::ProgramExecutorTeeInitializeConfig;
using ::fcp::confidentialcompute::ReadResponse;
using ::fcp::confidentialcompute::SessionResponse;
using ::fcp::confidentialcompute::WriteRequest;
using ::google::protobuf::Struct;

PYBIND11_EMBEDDED_MODULE(data_parser, m) {
  pybind11_protobuf::ImportNativeProtoCasters();

  pybind11::class_<BlobDecryptor>(m, "BlobDecryptor");

  pybind11::class_<DataParser>(m, "DataParser")
      .def(pybind11::init<BlobDecryptor*, std::string&>())
      .def("resolve_uri_to_tensor",
           [](DataParser& self, std::string& uri, std::string& key) {
             auto tensor = self.ResolveUriToTensor(uri, key);
             if (!tensor.ok()) {
               throw std::runtime_error("Failed to fetch Tensor: " +
                                        std::string(tensor.status().message()));
             }
             return *tensor;
           });
}

absl::StatusOr<fcp::confidentialcompute::ConfigureResponse>
ProgramExecutorTeeSession::Configure(
    fcp::confidentialcompute::ConfigureRequest request, Context& context) {
  return fcp::confidentialcompute::ConfigureResponse();
}

absl::StatusOr<fcp::confidentialcompute::WriteFinishedResponse>
ProgramExecutorTeeSession::Write(fcp::confidentialcompute::WriteRequest request,
                                 std::string unencrypted_data,
                                 Context& context) {
  return absl::UnimplementedError(
      "Writing to a session is not supported in program executor TEE.");
}

absl::StatusOr<fcp::confidentialcompute::CommitResponse>
ProgramExecutorTeeSession::Commit(
    fcp::confidentialcompute::CommitRequest request, Context& context) {
  return absl::UnimplementedError(
      "Committing to a session is not supported in program executor TEE.");
}

absl::StatusOr<fcp::confidentialcompute::FinalizeResponse>
ProgramExecutorTeeSession::Finalize(
    fcp::confidentialcompute::FinalizeRequest request,
    fcp::confidentialcompute::BlobMetadata input_metadata, Context& context) {
  std::vector<std::string> client_ids;
  client_ids.reserve(initialize_config_.client_ids().size());
  for (const auto& client_id : initialize_config_.client_ids()) {
    client_ids.push_back(client_id);
  }

  std::vector<std::string> worker_bns_addresses;
  worker_bns_addresses.reserve(
      initialize_config_.worker_bns_addresses().size());
  for (const auto& address : initialize_config_.worker_bns_addresses()) {
    worker_bns_addresses.push_back(address);
  }

  pybind11::scoped_interpreter guard{};
  try {
    // Load the python function for running the program.
    auto run_program =
        pybind11::module::import(
            "program_executor_tee.program_context.program_runner")
            .attr("run_program");

    // Create a DataParser object bound to the BlobDecryptor pointer.
    pybind11::object data_parser_instance =
        pybind11::module::import("data_parser")
            .attr("DataParser")(blob_decryptor_,
                                initialize_config_.outgoing_server_address());

    run_program(get_program_initialize_fn_(),
                pybind11::bytes(initialize_config_.program()), client_ids,
                initialize_config_.client_data_dir(), model_id_to_zip_file_,
                initialize_config_.outgoing_server_address(),
                data_parser_instance.attr("resolve_uri_to_tensor"));
  } catch (const std::exception& e) {
    LOG(INFO) << "Error executing federated program: " << e.what();
  }
  return fcp::confidentialcompute::FinalizeResponse();
}

absl::StatusOr<google::protobuf::Struct>
ProgramExecutorTeeConfidentialTransform::StreamInitializeTransform(
    const fcp::confidentialcompute::InitializeRequest* request) {
  ProgramExecutorTeeInitializeConfig config;
  if (!request->configuration().UnpackTo(&config)) {
    return absl::InvalidArgumentError(
        "ProgramExecutorTeeInitializeConfig cannot be unpacked.");
  }
  initialize_config_ = std::move(config);

  worker_bns_addresses_.reserve(
      initialize_config_.worker_bns_addresses().size());
  for (const auto& address : initialize_config_.worker_bns_addresses()) {
    worker_bns_addresses_.push_back(address);
  }

  Struct config_properties;
  (*config_properties.mutable_fields())["program"].set_string_value(
      initialize_config_.program());
  (*config_properties.mutable_fields())["worker_reference_values"]
      .set_string_value(absl::Base64Escape(
          initialize_config_.reference_values().SerializeAsString()));
  return config_properties;
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

absl::Status
ProgramExecutorTeeConfidentialTransform::ReadWriteConfigurationRequest(
    const fcp::confidentialcompute::WriteConfigurationRequest&
        write_configuration) {
  std::ios_base::openmode file_open_mode;
  // First request metadata is set for the first WriteConfigurationRequest of a
  // new data blob.
  if (write_configuration.has_first_request_metadata()) {
    // Create a new file.
    file_open_mode = std::ios::binary;
    current_model_id_ =
        write_configuration.first_request_metadata().configuration_id();
    if (write_configuration_map_.find(current_model_id_) !=
        write_configuration_map_.end()) {
      return absl::InvalidArgumentError(
          "Duplicated configuration_id found in WriteConfigurationRequest.");
    }
    // Create new temp files. Temp files are saved as "/tmp/model_1.zip",
    // "/tmp/model_2.zip", etc. Use model_id to distinguish different model
    // file names.
    std::string temp_file_path =
        absl::StrCat("/tmp/", current_model_id_, ".zip");

    LOG(INFO) << "Start writing bytes for model_id: " << current_model_id_
              << " to " << temp_file_path;

    write_configuration_map_[current_model_id_] = WriteConfigurationMetadata{
        .file_path = std::move(temp_file_path),
        .total_size_bytes = static_cast<uint64_t>(
            write_configuration.first_request_metadata().total_size_bytes()),
        .commit = write_configuration.commit()};
  } else {
    // If the current write_configuration is not the first
    // WriteConfigurationRequest of a data blob, append to existing file.
    file_open_mode = std::ios::binary | std::ios::app;
  }

  auto& [current_file_path, current_total_size_bytes, commit] =
      write_configuration_map_[current_model_id_];
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

absl::StatusOr<std::string> ProgramExecutorTeeConfidentialTransform::GetKeyId(
    const fcp::confidentialcompute::BlobMetadata& metadata) {
  return GetKeyIdFromMetadata(metadata);
}

absl::StatusOr<std::unique_ptr<confidential_federated_compute::Session>>
ProgramExecutorTeeConfidentialTransform::CreateSession() {
  // Check that all zipped models passed in through WriteConfigurationRequest
  // are committed and populate model_id_to_zip_file_.
  for (const auto& [model_id, config_metadata] : write_configuration_map_) {
    if (!config_metadata.commit) {
      return absl::InvalidArgumentError(
          absl::StrCat("Model with model_id ", model_id, " is not committed."));
    }
    model_id_to_zip_file_[model_id] = config_metadata.file_path;
  }

  FCP_ASSIGN_OR_RETURN(BlobDecryptor * blob_decryptor, GetBlobDecryptor());

  auto get_program_initialize_fn =
      [this]() -> std::optional<pybind11::function> {
    return this->GetProgramInitializeFn();
  };

  return std::make_unique<ProgramExecutorTeeSession>(
      initialize_config_, model_id_to_zip_file_, blob_decryptor,
      get_program_initialize_fn);
}

}  // namespace confidential_federated_compute::program_executor_tee
