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
#include "containers/agg_core/pipeline_transform_server.h"

#include <execution>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "containers/crypto.h"
#include "fcp/base/status_converters.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/intrinsic.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_aggregator.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/config_converter.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/configuration.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"

namespace confidential_federated_compute::agg_core {

namespace {

using ::fcp::base::ToGrpcStatus;
using ::fcp::confidentialcompute::ConfigureAndAttestRequest;
using ::fcp::confidentialcompute::ConfigureAndAttestResponse;
using ::fcp::confidentialcompute::PipelineTransform;
using ::fcp::confidentialcompute::Record;
using ::fcp::confidentialcompute::TransformRequest;
using ::fcp::confidentialcompute::TransformResponse;
using ::grpc::ServerContext;
using ::tensorflow_federated::aggregation::CheckpointAggregator;
using ::tensorflow_federated::aggregation::CheckpointBuilder;
using ::tensorflow_federated::aggregation::CheckpointParser;
using ::tensorflow_federated::aggregation::Configuration;
using ::tensorflow_federated::aggregation::DT_DOUBLE;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::Intrinsic;
using ::tensorflow_federated::aggregation::Tensor;

constexpr char kFedSqlDPGroupByUri[] = "fedsql_dp_group_by";

absl::Status ValidateFedSqlDpGroupByParameters(const Intrinsic& intrinsic) {
  if (intrinsic.parameters.size() < 2) {
    return absl::InvalidArgumentError(
        "`fedsql_dp_group_by` IntrinsicConfig must have at least two "
        "parameters.");
  }
  if (intrinsic.parameters.at(0).dtype() != DT_DOUBLE ||
      intrinsic.parameters.at(1).dtype() != DT_DOUBLE) {
    return absl::InvalidArgumentError(
        "`fedsql_dp_group_by` parameters must both be have type DT_DOUBLE.");
  }
  if (intrinsic.parameters.at(0).num_elements() != 1 ||
      intrinsic.parameters.at(1).num_elements() != 1) {
    return absl::InvalidArgumentError(
        "`fedsql_dp_group_by` parameters must each have exactly one value.");
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

// Decrypts and parses a record and accumulates it into the state of the
// CheckpointAggregator `aggregator`.
//
// Returns `true` if the input was successfully decrypted, parsed, and
// accumulated into the state of `aggregator`.
//
// Returns `false` if the input could not be successfully decrypted or parsed
// but `aggregator` is guaranteed to be valid and unchanged.
//
// Returns a failed status if accumulating the input failed and `aggregator`
// should no longer be used due to being in a potentially invalid state where
// an input was partially accumulated.
absl::StatusOr<bool> AccumulateRecord(const Record& record,
                                      CheckpointAggregator* aggregator,
                                      RecordDecryptor* record_decryptor) {
  absl::StatusOr<std::string> unencrypted_data =
      record_decryptor->DecryptRecord(record);
  if (!unencrypted_data.ok()) {
    LOG(WARNING) << "Failed to decrypt input: " << unencrypted_data.status();
    return false;
  }

  FederatedComputeCheckpointParserFactory parser_factory;
  absl::StatusOr<std::unique_ptr<CheckpointParser>> parser =
      parser_factory.Create(absl::Cord(*std::move(unencrypted_data)));
  if (!parser.ok()) {
    LOG(WARNING) << "Failed to parse input into the expected "
                    "FederatedComputeCheckpoint format"
                 << parser.status();
    return false;
  }
  // The CheckpointAggregator contract does not guarantee that it will
  // be in a valid state if an Accumulate operation returns an invalid status.
  // If this changes, consider changing this logic to no longer return an error.
  FCP_RETURN_IF_ERROR(aggregator->Accumulate(*parser.value()));
  return true;
}
}  // namespace

absl::Status AggCorePipelineTransform::AggCoreConfigureAndAttest(
    const ConfigureAndAttestRequest* request,
    ConfigureAndAttestResponse* response) {
  Configuration config;
  if (!request->configuration().UnpackTo(&config)) {
    return absl::InvalidArgumentError("Configuration cannot be unpacked.");
  }
  FCP_RETURN_IF_ERROR(CheckpointAggregator::ValidateConfig(config));
  const RecordDecryptor* record_decryptor;
  {
    absl::MutexLock l(&mutex_);
    if (intrinsics_ != std::nullopt) {
      return absl::FailedPreconditionError(
          "ConfigureAndAttest can only be called once.");
    }

    FCP_ASSIGN_OR_RETURN(
        std::vector<Intrinsic> intrinsics,
        tensorflow_federated::aggregation::ParseFromConfig(config));
    FCP_RETURN_IF_ERROR(ValidateTopLevelIntrinsics(intrinsics));
    google::protobuf::Struct config_properties;
    (*config_properties.mutable_fields())["intrinsic_uri"].set_string_value(
        intrinsics.at(0).uri);
    if (intrinsics.at(0).uri == kFedSqlDPGroupByUri) {
      const Intrinsic& fedsql_dp_intrinsic = intrinsics.at(0);
      FCP_RETURN_IF_ERROR(
          ValidateFedSqlDpGroupByParameters(fedsql_dp_intrinsic));
      double epsilon =
          fedsql_dp_intrinsic.parameters.at(0).CastToScalar<double>();
      double delta =
          fedsql_dp_intrinsic.parameters.at(1).CastToScalar<double>();
      (*config_properties.mutable_fields())["epsilon"].set_number_value(
          epsilon);
      (*config_properties.mutable_fields())["delta"].set_number_value(delta);
    }

    intrinsics_.emplace(std::move(intrinsics));
    record_decryptor_.emplace(crypto_stub_, config_properties);

    // Since record_decryptor_ is set once in ConfigureAndAttest and never
    // modified, and the underlying object is threadsafe, it is safe to store a
    // local pointer to it and access the object without a lock after we check
    // under the mutex that a value has been set for the std::optional wrapper.
    record_decryptor = &*record_decryptor_;
  }

  FCP_ASSIGN_OR_RETURN(*response->mutable_public_key(),
                       record_decryptor->GetPublicKey());
  return absl::OkStatus();
}

absl::Status AggCorePipelineTransform::AggCoreGenerateNonces(
    const fcp::confidentialcompute::GenerateNoncesRequest* request,
    fcp::confidentialcompute::GenerateNoncesResponse* response) {
  RecordDecryptor* record_decryptor;
  {
    absl::MutexLock l(&mutex_);
    if (record_decryptor_ == std::nullopt) {
      return absl::FailedPreconditionError(
          "ConfigureAndAttest must be called before GenerateNonces.");
    }
    // Since record_decryptor_ is set once in ConfigureAndAttest and never
    // modified, and the underlying object is threadsafe, it is safe to store a
    // local pointer to it and access the object without a lock after we check
    // under the mutex that a value has been set for the std::optional wrapper.
    record_decryptor = &*record_decryptor_;
  }
  for (int i = 0; i < request->nonces_count(); ++i) {
    FCP_ASSIGN_OR_RETURN(std::string nonce, record_decryptor->GenerateNonce());
    response->add_nonces(std::move(nonce));
  }
  return absl::OkStatus();
}

absl::Status AggCorePipelineTransform::AggCoreTransform(
    const TransformRequest* request, TransformResponse* response) {
  RecordDecryptor* record_decryptor;
  std::unique_ptr<CheckpointAggregator> aggregator;
  {
    absl::MutexLock l(&mutex_);
    if (intrinsics_ == std::nullopt || record_decryptor_ == std::nullopt) {
      return absl::FailedPreconditionError(
          "ConfigureAndAttest must be called before Transform.");
    }

    // Since record_decryptor_ is set once in ConfigureAndAttest and never
    // modified, and the underlying object is threadsafe, it is safe to store a
    // local pointer to it and access the object without a lock after we check
    // under the mutex that values have been set for the std::optional wrappers.
    record_decryptor = &*record_decryptor_;

    FCP_ASSIGN_OR_RETURN(aggregator,
                         CheckpointAggregator::Create(&*intrinsics_));
  }

  std::vector<absl::StatusOr<bool>> accumulate_results;
  accumulate_results.resize(request->inputs_size());

  std::transform(
#ifndef _LIBCPP_VERSION
      // std::execution is not currently supported by libc++.
      std::execution::par,
#endif
      request->inputs().begin(), request->inputs().end(),
      accumulate_results.begin(),
      [&aggregator, record_decryptor](const Record& record) {
        return AccumulateRecord(record, aggregator.get(), record_decryptor);
      });

  uint32_t ignored_inputs = 0;
  for (absl::StatusOr<bool> result : accumulate_results) {
    FCP_RETURN_IF_ERROR(result);
    if (!result.value()) {
      ++ignored_inputs;
    }
  }
  response->set_num_ignored_inputs(ignored_inputs);

  if (!aggregator->CanReport()) {
    return absl::FailedPreconditionError(
        "The aggregation can't be completed due to failed preconditions.");
  }

  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> checkpoint_builder =
      builder_factory.Create();
  FCP_RETURN_IF_ERROR(aggregator->Report(*checkpoint_builder));
  FCP_ASSIGN_OR_RETURN(absl::Cord checkpoint_cord, checkpoint_builder->Build());
  // Protobuf version 23.0 is required to use [ctype = CORD], however, we can't
  // use this since it isn't currently compatible with TensorFlow.
  std::string checkpoint_string;
  absl::CopyCordToString(checkpoint_cord, &checkpoint_string);

  Record* output = response->add_outputs();
  output->set_unencrypted_data(std::move(checkpoint_string));
  output->set_compression_type(Record::COMPRESSION_TYPE_NONE);
  return absl::OkStatus();
}

grpc::Status AggCorePipelineTransform::ConfigureAndAttest(
    ServerContext* context, const ConfigureAndAttestRequest* request,
    ConfigureAndAttestResponse* response) {
  return ToGrpcStatus(AggCoreConfigureAndAttest(request, response));
}

grpc::Status AggCorePipelineTransform::GenerateNonces(
    grpc::ServerContext* context,
    const fcp::confidentialcompute::GenerateNoncesRequest* request,
    fcp::confidentialcompute::GenerateNoncesResponse* response) {
  return ToGrpcStatus(AggCoreGenerateNonces(request, response));
}

grpc::Status AggCorePipelineTransform::Transform(
    ServerContext* context, const TransformRequest* request,
    TransformResponse* response) {
  return ToGrpcStatus(AggCoreTransform(request, response));
}

}  // namespace confidential_federated_compute::agg_core
