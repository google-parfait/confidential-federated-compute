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

#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "containers/crypto.h"
#include "fcp/aggregation/protocol/checkpoint_aggregator.h"
#include "fcp/aggregation/protocol/configuration.pb.h"
#include "fcp/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "fcp/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "fcp/base/monitoring.h"
#include "fcp/base/status_converters.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"
#include "google/protobuf/repeated_ptr_field.h"

namespace confidential_federated_compute::agg_core {

using ::fcp::aggregation::CheckpointAggregator;
using ::fcp::aggregation::CheckpointBuilder;
using ::fcp::aggregation::CheckpointParser;
using ::fcp::aggregation::Configuration;
using ::fcp::aggregation::FederatedComputeCheckpointBuilderFactory;
using ::fcp::aggregation::FederatedComputeCheckpointParserFactory;
using ::fcp::aggregation::Tensor;
using ::fcp::base::ToGrpcStatus;
using ::fcp::confidentialcompute::ConfigureAndAttestRequest;
using ::fcp::confidentialcompute::ConfigureAndAttestResponse;
using ::fcp::confidentialcompute::PipelineTransform;
using ::fcp::confidentialcompute::Record;
using ::fcp::confidentialcompute::TransformRequest;
using ::fcp::confidentialcompute::TransformResponse;
using ::grpc::ServerContext;

absl::Status AggCorePipelineTransform::AggCoreConfigureAndAttest(
    const ConfigureAndAttestRequest* request,
    ConfigureAndAttestResponse* response) {
  Configuration config;
  request->configuration().UnpackTo(&config);
  FCP_RETURN_IF_ERROR(CheckpointAggregator::ValidateConfig(config));
  const RecordDecryptor* record_decryptor;
  {
    absl::MutexLock l(&mutex_);
    if (configuration_ != std::nullopt) {
      return absl::FailedPreconditionError(
          "ConfigureAndAttest can only be called once.");
    }

    configuration_.emplace(config);
    // TODO: Once the DP intrinsic is implemented, we'll need to pull out the
    // attestable properties from the configuration and send them to the
    // RecordDecryptor via the `config_properties` argument.
    record_decryptor_.emplace(crypto_stub_);

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

static absl::Status AccumulateRecord(const Record& record,
                                     CheckpointAggregator* aggregator,
                                     RecordDecryptor* record_decryptor) {
  FCP_ASSIGN_OR_RETURN(std::string unencrypted_data,
                       record_decryptor->DecryptRecord(record));
  FederatedComputeCheckpointParserFactory parser_factory;
  FCP_ASSIGN_OR_RETURN(std::unique_ptr<CheckpointParser> parser,
                       parser_factory.Create(absl::Cord(unencrypted_data)));
  FCP_RETURN_IF_ERROR(aggregator->Accumulate(*parser));
  return absl::OkStatus();
}

absl::Status AggCorePipelineTransform::AggCoreTransform(
    const TransformRequest* request, TransformResponse* response) {
  RecordDecryptor* record_decryptor;
  const Configuration* configuration;
  {
    absl::MutexLock l(&mutex_);
    if (configuration_ == std::nullopt || record_decryptor_ == std::nullopt) {
      return absl::FailedPreconditionError(
          "ConfigureAndAttest must be called before Transform.");
    }

    // Since record_decryptor_ and configuration_ are set once in
    // ConfigureAndAttest and never modified, and both underlying objects are
    // threadsafe, it is safe to store a local pointer to them and access the
    // objects without a lock after we check under the mutex that values have
    // been set for the std::optional wrappers.
    record_decryptor = &*record_decryptor_;
    configuration = &*configuration_;
  }
  FCP_ASSIGN_OR_RETURN(std::unique_ptr<CheckpointAggregator> aggregator,
                       CheckpointAggregator::Create(*configuration));
  std::vector<std::thread> threads;
  for (const Record& record : request->inputs()) {
    threads.push_back(std::thread(AccumulateRecord, record, aggregator.get(),
                                  record_decryptor));
  }

  for (std::thread& t : threads) {
    t.join();
  }

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
