/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "containers/tff_worker/pipeline_transform_server.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "fcp/base/status_converters.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"
#include "fcp/protos/confidentialcompute/tff_worker_configuration.pb.h"
#include "grpcpp/server_context.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/tensorflow/converters.h"
#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/federating_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/reference_resolving_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_executor.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::tff_worker {

namespace tf = ::tensorflow;
namespace tff = ::tensorflow_federated;

struct ComputationAndInput {
  tff::v0::Computation computation;
  tff::v0::Value input;
};

using ::fcp::base::ToGrpcStatus;
using ::fcp::confidentialcompute::ConfigureAndAttestRequest;
using ::fcp::confidentialcompute::ConfigureAndAttestResponse;
using ::fcp::confidentialcompute::TransformRequest;
using ::fcp::confidentialcompute::TransformResponse;
using ::grpc::ServerContext;

// TODO: Move these methods and other transform logic to a separate file
// where they can be tested independently and kept independent from logic
// around decryption/encryption & protecting the configuration with mutexes.
namespace {

using ::fcp::confidentialcompute::Record;
using ::fcp::confidentialcompute::TffWorkerConfiguration;
using ::fcp::confidentialcompute::TffWorkerConfiguration_ClientWork;
using ::fcp::confidentialcompute::
    TffWorkerConfiguration_ClientWork_FedSqlTensorflowCheckpoint;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::tensorflow::ToTfTensor;

// Returns the unencrypted_data field from a specified Record object.
//
// The input Record must outlive the returned string_view.
absl::StatusOr<absl::string_view> GetInputDataFromRecord(const Record& record) {
  if (!record.has_unencrypted_data()) {
    return absl::InvalidArgumentError(
        "Processing records containing any kind of data other than "
        "unencrypted_data is unimplemented.");
  }
  return record.unencrypted_data();
}

absl::Status MergeCardinalityMaps(const tff::CardinalityMap& map_to_add_from,
                                  tff::CardinalityMap& map_to_mutate) {
  for (const auto& [key, value] : map_to_add_from) {
    auto [it, inserted] = map_to_mutate.insert({key, value});
    if (!inserted && it->second != value) {
      return absl::InvalidArgumentError(
          absl::StrCat("Conflicting cardinalities for ", key, ": ", it->second,
                       " vs ", value));
    }
  }
  return absl::OkStatus();
}

// Create a TFF tff::CardinalityMap for a TFF Value.
//
// This is needed to generate a TFF execution stack.
absl::StatusOr<tff::CardinalityMap> CreateCardinalityMap(
    const tff::v0::Value& value) {
  tff::CardinalityMap cardinality_map;
  if (value.has_federated()) {
    const int num_constituents = value.federated().value_size();
    if (value.federated().type().placement().value().uri() == tff::kServerUri) {
      if (num_constituents != 1) {
        return absl::InvalidArgumentError(
            "Cannot handle values with SERVER-placed values with more than one "
            "constituent.");
      }
      cardinality_map[tff::kServerUri] = num_constituents;
      cardinality_map[tff::kClientsUri] = 0;
    } else if (value.federated().type().placement().value().uri() ==
               tff::kClientsUri) {
      cardinality_map[tff::kServerUri] = 0;
      cardinality_map[tff::kClientsUri] = num_constituents;
    } else {
      return absl::InvalidArgumentError(
          "Can only handle SERVER- or CLIENTS-placed values.");
    }
  } else if (value.has_struct_()) {
    for (const auto element : value.struct_().element()) {
      FCP_ASSIGN_OR_RETURN(tff::CardinalityMap element_cardinality_map,
                           CreateCardinalityMap(element.value()));
      FCP_RETURN_IF_ERROR(
          MergeCardinalityMaps(element_cardinality_map, cardinality_map));
    }
  }
  return cardinality_map;
}

// Translate the information from a client checkpoint to a TFF Value.
//
// This method uses the information from the `fed_sql_tf_checkpoint_spec` to
// read in the checkpoint appropriately.
//
// TODO: Reassess if we need to be generating federated values at all for this
// container.
absl::StatusOr<tff::v0::Value> RestoreClientCheckpointToDict(
    absl::string_view input,
    TffWorkerConfiguration_ClientWork_FedSqlTensorflowCheckpoint
        fed_sql_tf_checkpoint_spec) {
  tff::v0::Value restored_value;
  FederatedComputeCheckpointParserFactory parser_factory;
  absl::Cord client_stacked_tensor_result(input);
  // Generate a
  // std::unique_ptr<::tensorflow_federated::aggregation::CheckpointParser>.
  FCP_ASSIGN_OR_RETURN(auto parser,
                       parser_factory.Create(client_stacked_tensor_result));

  // The output is a federated CLIENTS-placed Value with a struct of Tensors
  // given the names from the `fed_sql_tf_checkpoint_spec` and the values in
  // the FCP checkpoint.
  tff::v0::Value_Federated* federated = restored_value.mutable_federated();
  tff::v0::FederatedType* type_proto = federated->mutable_type();
  type_proto->set_all_equal(true);
  *type_proto->mutable_placement()->mutable_value()->mutable_uri() =
      tff::kClientsUri;
  tff::v0::Value_Struct* value_struct =
      federated->add_value()->mutable_struct_();
  for (auto& column : fed_sql_tf_checkpoint_spec.fed_sql_columns()) {
    FCP_ASSIGN_OR_RETURN(Tensor tensor_column_values,
                         parser->GetTensor(column.name()));
    absl::StatusOr<tf::Tensor> tensor =
        ToTfTensor(std::move(tensor_column_values));
    tf::TensorProto tensor_proto;
    tensor->AsProtoTensorContent(&tensor_proto);
    tff::v0::Value_Struct_Element* element = value_struct->add_element();
    element->set_name(column.name());
    element->mutable_value()->mutable_tensor()->PackFrom(tensor_proto);
  }
  return restored_value;
}

// Creates a local TF-based TFF executor stack.
//
//
//                   [reference resolving executor]
//                                |
//                       [federating executor]
//                                |
//                -----------------------------------
//                 |                                |
//  [reference resolving executor]  [reference resolving executor]
//                 |                                |
//           [tf executor]                   [tf executor]
//
//
absl::StatusOr<std::shared_ptr<tff::Executor>> CreateExecutorStack(
    const tff::CardinalityMap& cardinality_map) {
  std::shared_ptr<tff::Executor> tf_executor = tff::CreateTensorFlowExecutor();
  std::shared_ptr<tff::Executor> reference_resolving_executor =
      tff::CreateReferenceResolvingExecutor(tf_executor);
  FCP_ASSIGN_OR_RETURN(
      std::shared_ptr<tff::Executor> federating_executor,
      tff::CreateFederatingExecutor(
          /*server_child=*/reference_resolving_executor,
          /*client_child=*/reference_resolving_executor, cardinality_map));
  return tff::CreateReferenceResolvingExecutor(federating_executor);
}

absl::StatusOr<ComputationAndInput> GetClientComputationAndInput(
    const TffWorkerConfiguration_ClientWork& client_work,
    const Record& record) {
  ComputationAndInput computation_and_input;

  // Read in TFF computation from the TffWorkerConfiguration.
  tff::v0::Computation client_work_computation;
  if (!computation_and_input.computation.ParseFromString(
          client_work.serialized_client_work_computation())) {
    return absl::InvalidArgumentError(
        "Cannot parse serialized client work computation.");
  }

  // Read in the broadcasted TFF value from the TffWorkerConfiguration.
  tff::v0::Value broadcasted_value;
  if (!broadcasted_value.ParseFromString(
          client_work.serialized_broadcasted_data())) {
    return absl::InvalidArgumentError(
        "Cannot parse serialized broadcasted TFF Value for client work.");
  }

  // Read in FCP checkpoint and transform into the bytes of a TF checkpoint.
  // This conversion is necessary to execute the TF-based TFF computation.
  FCP_ASSIGN_OR_RETURN(absl::string_view input, GetInputDataFromRecord(record));
  FCP_ASSIGN_OR_RETURN(tff::v0::Value input_value,
                       RestoreClientCheckpointToDict(
                           input, client_work.fed_sql_tensorflow_checkpoint()));

  // Create a Struct-type TFF value of the shape
  // (input TF value from checkpoint, broadcasted value).
  // The TFF execution stack expects a single argument for the computation to
  // be invoked on.
  *(computation_and_input.input.mutable_struct_()
        ->add_element()
        ->mutable_value()) = std::move(input_value);
  *(computation_and_input.input.mutable_struct_()
        ->add_element()
        ->mutable_value()) = std::move(broadcasted_value);

  return computation_and_input;
}

}  // namespace

absl::Status TffPipelineTransform::TffTransform(const TransformRequest* request,
                                                TransformResponse* response) {
  ComputationAndInput computation_and_input;
  {
    absl::MutexLock l(&mutex_);
    if (!tff_worker_configuration_.has_value()) {
      return absl::FailedPreconditionError(
          "ConfigureAndAttest must be called before Transform.");
    }

    // Extract the computation and input value from the TffWorkerConfiguration.
    if (tff_worker_configuration_->has_client_work()) {
      if (request->inputs_size() != 1) {
        return absl::InvalidArgumentError(
            "Exactly one input must be provided to a `client_work` transform "
            "but got " +
            std::to_string(request->inputs_size()));
      }

      Record record = request->inputs(0);
      FCP_ASSIGN_OR_RETURN(
          computation_and_input,
          GetClientComputationAndInput(tff_worker_configuration_->client_work(),
                                       record));
    } else if (tff_worker_configuration_->has_aggregation()) {
      return absl::UnimplementedError(
          "Transform has not yet implemented aggregation transformations.");
    } else {
      return absl::InvalidArgumentError(
          "TffWorkerConfiguration must contain client work or an aggregation.");
    }
  }

  // Create a TFF tff::CardinalityMap. This object stores the cardinality of the
  // different placements in a TFF Value.
  FCP_ASSIGN_OR_RETURN(tff::CardinalityMap cardinality_map,
                       CreateCardinalityMap(computation_and_input.input));

  // Create the TFF Executor stack that will execute the computation. This
  // stack is currently created per computation call.
  // TODO: Cache executor stacks for different cardinalities.
  FCP_ASSIGN_OR_RETURN(std::shared_ptr<tff::Executor> executor,
                       CreateExecutorStack(cardinality_map));

  // Create a value in the Executor stack for the full input argument Value.
  FCP_ASSIGN_OR_RETURN(tff::OwnedValueId owned_value_id,
                       executor->CreateValue(computation_and_input.input));

  // Create a value in the Executor stack for the computation.
  tff::v0::Value computation_value_pb;
  *computation_value_pb.mutable_computation() =
      computation_and_input.computation;
  FCP_ASSIGN_OR_RETURN(tff::OwnedValueId computation_value_id,
                       executor->CreateValue(std::move(computation_value_pb)));

  // Create a call in the Executor stack for the invocation of the TFF
  // computation on the input Value.
  FCP_ASSIGN_OR_RETURN(
      tff::OwnedValueId all_value_id,
      executor->CreateCall(computation_value_id, owned_value_id));

  // Materialize the output of the computation invocation and pass to the
  // output.
  // TODO: Handle the case of encrypted data.
  tff::v0::Value output_value_pb;
  FCP_RETURN_IF_ERROR(executor->Materialize(all_value_id, &output_value_pb));
  std::string serialized_output;
  output_value_pb.SerializeToString(&serialized_output);
  Record* output = response->add_outputs();
  output->set_unencrypted_data(std::move(serialized_output));
  return absl::OkStatus();
}

grpc::Status TffPipelineTransform::Transform(ServerContext* context,
                                             const TransformRequest* request,
                                             TransformResponse* response) {
  return ToGrpcStatus(TffTransform(request, response));
}

absl::Status TffPipelineTransform::TffConfigureAndAttest(
    const ConfigureAndAttestRequest* request,
    ConfigureAndAttestResponse* response) {
  absl::MutexLock l(&mutex_);
  if (tff_worker_configuration_.has_value()) {
    return absl::FailedPreconditionError(
        "ConfigureAndAttest can only be called once.");
  }
  if (!request->has_configuration()) {
    return absl::InvalidArgumentError(
        "ConfigureAndAttestRequest must contain configuration.");
  }

  TffWorkerConfiguration tff_worker_configuration;
  if (!request->configuration().UnpackTo(&tff_worker_configuration)) {
    return absl::InvalidArgumentError(
        "ConfigureAndAttestRequest configuration must be a "
        "tff_worker_configuration_pb2.TffWorkerConfiguration.");
  }
  tff_worker_configuration_ =
      std::make_optional<TffWorkerConfiguration>(tff_worker_configuration);

  // TODO: When encryption is implemented, this should cause generation of a new
  // keypair.
  return absl::OkStatus();
}

grpc::Status TffPipelineTransform::ConfigureAndAttest(
    ServerContext* context, const ConfigureAndAttestRequest* request,
    ConfigureAndAttestResponse* response) {
  return ToGrpcStatus(TffConfigureAndAttest(request, response));
}

}  // namespace confidential_federated_compute::tff_worker
