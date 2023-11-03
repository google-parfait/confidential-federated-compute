# Copyright 2023 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of gRPC Pipeline Transform server."""

from concurrent import futures
import logging
from fcp.protos.confidentialcompute import pipeline_transform_pb2
from fcp.protos.confidentialcompute import pipeline_transform_pb2_grpc
from fcp.protos.confidentialcompute import tff_worker_configuration_pb2
from google.protobuf import empty_pb2
import grpc
from oak.oak_containers.proto import interfaces_pb2_grpc as oak_containers_pb2_grpc
import tff_transforms


def _get_input_data_from_record(
    context: grpc.ServicerContext, input: pipeline_transform_pb2.Record
) -> bytes:
  """Retrieves unencrypted data from a `pipeline_transform_pb2.Record` proto.

  Args: `context`: A `grpc.ServicerContext` object which can be used to abort
    the ongoing RPC call. `input`: A `pipeline_transform_pb2.Record` proto.
    unencrypted_inputs: A list of bytestrings encoding TFF `executor_pb2.Value`
    protos each representing data derived from a single client. These inputs
    will be deserialized into values that can be passed as the second argument
    to the aggregation computation.

  Returns:
    Unencrypted data extracted from `input`, or aborts `context` if the
    data cannot be retrieved.
  """
  if not input.HasField('unencrypted_data'):
    context.abort(
        grpc.StatusCode.UNIMPLEMENTED,
        'Processing records containing any kind other than '
        'unencrypted_data is unimplemented.',
    )
  return input.unencrypted_data


class PipelineTransformServicer(
    pipeline_transform_pb2_grpc.PipelineTransformServicer
):
  """Implements functionality of Pipeline Transform server."""

  def __init__(self):
    self._configuration = None
    super().__init__()

  def Initialize(
      self,
      request: pipeline_transform_pb2.InitializeRequest,
      context: grpc.ServicerContext,
  ) -> pipeline_transform_pb2.InitializeResponse:
    """Implementation of Initialize RPC."""
    context.abort(grpc.StatusCode.UNIMPLEMENTED, 'Initialize not implemented!')

  def ConfigureAndAttest(
      self,
      request: pipeline_transform_pb2.ConfigureAndAttestRequest,
      context: grpc.ServicerContext,
  ) -> pipeline_transform_pb2.ConfigureAndAttestResponse:
    """Implementation of ConfigureAndAttest RPC."""
    if not request.HasField('configuration'):
      context.abort(
          grpc.StatusCode.INVALID_ARGUMENT,
          'ConfigureAndAttestRequest must contain configuration.',
      )
    self._configuration = tff_worker_configuration_pb2.TffWorkerConfiguration()
    # TODO: Introduce additional early configuration checks to fail faster on
    # invalid configuration.
    if not request.configuration.Unpack(self._configuration):
      context.abort(
          grpc.StatusCode.INVALID_ARGUMENT,
          'ConfigureAndAttestRequest configuration must be a'
          ' tff_worker_configuration_pb2.TffWorkerConfiguration',
      )
    # TODO: When encryption is implemented this should cause generation of a new
    # keypair.
    return pipeline_transform_pb2.ConfigureAndAttestResponse()

  def GenerateNonces(
      self,
      request: pipeline_transform_pb2.GenerateNoncesRequest,
      context: grpc.ServicerContext,
  ) -> pipeline_transform_pb2.GenerateNoncesResponse:
    """Implementation of GenerateNonces RPC."""
    context.abort(
        grpc.StatusCode.UNIMPLEMENTED, 'Generate nonces not implemented!'
    )

  def Transform(
      self,
      request: pipeline_transform_pb2.TransformRequest,
      context: grpc.ServicerContext,
  ) -> pipeline_transform_pb2.TransformResponse:
    """Implementation of Transform RPC."""
    if not self._configuration:
      context.abort(
          grpc.StatusCode.FAILED_PRECONDITION,
          'ConfigureAndAttest must be called before Transform.',
      )

    response = pipeline_transform_pb2.TransformResponse()
    if self._configuration.HasField('client_work'):
      if len(request.inputs) != 1:
        context.abort(
            grpc.StatusCode.INVALID_ARGUMENT,
            'Exactly one input must be provided to a `client_work` transform'
            f' but got {len(request.inputs)}',
        )
      unencrypted_input = _get_input_data_from_record(
          context, request.inputs[0]
      )
      try:
        output_data = tff_transforms.perform_client_work(
            self._configuration.client_work,
            unencrypted_input,
        )
      except TypeError as e:
        context.abort(
            grpc.StatusCode.INVALID_ARGUMENT,
            'TypeError when executing client work computation: %s' % str(e),
        )
      response.outputs.add().unencrypted_data = output_data
    elif self._configuration.HasField('aggregation'):
      unencrypted_inputs = [
          _get_input_data_from_record(context, record)
          for record in request.inputs
      ]
      try:
        output_data = tff_transforms.aggregate(
            self._configuration.aggregation,
            unencrypted_inputs,
        )
      except TypeError as e:
        context.abort(
            grpc.StatusCode.INVALID_ARGUMENT,
            'TypeError when executing aggregation computation: %s' % str(e),
        )
      # TODO: Once encrypted inputs and outputs are supported, this
      # implementation should output unencrypted data only if the
      # min_clients_in_aggregate requirement is met.
      response.outputs.add().unencrypted_data = output_data
    else:
      context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'Unknown input kind.')
    return response


def serve():
  logging.basicConfig()
  logging.root.setLevel(logging.INFO)
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  pipeline_transform_pb2_grpc.add_PipelineTransformServicer_to_server(
      PipelineTransformServicer(), server
  )
  server.add_insecure_port('[::]:8080')

  # The Orchestrator gRPC service is listening on a UDS path. See
  # https://github.com/project-oak/oak/blob/55901b8a4c898c00ecfc14ef4bc65f30cd31d6a9/oak_containers_hello_world_trusted_app/src/orchestrator_client.rs#L45
  channel = grpc.insecure_channel('unix:/oak_utils/orchestrator_ipc')
  oak_orchestrator_stub = oak_containers_pb2_grpc.OrchestratorStub(channel)

  logging.info('Starting Pipeline Transform server')
  server.start()

  empty_request = empty_pb2.Empty()
  _ = oak_orchestrator_stub.NotifyAppReady(empty_request)

  server.wait_for_termination()


if __name__ == '__main__':
  serve()
