# Copyright 2023 The Confidential Federated Compute Authors.
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
import grpc
from tff_worker.server import tff_transforms


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
  """Provides methods that implement functionality of Pipeline Transform server."""

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
      context.abort(
          grpc.StatusCode.UNIMPLEMENTED,
          'Performing a `client_work` Transform is unimplemented.',
      )
    elif self._configuration.HasField('aggregation'):
      unencrypted_inputs = [
          _get_input_data_from_record(context, record)
          for record in request.inputs
      ]
      try:
        output_data = tff_transforms.aggregate(
            self._configuration.aggregation.serialized_client_to_server_aggregation_computation,
            self._configuration.aggregation.serialized_temporary_state,
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
  server.add_insecure_port('[::]:50051')
  logging.info('Starting Pipeline Transform server')
  server.start()
  server.wait_for_termination()


if __name__ == '__main__':
  serve()
