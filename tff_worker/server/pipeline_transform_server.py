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
import grpc
from tff_worker.server.pipeline_transform_py_grpc_pb.fcp.protos.confidentialcompute import pipeline_transform_pb2_grpc


class PipelineTransformServicer(
    pipeline_transform_pb2_grpc.PipelineTransformServicer
):
  """Provides methods that implement functionality of Pipeline Transform server."""

  def Initialize(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Initialize not implemented!')
    raise NotImplementedError('Initialize not implemented!')

  def Transform(self, request, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Transform not implemented!')
    raise NotImplementedError('Transform not implemented!')


def serve():
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  pipeline_transform_pb2_grpc.add_PipelineTransformServicer_to_server(
      PipelineTransformServicer(), server
  )
  server.add_insecure_port('[::]:50051')
  print('Starting Pipeline Transform server')
  server.start()
  server.wait_for_termination()


if __name__ == '__main__':
  serve()
