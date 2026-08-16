# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for execution_context_helper."""

from absl.testing import absltest
from fcp.protos.confidentialcompute import computation_delegation_pb2
from fcp.protos.confidentialcompute import tff_config_pb2
import federated_language
from google.protobuf import any_pb2
import numpy as np
from program_executor_tee.program_context import execution_context_helper
import tensorflow_federated as tff
from tensorflow_federated.proto.v0 import executor_pb2


class ExecutionContextHelperTest(absltest.TestCase):

  def test_create_computation_request_no_arg(self):
    @federated_language.federated_computation()
    def return_ten():
      return 10

    req = execution_context_helper.create_computation_request(return_ten)
    self.assertTrue(
        req.computation.Is(tff_config_pb2.TffSessionConfig.DESCRIPTOR)
    )
    session_config = tff_config_pb2.TffSessionConfig()
    req.computation.Unpack(session_config)
    self.assertIsInstance(session_config.function, executor_pb2.Value)
    self.assertEqual(session_config.num_clients, 0)
    self.assertFalse(session_config.HasField("initial_arg"))

  def test_create_computation_request_with_arg(self):
    @federated_language.federated_computation(
        federated_language.FederatedType(np.int32, federated_language.CLIENTS)
    )
    def sum_comp(val):
      return federated_language.federated_sum(val)

    req = execution_context_helper.create_computation_request(
        sum_comp, arg=[10, 20, 30]
    )
    self.assertTrue(
        req.computation.Is(tff_config_pb2.TffSessionConfig.DESCRIPTOR)
    )
    session_config = tff_config_pb2.TffSessionConfig()
    req.computation.Unpack(session_config)
    self.assertIsInstance(session_config.function, executor_pb2.Value)
    self.assertEqual(session_config.num_clients, 3)
    self.assertTrue(session_config.HasField("initial_arg"))
    self.assertIsInstance(session_config.initial_arg, executor_pb2.Value)

  def test_create_computation_request_server_arg(self):
    server_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )

    @federated_language.federated_computation(server_type)
    def identity_comp(val):
      return val

    req = execution_context_helper.create_computation_request(
        identity_comp, arg=42
    )
    session_config = tff_config_pb2.TffSessionConfig()
    req.computation.Unpack(session_config)
    self.assertEqual(session_config.num_clients, 0)

  def test_create_computation_request_mixed_arg(self):
    client_type = federated_language.FederatedType(
        np.int32, federated_language.CLIENTS
    )
    server_type = federated_language.FederatedType(
        np.int32, federated_language.SERVER
    )

    @federated_language.federated_computation([client_type, server_type])
    def mixed_comp(client_val, server_val):
      return federated_language.federated_sum(client_val), server_val

    req = execution_context_helper.create_computation_request(
        mixed_comp, arg=([10, 20, 30, 40], 99)
    )
    session_config = tff_config_pb2.TffSessionConfig()
    req.computation.Unpack(session_config)
    self.assertEqual(session_config.num_clients, 4)

  def test_unpack_and_deserialize_computation_response(self):
    val_proto, _ = tff.framework.serialize_value(
        np.int32(42), federated_language.TensorType(np.int32)
    )
    any_proto = any_pb2.Any()
    any_proto.Pack(val_proto)
    resp = computation_delegation_pb2.ComputationResponse(result=any_proto)
    res = execution_context_helper.unpack_and_deserialize_computation_response(
        resp, federated_language.TensorType(np.int32)
    )
    self.assertEqual(res, 42)


if __name__ == "__main__":
  absltest.main()
