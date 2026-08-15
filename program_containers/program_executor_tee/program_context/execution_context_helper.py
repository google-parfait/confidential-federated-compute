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
"""Helper utilities for serialized computation requests and responses."""

from typing import Optional

from fcp.protos.confidentialcompute import computation_delegation_pb2
from fcp.protos.confidentialcompute import tff_config_pb2
import federated_language
from google.protobuf import any_pb2
import tensorflow_federated as tff
from tensorflow_federated.proto.v0 import executor_pb2


def create_computation_request(
    comp: object, arg: Optional[object] = None
) -> computation_delegation_pb2.ComputationRequest:
  """Builds a ComputationRequest containing the serialized comp, arg, and num_clients."""
  session_config = tff_config_pb2.TffSessionConfig()
  serialized_comp, _ = tff.framework.serialize_value(comp)
  session_config.function.CopyFrom(serialized_comp)

  serialized_arg = None
  clients_cardinality = 0
  if arg is not None:
    cardinalities = federated_language.framework.infer_cardinalities(
        arg, comp.type_signature.parameter
    )
    clients_cardinality = cardinalities.get(
        federated_language.CLIENTS, clients_cardinality
    )
    serialized_arg, _ = tff.framework.serialize_value(
        arg, comp.type_signature.parameter
    )
    session_config.initial_arg.CopyFrom(serialized_arg)
  session_config.num_clients = clients_cardinality

  any_proto = any_pb2.Any()
  any_proto.Pack(session_config)
  return computation_delegation_pb2.ComputationRequest(computation=any_proto)


def unpack_and_deserialize_computation_response(
    response: computation_delegation_pb2.ComputationResponse,
    return_type: federated_language.Type,
) -> object:
  """Unpacks and deserializes the ComputationResponse into a Python structure."""
  result_value = executor_pb2.Value()
  response.result.Unpack(result_value)
  deserialized_result, _ = tff.framework.deserialize_value(result_value)
  if isinstance(
      deserialized_result, federated_language.common_libs.structure.Struct
  ):
    deserialized_result = (
        federated_language.common_libs.structure.to_odict_or_tuple(
            deserialized_result
        )
    )
  return federated_language.framework.to_structure_with_type(
      deserialized_result, return_type
  )
