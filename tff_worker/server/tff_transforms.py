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

"""Library for executing TFF transforms on unencrypted data."""
from typing import Any, List, Tuple, Type, Union
from google.protobuf import message
import tensorflow_federated as tff
from tensorflow_federated.proto.v0 import computation_pb2
from tensorflow_federated.proto.v0 import executor_pb2


def _check_type(
    obj: Any, t: Union[Type[Any], Tuple[Type[Any], ...]], name: str
) -> None:
  """Checks that `obj` has type `t`.

  Args:
    obj: Any object.
    t: A Python type.
    name: The name of the object to be used in the error message.

  Raises:
    TypeError: If `obj` is not an instance of type `t`.
  """
  if not isinstance(obj, t):
    raise TypeError(
        f'Expected {name} to be an instance of type {t!r}, but '
        f'found an instance of type {type(obj)!r}.'
    )


def _get_tff_computation(serialized_computation: bytes) -> tff.Computation:
  """Deserializes a TFF computation from a bytestring.

  Args:
    serialized_computation: bytes encoding a TFF `computation_pb2.Computation`
      proto.

  Returns:
    A `tff.Computation`.

  Raises:
    TypeError: If the bytes cannot be parsed into a valid `tff.Computation`.
  """
  computation_proto = computation_pb2.Computation()
  try:
    computation_proto.ParseFromString(serialized_computation)
  except message.DecodeError as decode_err:
    raise TypeError(
        'Could not parse serialized computation as a tff'
        ' computation_pb2.Computation'
    ) from decode_err
  return tff.framework.deserialize_computation(computation_proto)


def _get_tff_value(
    serialized_value: bytes, type_spec: tff.types.Type, name: str
) -> Any:
  """Deserializes a TFF value from a bytestring.

  Args:
    serialized_value: bytes encoding a TFF `executor_pb2.Value` proto.
    type_spec: a `tff.types.Type` specifying the expected type of the resulting
      value.

  Returns:
    A value of the `tff.types.Type` specified by `type_spec`.

  Raises:
    TypeError: If the bytes cannot be parsed into a valid value of the type
     specified by `type_spec`.
  """
  value_proto = executor_pb2.Value()
  try:
    value_proto.ParseFromString(serialized_value)
    value, deserialized_type = tff.framework.deserialize_value(value_proto)
  except Exception as e:
    raise TypeError(
        f'Could not parse {name} as a tff executor_pb2.Value of type '
        f'{repr(type_spec)}'
    ) from e
  type_spec.check_assignable_from(deserialized_type)
  return value


def _serialize_output(output_value: Any, type_spec: tff.types.Type) -> bytes:
  """Serializes a TFF value to a bytestring.

  Args:
    output_value: The TFF value to be serialized as a bytestring encoding a
      `executor_pb2.Value` proto.
    type_spec: a `tff.types.Type` specifying the type of the value.

  Returns:
    A bytestring encoding a TFF `executor_pb2.Value` proto representing
    `output_value`.
  """
  (output_value_proto, _) = tff.framework.serialize_value(
      output_value, type_spec
  )
  return output_value_proto.SerializeToString()


def aggregate(
    serialized_client_to_server_aggregation_computation: bytes,
    serialized_temporary_state: bytes,
    unencrypted_inputs: List[bytes],
) -> bytes:
  """Aggregates unencrypted inputs using a serialized TFF computation.

  Args:
    serialized_client_to_server_aggregation_computation: bytes encoding a TFF
      `computation_pb2.Computation` proto which specifies how to perform the
      aggregation. The `tff.Computation` resulting from deserializing the bytes
      must take 2 inputs, the first being temporary state and the second being a
      client input. This computation will be iteratively applied on each client
      input, and the temporary state passed to each invocation will be the
      output of the previous invocation.
    serialized_temporary_state: bytes encoding a TFF `executor_pb2.Value` proto
      that will be deserialized and passed as the initial input to the
      computation.
    unencrypted_inputs: A list of bytestrings encoding TFF `executor_pb2.Value`
      protos each representing data derived from a single client. These inputs
      will be deserialized into values that can be passed as the second argument
      to the aggregation computation.

  Returns:
    A bytestring encoding a TFF `executor_pb2.Value` proto which is the
    serialized output of executing the computation iteratively on each input in
    `unencrypted_inputs` starting from the provided
    `serialized_temporary_state`.

  Raises:
    TypeError: If `serialized_client_to_server_aggregation_computation` cannot
    be deserialized into an aggregation
    computation with the expected structure, or `serialized_temporary_state` or
    `unencrypted_inputs`
    cannot be deserialized into values with types accepted by the computation
    parameters.
  """
  tff.backends.native.set_sync_local_cpp_execution_context()
  computation = _get_tff_computation(
      serialized_client_to_server_aggregation_computation
  )
  # Check that the deserialized computation has the expected types for an
  # aggregation computation.
  input_type_spec = computation.type_signature.parameter
  _check_type(
      input_type_spec, tff.StructType, name='aggregation computation input type'
  )
  if len(input_type_spec) != 2:
    raise TypeError(
        'Unexpected number of elements in the tff.StructType '
        f'{repr(input_type_spec)}'
    )
  temp_state_type, client_input_type = input_type_spec
  temp_state = _get_tff_value(
      serialized_temporary_state,
      temp_state_type,
      name='serialized temporary state',
  )
  for serialized_input_data in unencrypted_inputs:
    # Aggregation inputs should be serialized TFF values, until we move to
    # using the client report wire format everywhere.
    client_input = _get_tff_value(
        serialized_input_data, client_input_type, name='serialized client input'
    )
    temp_state = computation(temp_state, client_input)

  return _serialize_output(temp_state, computation.type_signature.result)
