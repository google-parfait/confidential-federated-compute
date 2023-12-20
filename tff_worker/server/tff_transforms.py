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

"""Library for executing TFF transforms on unencrypted data."""
from collections import OrderedDict
import logging
from typing import Any, List, Tuple, Type, Union
import uuid
from fcp.protos.confidentialcompute import tff_worker_configuration_pb2 as worker_pb2
from google.protobuf import message
import tensorflow as tf
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


def _check_client_input_type_and_fedsql_config_compatible(
    client_input_type: tff.types.Type,
    fed_sql_tf_checkpoint_spec: worker_pb2.TffWorkerConfiguration.ClientWork.FedSqlTensorflowCheckpoint,
) -> None:
  """Determines if the client input type matches the checkpoint specification.

  When using a FedSQL TensorFlow checkpoint as the input type, a preprocessing
  computation should be applied to the customer-provided computation making it
  compatible with the input type created from a FedSQL checkpoint. This method
  ensures that this is the case. If it fails, there is a mismatch between the
  computation generation logic and the logic in this file to convert a FedSQL
  TensorFlow checkpoint into an input to the computation.

  Args:
    client_input_type: The TFF type that is expected as the client input to the
      client work computation.
    fed_sql_tf_checkpoint_spec: The specification of the names and tensors to
      extract from a TensorFlow checkpoint produced by a FedSQL query.

  Returns:
    None if an OrderedDict created by extracting the tensors specified in the
    `fed_sql_tf_checkpoint_spec` from a serialized checkpoint can be passed to
    a computation expecting an input of the type `client_input_type`.
  Raises:
    TypeError if the OrderedDict that will be created by parsing a valid
    checkpoint using `fed_sql_tf_checkpoint_spec` will not be compatible with
    `client_input_type`.
  """
  _check_type(
      client_input_type,
      tff.FederatedType,
      name='client work computation client input type',
  )
  client_input_struct = client_input_type.member
  _check_type(
      client_input_struct,
      tff.StructType,
      name='client work computation client input type member',
  )
  expected_num_columns = len(fed_sql_tf_checkpoint_spec.fed_sql_columns)
  if len(client_input_struct) != expected_num_columns:
    raise TypeError(
        'Number of elements computation expects in client input '
        f'({len(client_input_struct)}) does not match the number of FedSql '
        f'columns provided by configuration ({expected_num_columns}).'
    )
  fedsql_type = tff.types.at_clients(
      tff.types.to_type(
          OrderedDict(
              [
                  (
                      col.name,
                      tff.TensorType(
                          tf.dtypes.as_dtype(col.data_type), shape=[None]
                      ),
                  )
                  for col in fed_sql_tf_checkpoint_spec.fed_sql_columns
              ]
          )
      )
  )
  try:
    client_input_type.check_assignable_from(fedsql_type)
  except TypeError as e:
    raise TypeError(
        'FedSql checkpoint specification incompatible with client work'
        ' computation client input type.\nFedSql checkpoint spec:'
        f' {repr(fed_sql_tf_checkpoint_spec)}\nClient work computation client'
        f' input type: {repr(client_input_type)}'
    ) from e


def _restore_tensorflow_checkpoint_to_dict(
    fed_sql_tf_checkpoint_spec: worker_pb2.TffWorkerConfiguration.ClientWork.FedSqlTensorflowCheckpoint,
    checkpoint: bytes,
) -> OrderedDict[str, tf.Tensor]:
  """Creates an OrderedDict from bytes encoding a TensorFlow checkpoint.

  Args:
    fed_sql_tf_checkpoint_spec: Specifies the names and types of the tensors to
      be retrieved from the checkpoint. Each retrieved tensor is expected to be
      a single-dimensional tensor and all tensors retrieved must have the same
      number of elements.
    checkpoint: bytes encoding a TensorFlow checkpoint from which the tensors
      specified in `fed_sql_tf_checkpoint_spec` will be retrieved.

  Returns:
    An OrderedDict from tensor names to tensors created by retrieving the
    tensors specified by the names of each column in
    `fed_sql_tf_checkpoint_spec`.

  Raises:
    TypeError if the checkpoint cannot be parsed into an OrderedDict with the
    expected column names and types specified in `fed_sql_tf_checkpoint_spec`,
    or the tensors in the checkpoint have more than one dimension, or do not all
    have the same number of elements.
  """
  names = [column.name for column in fed_sql_tf_checkpoint_spec.fed_sql_columns]
  dtypes = [
      column.data_type for column in fed_sql_tf_checkpoint_spec.fed_sql_columns
  ]
  # The checkpoint must be written to a file in order to restore the tensors
  # using TensorFlow.
  # Write to a file in TensorFlow's RamFileSystem to avoid disk I/O.
  tmp_path = f'ram://{uuid.uuid4()}.ckpt'
  try:
    with tf.io.gfile.GFile(tmp_path, 'wb') as f:
      f.write(checkpoint)
    restored_tensors = tf.raw_ops.RestoreV2(
        prefix=tmp_path,
        tensor_names=names,
        shape_and_slices=[''] * len(names),
        dtypes=dtypes,
    )
    prev_shape = None
    for tensor in restored_tensors:
      if len(tensor.shape.dims) != 1:
        raise TypeError(
            'Could not parse FedSql checkpoint with expected columns'
            f' {repr(fed_sql_tf_checkpoint_spec.fed_sql_columns)}.\nAll tensors'
            ' must be one-dimensional but found tensor with dimensions'
            f' {repr(tensor.shape.dims)}'
        )
      if prev_shape is not None:
        if prev_shape != tensor.shape:
          raise TypeError(
              'Could not parse FedSql checkpoint with expected columns'
              f' {repr(fed_sql_tf_checkpoint_spec.fed_sql_columns)}.\nShapes of'
              ' all tensors must match but found tensors with different'
              f' shapes: {repr(prev_shape)} and {repr(tensor.shape)}'
          )
      prev_shape = tensor.shape
    return OrderedDict(
        [(names[i], restored_tensors[i]) for i in range(len(names))]
    )
  except TypeError:
    raise
  except Exception as e:
    raise TypeError(
        'Could not parse FedSql checkpoint with expected columns'
        f' {repr(fed_sql_tf_checkpoint_spec.fed_sql_columns)}'
    ) from e
  finally:
    tf.io.gfile.remove(tmp_path)


def _join_clients_placed_inputs(inputs: List[Any], type: tff.Type) -> Any:
  """Joins all client inputs into a single object containing data to aggregate.

  TFF encodes clients-placed values which are not all equal as lists, even if
  they have only a single client value, which is the case in Trusted Brella
  since the client_work computation is executed independently on each client.
  Thus, the code must join the singleton lists for each client together to
  obtain a single object containing all the data to aggregate to obtain an input
  with the right type to pass to the aggregation computation. However, the
  `tff.types.FederatedType` that encodes the client placement is not necessarily
  at the top level- the client input type may contain nested
  `tff.types.StructType` objects. In that case, it is the lists nested within
  the structs that need to be joined together. Structs may be named or unnamed.


  Args:
    inputs: A list of client inputs of the TFF type specified by `type`.
    type: The TFF type of each of the client inputs in the list and also of the
      output that this function will generate containing the data from all
      client inputs.

  Returns:
    An object of TFF type `type` which contains the data from all of the client
    inputs in the `inputs` list.

  Raises:
    TypeError if the TFF type is not a `tff.types.FederatedType` or nested
      layers of `tff.types.StructType` types for which each branch contains a
      `tff.types.FederatedType`. There may be other types nested within each
      `tff.types.FederatedType`, but no other types should be present above the
      `tff.types.FederatedType`. A TypeError is also raised if any
      `tff.types.FederatedType` encountered is not placed @CLIENTS or has
      all_equal=True.
  """
  if isinstance(type, tff.FederatedType):
    if type.placement is not tff.CLIENTS:
      raise TypeError(
          'Expected client input type to contain only CLIENTS-placed inputs.'
      )
    if type.all_equal:
      raise TypeError(
          'Expected client input type to contain CLIENTS-placed inputs that are'
          ' not all equal.'
      )
    result = []
    for input in inputs:
      _check_type(input, list, name='client input')
      result.extend(input)
    return result
  elif isinstance(type, tff.StructType):
    result = None
    for index, (name, element) in enumerate(tff.structure.to_elements(type)):
      if name is not None:
        if result is None:
          result = OrderedDict()
        inputs_for_element = [input[name] for input in inputs]
        result[name] = _join_clients_placed_inputs(
            inputs_for_element, type=element
        )
      else:
        if result is None:
          result = []
        inputs_for_element = [input[index] for input in inputs]
        result.append(
            _join_clients_placed_inputs(inputs_for_element, type=element)
        )
    if isinstance(result, list):
      return tuple(result)
    return result
  else:
    raise TypeError(
        'Unexpected client input type. Expected only StructTypes to wrap a'
        ' FederatedType of placement CLIENTS and all_equal=False.'
    )


def perform_client_work(
    client_work_config: worker_pb2.TffWorkerConfiguration.ClientWork,
    unencrypted_input: bytes,
) -> bytes:
  """Transforms an unencrypted client input using a serialized TFF computation.

  Args:
    client_work_config: A
      `tff_worker_configuration_pb2.TffWorkerConfiguration.ClientWork` message.
      This message contains bytes encoding a TFF `computation_pb2.Computation`
      proto which determines how to perform the transformation. The
      `tff.Computation` resulting from deserializing the bytes must take 2
      inputs, the first being a client input and the second being broadcasted
      data. The computation must accept the client input as an OrderedDict
      containing mappings from strings to tensors. The configuration also
      includes bytes encoding a TFF `executor_pb2.Value` proto that will be
      deserialized and passed as the broadcasted data input to the computation.
      Finally, the computation includes a `client_input_format` which gives this
      method the necessary instructions for parsing the `unencrypted_input`
      bytes into an OrderedDict that can be passed to the conputation.
    unencrypted_input: A bytestring that will be parsed into an OrderedDict from
      tensor names to tensors according to the
      `client_work_config.client_input_format` specification.

  Returns:
    A bytestring encoding a TFF `executor_pb2.Value` proto which is the
    serialized output of executing the computation on the parsed
    `unencrypted_input` and the deserialized broadcasted data.

  Raises:
    TypeError: If
    `client_work_config.serialized_client_work_computation`
    cannot be deserialized into a computation with the expected
    structure, or `client_work_config.serialized_broadcasted_data` or
    `unencrypted_input` cannot be deserialized into values with types accepted
    by the computation parameters.
  """
  tff.backends.native.set_sync_local_cpp_execution_context()
  computation = _get_tff_computation(
      client_work_config.serialized_client_work_computation
  )
  logging.info(
      'Client work computation type signature is: %s',
      computation.type_signature.formatted_representation(),
  )
  # Check that the deserialized computation has the expected types for a
  # client work computation.
  input_type_spec = computation.type_signature.parameter
  _check_type(
      input_type_spec, tff.StructType, name='client work computation input type'
  )
  if len(input_type_spec) != 2:
    raise TypeError(
        'Unexpected number of elements in the tff.StructType '
        f'{repr(input_type_spec)}'
    )
  client_input_type, broadcasted_data_type = input_type_spec
  broadcasted_data = _get_tff_value(
      client_work_config.serialized_broadcasted_data,
      broadcasted_data_type,
      name='serialized broadcasted data',
  )
  if not client_work_config.HasField('fed_sql_tensorflow_checkpoint'):
    raise ValueError(
        'Unknown client input format:'
        f' {client_work_config.WhichOneof("client_input_format")}'
    )

  _check_client_input_type_and_fedsql_config_compatible(
      client_input_type, client_work_config.fed_sql_tensorflow_checkpoint
  )
  client_input_dict = _restore_tensorflow_checkpoint_to_dict(
      client_work_config.fed_sql_tensorflow_checkpoint, unencrypted_input
  )
  logging.info('Executing client work computation on 1 input')
  # Wrap the client input as a list as TFF expects clients-placed inputs that
  # are not all equal to be in a list form.
  output = computation([client_input_dict], broadcasted_data)

  logging.info(
      'Serializing client work computation output to type signature %s',
      repr(computation.type_signature.result),
  )
  return _serialize_output(output, computation.type_signature.result)


def aggregate(
    aggregation_config: worker_pb2.TffWorkerConfiguration.Aggregation,
    unencrypted_inputs: List[bytes],
) -> bytes:
  """Aggregates unencrypted inputs using a serialized TFF computation.

  Args:
    aggregation_config: A
      `tff_worker_configuration_pb2.TffWorkerConfiguration.Aggregation` message.
      This message contains bytes encoding a TFF `computation_pb2.Computation`
      proto which determines how to perform the aggregation. The
      `tff.Computation` resulting from deserializing the bytes must take 2
      inputs, the first being temporary state and the second being a client
      input. This computation will be iteratively applied on each client input,
      and the temporary state passed to each invocation will be the output of
      the previous invocation. The configuration also includes bytes encoding a
      TFF `executor_pb2.Value` proto that will be deserialized and passed as the
      initial temporary state input to the computation.
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
    TypeError: If
    `aggregation_config.serialized_client_to_server_aggregation_computation`
    cannot be deserialized into an aggregation computation with the expected
    structure, or `aggregation_config.serialized_temporary_state` or
    `unencrypted_inputs` cannot be deserialized into values with types accepted
    by the computation parameters.
  """
  tff.backends.native.set_sync_local_cpp_execution_context()
  computation = _get_tff_computation(
      aggregation_config.serialized_client_to_server_aggregation_computation
  )
  logging.info(
      'Aggregation computation type signature is: %s',
      computation.type_signature.formatted_representation(),
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
      aggregation_config.serialized_temporary_state,
      temp_state_type,
      name='serialized temporary state',
  )

  client_inputs = []
  for serialized_input_data in unencrypted_inputs:
    # Aggregation inputs should be serialized TFF values, until we move to
    # using the client report wire format everywhere.
    client_input = _get_tff_value(
        serialized_input_data, client_input_type, name='serialized client input'
    )
    client_inputs.append(client_input)
  logging.info(
      'Joining client inputs to pass to aggregation computation. Client input'
      ' type: %s',
      repr(client_input_type),
  )
  joined_inputs = _join_clients_placed_inputs(client_inputs, client_input_type)

  logging.info(
      'Executing aggregation computation on %d inputs', len(client_inputs)
  )
  updated_server_state = computation(temp_state, joined_inputs)

  logging.info(
      'Serializing aggregation computation output to type signature %s',
      repr(computation.type_signature.result),
  )
  return _serialize_output(
      updated_server_state, computation.type_signature.result
  )
