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
"""Converts values to and from the leaves passed to `ComputationRunner`.

Values cross between Python and C++ as nested lists of leaves (tensors,
sequences, or computations). A tensor of a bool or numeric dtype is passed as a
C-contiguous NumPy array, with no serialization. Any other leaf (a string
tensor, sequence or computation) is passed as a serialized `executor_pb2.Value`,
so it must be smaller than the 2 GiB proto limit. The total size of a value is
not limited. This is the Python half of the conversion; `cc/value_conversion.h`
is the C++ half.

Only structs and federated values are split into lists. A sequence also
contains other values (its elements), but it is a single leaf, so each sequence
(for example, each client's dataset) is assumed to be smaller than 2 GiB. If
larger sequences are needed, they could be split into their elements in the
same way.
"""

from collections.abc import Sequence
import typing

import federated_language
import numpy as np
import tensorflow_federated as tff
from tensorflow_federated.proto.v0 import executor_pb2

# The Python types of the values that `tff.framework.serialize_value` handles as
# arrays (besides sequences of them).
_ARRAY_TYPES = typing.get_args(federated_language.Array)


def encode_value(val: object, type_spec: federated_language.Type) -> object:
  """Encodes `val` as the leaves expected by `ComputationRunner.execute`.

  Structs and federated values become lists with the leaves of their elements or
  members (just one for an all-equal value). Any other value is a leaf: a
  C-contiguous NumPy array for a tensor of a bool or numeric dtype, and a
  serialized `executor_pb2.Value` otherwise.

  Raises:
    TypeError: If a tensor does not match its type, as
      `tff.framework.serialize_value` would.
  """
  if isinstance(type_spec, federated_language.StructType):
    elements = federated_language.common_libs.structure.from_container(val)
    if len(elements) != len(type_spec):
      raise TypeError(
          f"Cannot serialize a struct value of {len(elements)} elements to a"
          f" struct type requiring {len(type_spec)} elements."
      )
    return [
        encode_value(elem_val, elem_type)
        for elem_val, elem_type in zip(elements, type_spec, strict=True)
    ]
  if isinstance(type_spec, federated_language.FederatedType):
    if type_spec.all_equal:
      members = [val]
    elif isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
      members = val
    else:
      raise TypeError(
          f"Expected non-all-equal value with placement {type_spec.placement}"
          f" to be a sequence, found {type(val)}."
      )
    return [encode_value(member, type_spec.member) for member in members]
  if (
      isinstance(type_spec, federated_language.TensorType)
      and type_spec.dtype.type is not np.str_
      and _is_array_like(val)
  ):
    return _to_numpy(val, type_spec)
  value_pb, _ = tff.framework.serialize_value(val, type_spec)
  return value_pb.SerializeToString()


def _is_array_like(val: object) -> bool:
  """Returns whether `tff.framework.serialize_value` handles `val` as an array.

  Matches the types handled by TFF's `_serialize_tensor_value` (arrays, scalars,
  or sequences of them). Other values for tensor types, such as
  `array_pb2.Array`
  protos, are serialized as before.
  """
  return isinstance(val, _ARRAY_TYPES) or (
      isinstance(val, Sequence) and not isinstance(val, (str, bytes))
  )


def _flatten(val: object):
  """Yields the leaf elements of a (nested) sequence."""
  if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
    for item in val:
      yield from _flatten(item)
  else:
    yield val


def _to_numpy(
    val: object, type_spec: federated_language.TensorType
) -> np.ndarray:
  """Returns `val` as a C-contiguous NumPy array of `type_spec.dtype`.

  Matches the validation and coercion logic from TFF's internal
  `tensorflow_federated.python.core.impl.executors.value_serialization._serialize_tensor_value`,
  accepting the same values with the same checks, but returns a NumPy array
  directly instead of converting to an `array_pb2.Array` proto. Copies `val`
  only
  if it is not already a C-contiguous array of the required dtype.

  Args:
    val: A NumPy array or scalar, a Python scalar, or a (nested) sequence of
      them.
    type_spec: The type of `val`.
  """
  dtype = type_spec.dtype.type
  # Coerce Python lists and tuples (instances of collections.abc.Sequence) to a
  # NumPy array, matching TFF's behavior for values passed as sequences.
  if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
    if not all(
        federated_language.array_is_compatible_dtype(x, dtype)
        for x in _flatten(val)
    ):
      raise TypeError(f"Failed to serialize the value {val} to dtype {dtype}.")
    val = np.asarray(val, dtype)
  if not federated_language.array_is_compatible_shape(val, type_spec.shape):
    shape = val.shape if isinstance(val, (np.ndarray, np.generic)) else ()
    raise TypeError(
        f"Failed to serialize the value {val} with shape {shape} to shape"
        f" {type_spec.shape}."
    )
  if not federated_language.array_is_compatible_dtype(val, dtype):
    val_dtype = (
        val.dtype.type
        if isinstance(val, (np.ndarray, np.generic))
        else type(val)
    )
    raise TypeError(
        f"Failed to serialize the value {val} of dtype {val_dtype} to dtype"
        f" {dtype}."
    )
  # Not `np.ascontiguousarray`, which turns scalars into arrays of shape [1].
  return np.asarray(val, dtype=dtype, order="C")


def decode_value(leaves: object, type_spec: federated_language.Type) -> object:
  """Decodes a value from the leaves returned by `ComputationRunner.execute`.

  The Python containers of structs (such as namedtuples) are restored from
  `type_spec`. Tensors are the same as `tff.framework.deserialize_value`
  returns.
  """
  return federated_language.framework.to_structure_with_type(
      _deserialize_leaves(leaves, type_spec), type_spec
  )


def _deserialize_leaves(
    leaves: object, type_spec: federated_language.Type
) -> object:
  """Deserializes `leaves`, with structs and federated values as lists.

  All-equal federated values become their single member instead.
  """
  if isinstance(type_spec, federated_language.StructType):
    return [
        _deserialize_leaves(elem_leaves, elem_type)
        for elem_leaves, elem_type in zip(leaves, type_spec, strict=True)
    ]
  if isinstance(type_spec, federated_language.FederatedType):
    members = [
        _deserialize_leaves(member_leaves, type_spec.member)
        for member_leaves in leaves
    ]
    return members[0] if type_spec.all_equal else members
  if isinstance(leaves, np.ndarray):
    # A tensor. Like `tff.framework.deserialize_value`, return scalars as NumPy
    # scalars rather than 0-d arrays.
    return leaves[()] if leaves.ndim == 0 else leaves
  value, _ = tff.framework.deserialize_value(
      executor_pb2.Value.FromString(leaves), type_spec
  )
  return value
