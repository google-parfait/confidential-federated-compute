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
"""Round-trip tests for value_conversion.py and cc/value_conversion.h."""

import collections

from absl.testing import absltest
import federated_language
import ml_dtypes
import numpy as np
from program_executor_tee.program_context import value_conversion
from program_executor_tee.program_context.cc import value_conversion_test_bindings
import tensorflow_federated as tff

# The dtypes of tensors that are passed as NumPy arrays.
_NUMPY_DTYPES = (
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
    ml_dtypes.bfloat16,
)

# Scalars, empty arrays, and arrays of one or more dimensions.
_SHAPES = ((), (0,), (2, 0), (5,), (2, 3), (2, 1, 3))


def _round_trip(value, type_spec):
  """Sends `value` from Python to C++ and back, with no executor in between."""
  leaves = value_conversion.encode_value(value, type_spec)
  leaves = value_conversion_test_bindings.round_trip(
      type_spec.to_proto().SerializeToString(), leaves
  )
  return value_conversion.decode_value(leaves, type_spec)


def _serialize_type(type_spec):
  return type_spec.to_proto().SerializeToString()


def _make_value(dtype, shape):
  """Returns a value of `dtype` and `shape`, as a NumPy scalar for scalars."""
  rng = np.random.default_rng(seed=len(shape) + np.dtype(dtype).num)
  size = int(np.prod(shape))
  if dtype is np.bool_:
    flat = rng.integers(0, 2, size).astype(np.bool_)
  elif np.issubdtype(dtype, np.integer):
    info = np.iinfo(dtype)
    flat = rng.integers(info.min, info.max, size, dtype=dtype, endpoint=True)
  elif np.issubdtype(dtype, np.complexfloating):
    flat = (rng.normal(size=size) + 1j * rng.normal(size=size)).astype(dtype)
  else:
    flat = (rng.normal(size=size) * 100).astype(dtype)
  # For scalar shapes `()`, indexing `flat[0]` returns a NumPy scalar
  # (`np.generic`) rather than a 0-D `np.ndarray`.
  return flat[0] if not shape else flat.reshape(shape)


class ValueConversionTest(absltest.TestCase):

  def assert_same_tensor(self, actual, expected):
    """Asserts that two tensors have the same type, dtype, shape, and values."""
    self.assertIs(type(actual), type(expected))
    self.assertEqual(actual.dtype, expected.dtype)
    np.testing.assert_array_equal(actual, expected)

  def test_struct(self):
    type_spec = federated_language.StructType([
        ("x", federated_language.TensorType(np.int32)),
        ("y", federated_language.TensorType(np.int32)),
    ])
    self.assertEqual(_round_trip((10, 20), type_spec), {"x": 10, "y": 20})

  def test_namedtuple(self):
    TestValue = collections.namedtuple("TestValue", ["x", "y"])
    type_spec = federated_language.StructWithPythonType(
        elements=[
            ("x", federated_language.TensorType(np.int32)),
            ("y", federated_language.TensorType(np.float32)),
        ],
        container_type=TestValue,
    )
    value = TestValue(x=1, y=0.5)
    result = _round_trip(value, type_spec)
    self.assertIsInstance(result, TestValue)
    self.assertEqual(result, value)

  def test_federated_clients_value(self):
    type_spec = federated_language.FederatedType(
        np.int32, federated_language.CLIENTS
    )
    value = [1, 2, 3]
    self.assertEqual(_round_trip(value, type_spec), value)

  def test_federated_struct_values(self):
    type_spec = federated_language.FederatedType(
        federated_language.StructType([
            ("a", federated_language.TensorType(np.int32)),
            (
                "b",
                federated_language.StructType([
                    federated_language.TensorType(np.float32),
                    federated_language.TensorType(np.int32),
                ]),
            ),
        ]),
        federated_language.CLIENTS,
    )
    value = [{"a": 1, "b": [2.0, 3]}, {"a": 4, "b": [5.0, 6]}]
    self.assertEqual(_round_trip(value, type_spec), value)

  def test_federated_server_value(self):
    # SERVER values are all-equal, so only their single member is sent.
    TestState = collections.namedtuple("TestState", ["weights", "step"])
    type_spec = federated_language.FederatedType(
        federated_language.StructWithPythonType(
            elements=[
                ("weights", federated_language.TensorType(np.float32, [2])),
                ("step", federated_language.TensorType(np.int32)),
            ],
            container_type=TestState,
        ),
        federated_language.SERVER,
    )
    value = TestState(weights=np.array([1.0, 2.0], np.float32), step=7)
    result = _round_trip(value, type_spec)
    self.assertIsInstance(result, TestState)
    np.testing.assert_array_equal(result.weights, value.weights)
    self.assertEqual(result.step, value.step)

  def test_federated_sequences(self):
    type_spec = federated_language.FederatedType(
        federated_language.SequenceType(
            federated_language.StructType([
                ("x", federated_language.TensorType(np.int32)),
                ("y", federated_language.TensorType(np.float32, [2])),
            ])
        ),
        federated_language.CLIENTS,
    )
    value = [
        [{"x": 1, "y": [1.0, 2.0]}, {"x": 2, "y": [3.0, 4.0]}],
        [],
        [{"x": 3, "y": [5.0, 6.0]}],
    ]
    np.testing.assert_equal(_round_trip(value, type_spec), value)

  def test_tensors_are_sent_as_numpy_arrays(self):
    type_spec = federated_language.StructType([
        ("weights", federated_language.TensorType(np.float32, [2, 3])),
        ("step", federated_language.TensorType(np.int64)),
        ("mask", federated_language.TensorType(np.bool_, [None])),
        ("name", federated_language.TensorType(np.str_)),
    ])
    weights = np.arange(6, dtype=np.float32).reshape([2, 3])
    value = (weights, 7, [True, False, True], b"abc")

    leaves = value_conversion.encode_value(value, type_spec)

    weights_leaf, step_leaf, mask_leaf, name_leaf = leaves
    # A C-contiguous array of the right dtype is sent as is, without a copy.
    self.assertIs(weights_leaf, weights)
    self.assertIsInstance(step_leaf, np.ndarray)
    self.assertEqual(step_leaf.dtype, np.int64)
    self.assertEqual(step_leaf.shape, ())
    self.assertIsInstance(mask_leaf, np.ndarray)
    self.assertEqual(mask_leaf.dtype, np.bool_)
    self.assertEqual(mask_leaf.shape, (3,))
    # Strings are still serialized.
    self.assertIsInstance(name_leaf, bytes)

    result_leaves = value_conversion_test_bindings.round_trip(
        _serialize_type(type_spec), leaves
    )
    np.testing.assert_equal(
        value_conversion.decode_value(result_leaves, type_spec),
        {"weights": weights, "step": 7, "mask": value[2], "name": b"abc"},
    )

  def test_non_contiguous_arrays_are_copied_once(self):
    type_spec = federated_language.TensorType(np.float32, [2, 2])
    value = np.arange(8, dtype=np.float32).reshape([2, 4])[:, ::2]
    self.assertFalse(value.flags.c_contiguous)

    leaf = value_conversion.encode_value(value, type_spec)

    self.assertTrue(leaf.flags.c_contiguous)
    np.testing.assert_array_equal(leaf, value)
    np.testing.assert_array_equal(_round_trip(value, type_spec), value)

  def test_numpy_arrays_match_tff_serialization(self):
    for dtype in _NUMPY_DTYPES:
      for shape in _SHAPES:
        with self.subTest(dtype=np.dtype(dtype).name, shape=shape):
          type_spec = federated_language.TensorType(dtype, shape)
          serialized_type = _serialize_type(type_spec)
          value = _make_value(dtype, shape)
          value_pb, _ = tff.framework.serialize_value(value, type_spec)
          serialized_value = value_pb.SerializeToString()
          expected, _ = tff.framework.deserialize_value(value_pb, type_spec)

          # Python to C++: the C++ proto is the same as TFF's.
          leaf = value_conversion.encode_value(value, type_spec)
          self.assertIsInstance(leaf, np.ndarray)
          self.assertEqual(
              value_conversion_test_bindings.serialize(serialized_type, leaf),
              serialized_value,
          )

          # C++ to Python: the value is the same as TFF deserializes from the
          # same proto.
          leaf = value_conversion_test_bindings.round_trip(
              serialized_type, serialized_value
          )
          self.assertIsInstance(leaf, np.ndarray)
          self.assert_same_tensor(
              value_conversion.decode_value(leaf, type_spec), expected
          )

          # Both ways.
          self.assert_same_tensor(_round_trip(value, type_spec), expected)

  def test_numpy_arrays_from_content_match_tff_deserialization(self):
    # Results of the TensorFlow executor hold their elements in `content`.
    for dtype in _NUMPY_DTYPES:
      for shape in _SHAPES:
        with self.subTest(dtype=np.dtype(dtype).name, shape=shape):
          type_spec = federated_language.TensorType(dtype, shape)
          value_pb = tff.framework.serialize_value(
              federated_language.array_to_proto_content(
                  np.asarray(_make_value(dtype, shape)), dtype_hint=dtype
              ),
              type_spec,
          )[0]
          self.assertTrue(value_pb.array.HasField("content"))
          expected, _ = tff.framework.deserialize_value(value_pb, type_spec)

          leaf = value_conversion_test_bindings.round_trip(
              _serialize_type(type_spec), value_pb.SerializeToString()
          )
          self.assertIsInstance(leaf, np.ndarray)
          self.assert_same_tensor(
              value_conversion.decode_value(leaf, type_spec), expected
          )

  def test_unsupported_dtypes_are_serialized(self):
    type_spec = federated_language.TensorType(np.str_, [2])
    value = np.array([b"a", b"bc"], np.object_)
    leaf = value_conversion.encode_value(value, type_spec)
    self.assertIsInstance(leaf, bytes)
    leaf = value_conversion_test_bindings.round_trip(
        _serialize_type(type_spec), leaf
    )
    self.assertIsInstance(leaf, bytes)
    np.testing.assert_array_equal(
        value_conversion.decode_value(leaf, type_spec), value
    )

  def test_array_protos_are_serialized(self):
    # Like `tff.framework.serialize_value`, accept `array_pb2.Array` protos for
    # tensors.
    type_spec = federated_language.TensorType(np.int32, [2])
    array_pb = federated_language.array_to_proto(np.array([1, 2], np.int32))
    leaf = value_conversion.encode_value(array_pb, type_spec)
    self.assertIsInstance(leaf, bytes)
    np.testing.assert_array_equal(
        _round_trip(array_pb, type_spec), np.array([1, 2], np.int32)
    )

  def test_python_values_are_checked_like_tff_serialization(self):
    for value, type_spec in (
        # Wrong dtypes.
        (
            np.array([1.0, 2.0], np.float64),
            federated_language.TensorType(np.float32, [2]),
        ),
        (np.int64(1), federated_language.TensorType(np.int32)),
        ([1, 2.5], federated_language.TensorType(np.int32, [2])),
        (1, federated_language.TensorType(np.float32)),
        (b"abc", federated_language.TensorType(np.int32)),
        # Wrong shapes.
        (
            np.array([1, 2, 3], np.int32),
            federated_language.TensorType(np.int32, [2]),
        ),
        (np.int32(1), federated_language.TensorType(np.int32, [1])),
        ([[1, 2]], federated_language.TensorType(np.int32, [2])),
    ):
      with self.subTest(value=value, type_spec=type_spec):
        with self.assertRaisesRegex(TypeError, "Failed to serialize"):
          tff.framework.serialize_value(value, type_spec)
        with self.assertRaisesRegex(TypeError, "Failed to serialize"):
          value_conversion.encode_value(value, type_spec)

  def test_arrays_that_do_not_match_their_type_raise(self):
    type_spec = federated_language.TensorType(np.int32, [2, None])
    serialized_type = _serialize_type(type_spec)
    with self.assertRaisesRegex(
        TypeError, "Expected a NumPy array of dtype int32, found dtype int64"
    ):
      value_conversion_test_bindings.round_trip(
          serialized_type, np.zeros([2, 3], np.int64)
      )
    with self.assertRaisesRegex(
        TypeError, "Expected a NumPy array of dtype int32, found dtype >i4"
    ):
      value_conversion_test_bindings.round_trip(
          serialized_type, np.zeros([2, 3], ">i4")
      )
    with self.assertRaisesRegex(
        ValueError,
        r"Expected a NumPy array of shape \(2, None\), found shape \(3, 2\)",
    ):
      value_conversion_test_bindings.round_trip(
          serialized_type, np.zeros([3, 2], np.int32)
      )
    with self.assertRaisesRegex(ValueError, r"found shape \(2,\)"):
      value_conversion_test_bindings.round_trip(
          serialized_type, np.zeros([2], np.int32)
      )
    with self.assertRaisesRegex(ValueError, "Expected a C-contiguous"):
      value_conversion_test_bindings.round_trip(
          serialized_type, np.zeros([3, 2], np.int32).T
      )
    string_type = federated_language.TensorType(np.str_)
    with self.assertRaisesRegex(
        TypeError, "Tensors of dtype DT_STRING must be passed as serialized"
    ):
      value_conversion_test_bindings.round_trip(
          _serialize_type(string_type), np.array(b"abc")
      )

  def test_leaves_that_are_not_serialized_values_raise(self):
    type_spec = federated_language.TensorType(np.int32)
    serialized_type = type_spec.to_proto().SerializeToString()
    leaf_pb, _ = tff.framework.serialize_value(1, type_spec)
    # Leaves must be arrays or bytes, so even an unserialized proto is rejected.
    for leaf in (None, "not bytes", leaf_pb, np.int32(1)):
      with self.assertRaisesRegex(TypeError, "is not an instance of 'bytes'"):
        value_conversion_test_bindings.round_trip(serialized_type, leaf)
    with self.assertRaisesRegex(
        ValueError, "Failed to parse serialized tensorflow_federated.v0.Value"
    ):
      value_conversion_test_bindings.round_trip(serialized_type, b"not a proto")

  def test_struct_and_federated_type_mismatches_raise(self):
    struct_type = federated_language.StructType([
        ("x", federated_language.TensorType(np.int32)),
        ("y", federated_language.TensorType(np.int32)),
    ])
    for wrong_struct in ((1,), (1, 2, 3)):
      with self.assertRaisesRegex(TypeError, "Cannot serialize a struct"):
        value_conversion.encode_value(wrong_struct, struct_type)
    serialized_struct_type = _serialize_type(struct_type)
    for wrong_leaves in ([np.array(1, np.int32)], [np.array(1, np.int32)] * 3):
      with self.assertRaisesRegex(
          ValueError, "Expected 2 struct elements, found"
      ):
        value_conversion_test_bindings.round_trip(
            serialized_struct_type, wrong_leaves
        )

    clients_type = federated_language.FederatedType(
        np.str_, federated_language.CLIENTS
    )
    for non_sequence in ("abc", b"abc", np.array([b"a", b"b"], np.object_)):
      with self.assertRaisesRegex(TypeError, "Expected non-all-equal value"):
        value_conversion.encode_value(non_sequence, clients_type)


if __name__ == "__main__":
  absltest.main()
