// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Converts values to and from the leaves passed between Python and C++. This is
// the C++ half of the conversion; the Python `value_conversion` module is the
// Python half.
//
// A struct is passed as a list with the leaves of each element, and a federated
// value as a list with the leaves of each member (just one for an all-equal
// value). Any other value is a leaf:
//
// - A tensor of a bool or numeric dtype is passed as a C-contiguous NumPy
//   array, with no serialization. Its `federated_language.Array` holds the
//   elements in typed lists, laid out as by `federated_language.array_to_proto`
//   so that every executor accepts it. Results may also hold them in
//   `content`, as the TensorFlow executor's do. Typed lists can hold at most
//   2^31 - 1 elements (half that for complex dtypes).
// - Any other leaf (a string tensor, sequence or computation) is passed as a
//   serialized `executor_pb2.Value` (Python `bytes`), so it must be smaller
//   than the 2 GiB proto limit.
//
// Only structs and federated values are split into lists. A sequence also
// contains other values (its elements), but it is a single leaf, so each
// sequence (for example, each client's dataset) is assumed to be smaller than
// 2 GiB. If larger sequences are needed, they could be split into their
// elements in the same way.

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_VALUE_CONVERSION_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_VALUE_CONVERSION_H_

#include <pybind11/pybind11.h>

#include "federated_language/proto/computation.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::program_executor_tee {

// Builds `*out`, a value of type `type`, from `leaves` (see the Python
// `value_conversion.encode_value`). The elements of a NumPy array are copied
// once, into the typed list of the `federated_language.Array`. Throws a
// TypeError if `leaves` for a struct or federated value is not a list, if a
// leaf is neither a NumPy array nor Python `bytes`, or if the dtype of an array
// does not match its type; and a ValueError if the number of struct elements or
// the shape of an array does not match `type`, if an array is not C-contiguous
// or has more than 2^31 - 1 typed-list values, or if serialized bytes cannot be
// parsed (for example because they are 2 GiB or more).
void DecodeValue(const federated_language::Type& type, pybind11::handle leaves,
                 tensorflow_federated::v0::Value* out);

// Converts `value`, a value of type `type`, to leaves (see the Python
// `value_conversion.decode_value`). An all-equal federated value is converted
// to the leaves of its first member only. A tensor in any bool or numeric dtype
// becomes a NumPy array; other leaves (such as string tensors, sequences, or
// computations) are serialized. Each leaf is cleared once it has been
// converted, so the C++ and Python copies of the value never both exist in
// full. Throws a ValueError if the number of struct elements does not match
// `type` or if a serialized leaf cannot be serialized (for example because it
// is 2 GiB or more).
//
// The returned NumPy array is zero-copy whenever the proto's in-memory layout
// matches NumPy's:
// - For typed lists whose element size matches the proto field (bool and
//   32/64-bit integer, float, and complex types), the array takes ownership of
//   the RepeatedField and points into its buffer. For 8-bit and 16-bit integer
//   and float types, which protobuf stores widened in a `repeated int32` field,
//   elements are narrowed and copied into a new array.
// - For raw `content` (as returned by the TensorFlow executor), the array is
//   zero-copy if `content` is a single aligned chunk, and copied if unaligned
//   or split across multiple chunks. As with `tff.framework.deserialize_value`,
//   arrays made from `content` are read-only and arrays made from typed lists
//   are writable.
pybind11::object EncodeValue(const federated_language::Type& type,
                             tensorflow_federated::v0::Value& value);

}  // namespace confidential_federated_compute::program_executor_tee

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PROGRAM_CONTEXT_CC_VALUE_CONVERSION_H_
