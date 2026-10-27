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

#include "program_executor_tee/program_context/cc/value_conversion.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "federated_language/proto/array.pb.h"
#include "federated_language/proto/computation.pb.h"
#include "federated_language/proto/data_type.pb.h"
#include "google/protobuf/repeated_field.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::program_executor_tee {

namespace py = pybind11;

using ::federated_language::Array;
using ::federated_language::DataType;
using ::google::protobuf::RepeatedField;
using ::tensorflow_federated::v0::Value;

namespace {

py::dtype BFloat16Dtype() {
  return py::dtype::from_args(
      py::module_::import("ml_dtypes").attr("bfloat16"));
}

// Maps a federated_language DataType to the corresponding NumPy dtype.
std::optional<py::dtype> DtypeForDataType(DataType dtype) {
  switch (dtype) {
    case federated_language::DT_BOOL:
      return py::dtype::of<bool>();
    case federated_language::DT_INT8:
      return py::dtype::of<int8_t>();
    case federated_language::DT_INT16:
      return py::dtype::of<int16_t>();
    case federated_language::DT_INT32:
      return py::dtype::of<int32_t>();
    case federated_language::DT_INT64:
      return py::dtype::of<int64_t>();
    case federated_language::DT_UINT8:
      return py::dtype::of<uint8_t>();
    case federated_language::DT_UINT16:
      return py::dtype::of<uint16_t>();
    case federated_language::DT_UINT32:
      return py::dtype::of<uint32_t>();
    case federated_language::DT_UINT64:
      return py::dtype::of<uint64_t>();
    case federated_language::DT_HALF:
      return py::dtype("float16");
    case federated_language::DT_FLOAT:
      return py::dtype::of<float>();
    case federated_language::DT_DOUBLE:
      return py::dtype::of<double>();
    case federated_language::DT_COMPLEX64:
      return py::dtype::of<std::complex<float>>();
    case federated_language::DT_COMPLEX128:
      return py::dtype::of<std::complex<double>>();
    case federated_language::DT_BFLOAT16:
      return BFloat16Dtype();
    default:
      return std::nullopt;
  }
}

std::string ToString(py::handle object) {
  return py::str(object).cast<std::string>();
}

// Returns the shape of `type` as a Python tuple string, e.g. "(2,)" or "(None,
// 3)".
std::string ShapeString(const federated_language::TensorType& type) {
  return absl::StrCat("(",
                      absl::StrJoin(type.dims(), ", ",
                                    [](std::string* out, int64_t dim) {
                                      if (dim < 0) {
                                        absl::StrAppend(out, "None");
                                      } else {
                                        absl::StrAppend(out, dim);
                                      }
                                    }),
                      type.dims_size() == 1 ? ",)" : ")");
}

// Validates that `array` matches `expected_dtype` and `type` shape, and is
// C-contiguous.
void CheckArray(const py::array& array, const py::dtype& expected_dtype,
                const federated_language::TensorType& type) {
  if (!array.dtype().equal(expected_dtype)) {
    throw py::type_error(absl::StrCat(
        "Expected a NumPy array of dtype ", ToString(expected_dtype),
        ", found dtype ", ToString(array.dtype()), "."));
  }
  if (!type.unknown_rank()) {
    bool compatible = array.ndim() == type.dims_size();
    for (int i = 0; compatible && i < type.dims_size(); ++i) {
      compatible = type.dims(i) < 0 || type.dims(i) == array.shape(i);
    }
    if (!compatible) {
      throw py::value_error(absl::StrCat("Expected a NumPy array of shape ",
                                         ShapeString(type), ", found shape ",
                                         ToString(array.attr("shape")), "."));
    }
  }
  if ((array.flags() & py::array::c_style) == 0) {
    throw py::value_error("Expected a C-contiguous NumPy array.");
  }
}

// Copies `array` directly into protobuf `list` via memcpy.
// Used for types whose element size matches the protobuf RepeatedField (bool
// and 32/64-bit integer, float, and complex types).
template <typename ProtoFieldT>
void CopyDirect(const py::array& array, RepeatedField<ProtoFieldT>* list) {
  const size_t num_values =
      static_cast<size_t>(array.nbytes()) / sizeof(ProtoFieldT);
  if (num_values > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw py::value_error(
        "A tensor is too large: protobuf typed lists hold at most 2^31 - 1 "
        "values.");
  }
  const int list_size = static_cast<int>(num_values);
  list->Reserve(list_size);
  ProtoFieldT* dst = list->AddNAlreadyReserved(list_size);
  if (num_values == 0) return;
  std::memcpy(dst, array.data(), array.nbytes());
}

// Widens elements of `array` into 32-bit integers in protobuf `list`.
// Used for 8-bit and 16-bit integer and float types, which protobuf stores in a
// `repeated int32` field.
template <typename NativeT>
void CopyWidened(const py::array& array, RepeatedField<int32_t>* list) {
  const size_t num_elements = static_cast<size_t>(array.size());
  if (num_elements > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw py::value_error(
        "A tensor is too large: protobuf typed lists hold at most 2^31 - 1 "
        "values.");
  }
  const int list_size = static_cast<int>(num_elements);
  list->Reserve(list_size);
  int32_t* dst = list->AddNAlreadyReserved(list_size);
  if (num_elements == 0) return;
  const char* src = static_cast<const char*>(array.data());
  for (size_t i = 0; i < num_elements; ++i) {
    NativeT element;
    std::memcpy(&element, src + i * sizeof(NativeT), sizeof(NativeT));
    dst[i] = static_cast<int32_t>(element);
  }
}

// Converts `array`, a NumPy tensor of type `type`, into protobuf `out`.
void DecodeArray(const federated_language::TensorType& type,
                 const py::array& array, Array* out) {
  std::optional<py::dtype> expected_dtype = DtypeForDataType(type.dtype());
  if (!expected_dtype.has_value()) {
    throw py::type_error(absl::StrCat(
        "Tensors of dtype ", federated_language::DataType_Name(type.dtype()),
        " must be passed as serialized tensorflow_federated.v0.Value bytes, "
        "not as NumPy arrays."));
  }
  CheckArray(array, *expected_dtype, type);

  out->set_dtype(type.dtype());
  auto* dims = out->mutable_shape()->mutable_dim();
  for (py::ssize_t i = 0; i < array.ndim(); ++i) {
    dims->Add(array.shape(i));
  }

  switch (type.dtype()) {
    case federated_language::DT_BOOL:
      CopyDirect(array, out->mutable_bool_list()->mutable_value());
      break;
    case federated_language::DT_INT8:
      CopyWidened<int8_t>(array, out->mutable_int8_list()->mutable_value());
      break;
    case federated_language::DT_INT16:
      CopyWidened<int16_t>(array, out->mutable_int16_list()->mutable_value());
      break;
    case federated_language::DT_INT32:
      CopyDirect(array, out->mutable_int32_list()->mutable_value());
      break;
    case federated_language::DT_INT64:
      CopyDirect(array, out->mutable_int64_list()->mutable_value());
      break;
    case federated_language::DT_UINT8:
      CopyWidened<uint8_t>(array, out->mutable_uint8_list()->mutable_value());
      break;
    case federated_language::DT_UINT16:
      CopyWidened<uint16_t>(array, out->mutable_uint16_list()->mutable_value());
      break;
    case federated_language::DT_UINT32:
      CopyDirect(array, out->mutable_uint32_list()->mutable_value());
      break;
    case federated_language::DT_UINT64:
      CopyDirect(array, out->mutable_uint64_list()->mutable_value());
      break;
    case federated_language::DT_HALF:
      CopyWidened<uint16_t>(array,
                            out->mutable_float16_list()->mutable_value());
      break;
    case federated_language::DT_FLOAT:
      CopyDirect(array, out->mutable_float32_list()->mutable_value());
      break;
    case federated_language::DT_DOUBLE:
      CopyDirect(array, out->mutable_float64_list()->mutable_value());
      break;
    case federated_language::DT_COMPLEX64:
      CopyDirect(array, out->mutable_complex64_list()->mutable_value());
      break;
    case federated_language::DT_COMPLEX128:
      CopyDirect(array, out->mutable_complex128_list()->mutable_value());
      break;
    case federated_language::DT_BFLOAT16:
      CopyWidened<uint16_t>(array,
                            out->mutable_bfloat16_list()->mutable_value());
      break;
    default:
      break;
  }
}

// Returns a zero-copy NumPy array wrapping `list` without copying elements.
// The RepeatedField is moved to the heap and managed by a py::capsule.
template <typename ProtoFieldT>
std::optional<py::array> ZeroCopyArray(RepeatedField<ProtoFieldT>& list,
                                       const py::dtype& dtype,
                                       const std::vector<py::ssize_t>& shape,
                                       size_t num_elements) {
  const size_t values_per_element =
      static_cast<size_t>(dtype.itemsize()) / sizeof(ProtoFieldT);
  if (static_cast<size_t>(list.size()) != num_elements * values_per_element) {
    return std::nullopt;
  }
  if (num_elements == 0) return py::array(dtype, shape);

  auto owner = std::make_unique<RepeatedField<ProtoFieldT>>(std::move(list));
  const void* data = owner->data();
  py::capsule base(owner.get(), [](void* p) {
    delete static_cast<RepeatedField<ProtoFieldT>*>(p);
  });
  owner.release();
  return py::array(dtype, shape, /*strides=*/{}, data, base);
}

// Returns a NumPy array by narrowing elements from 32-bit integers in `list`.
template <typename NativeT>
std::optional<py::array> NarrowedArray(RepeatedField<int32_t>& list,
                                       const py::dtype& dtype,
                                       const std::vector<py::ssize_t>& shape,
                                       size_t num_elements) {
  if (static_cast<size_t>(list.size()) != num_elements) {
    return std::nullopt;
  }
  py::array result(dtype, shape);
  if (num_elements == 0) return result;

  auto* dst = static_cast<NativeT*>(result.mutable_data());
  const int32_t* src = list.data();
  for (size_t i = 0; i < num_elements; ++i) {
    dst[i] = static_cast<NativeT>(src[i]);
  }
  list.Clear();
  return result;
}

// Converts raw Cord `content` (from TensorFlow executor) to a NumPy array.
// Zero-copy if the Cord is a single aligned chunk; copied otherwise.
std::optional<py::array> ContentToArray(const absl::Cord& content,
                                        const py::dtype& dtype,
                                        const std::vector<py::ssize_t>& shape,
                                        size_t num_elements) {
  const size_t num_bytes = num_elements * static_cast<size_t>(dtype.itemsize());
  if (content.size() != num_bytes) return std::nullopt;

  auto owner = std::make_unique<absl::Cord>(content);
  std::optional<absl::string_view> flat = owner->TryFlat();
  const bool is_aligned = num_bytes > 0 && flat.has_value() &&
                          reinterpret_cast<uintptr_t>(flat->data()) %
                                  static_cast<uintptr_t>(dtype.alignment()) ==
                              0;

  py::array result;
  if (is_aligned) {
    const void* data = flat->data();
    py::capsule base(owner.get(),
                     [](void* p) { delete static_cast<absl::Cord*>(p); });
    owner.release();
    result = py::array(dtype, shape, /*strides=*/{}, data, base);
  } else {
    result = py::array(dtype, shape);
    char* dst = static_cast<char*>(result.mutable_data());
    for (absl::string_view chunk : content.Chunks()) {
      std::memcpy(dst, chunk.data(), chunk.size());
      dst += chunk.size();
    }
  }
  result.attr("setflags")(py::arg("write") = false);
  return result;
}

// Converts `array` to a NumPy array, or nullopt if unsupported/malformed.
std::optional<py::array> EncodeArray(Array& array) {
  if (array.shape().unknown_rank()) return std::nullopt;
  constexpr size_t kMaxElements =
      static_cast<size_t>(std::numeric_limits<py::ssize_t>::max()) /
      sizeof(std::complex<double>);
  std::vector<py::ssize_t> shape;
  size_t num_elements = 1;
  for (int64_t dim : array.shape().dim()) {
    if (dim < 0) return std::nullopt;
    if (dim > 0 && num_elements > kMaxElements / static_cast<size_t>(dim)) {
      return std::nullopt;
    }
    num_elements *= static_cast<size_t>(dim);
    shape.push_back(static_cast<py::ssize_t>(dim));
  }

  if (array.has_content()) {
    std::optional<py::dtype> dtype = DtypeForDataType(array.dtype());
    if (!dtype.has_value()) return std::nullopt;
    return ContentToArray(array.content(), *dtype, shape, num_elements);
  }

  switch (array.kind_case()) {
    case Array::kBoolList:
      return ZeroCopyArray(*array.mutable_bool_list()->mutable_value(),
                           py::dtype::of<bool>(), shape, num_elements);
    case Array::kInt8List:
      return NarrowedArray<int8_t>(*array.mutable_int8_list()->mutable_value(),
                                   py::dtype::of<int8_t>(), shape,
                                   num_elements);
    case Array::kInt16List:
      return NarrowedArray<int16_t>(
          *array.mutable_int16_list()->mutable_value(),
          py::dtype::of<int16_t>(), shape, num_elements);
    case Array::kInt32List:
      return ZeroCopyArray(*array.mutable_int32_list()->mutable_value(),
                           py::dtype::of<int32_t>(), shape, num_elements);
    case Array::kInt64List:
      return ZeroCopyArray(*array.mutable_int64_list()->mutable_value(),
                           py::dtype::of<int64_t>(), shape, num_elements);
    case Array::kUint8List:
      return NarrowedArray<uint8_t>(
          *array.mutable_uint8_list()->mutable_value(),
          py::dtype::of<uint8_t>(), shape, num_elements);
    case Array::kUint16List:
      return NarrowedArray<uint16_t>(
          *array.mutable_uint16_list()->mutable_value(),
          py::dtype::of<uint16_t>(), shape, num_elements);
    case Array::kUint32List:
      return ZeroCopyArray(*array.mutable_uint32_list()->mutable_value(),
                           py::dtype::of<uint32_t>(), shape, num_elements);
    case Array::kUint64List:
      return ZeroCopyArray(*array.mutable_uint64_list()->mutable_value(),
                           py::dtype::of<uint64_t>(), shape, num_elements);
    case Array::kFloat16List:
      return NarrowedArray<uint16_t>(
          *array.mutable_float16_list()->mutable_value(), py::dtype("float16"),
          shape, num_elements);
    case Array::kFloat32List:
      return ZeroCopyArray(*array.mutable_float32_list()->mutable_value(),
                           py::dtype::of<float>(), shape, num_elements);
    case Array::kFloat64List:
      return ZeroCopyArray(*array.mutable_float64_list()->mutable_value(),
                           py::dtype::of<double>(), shape, num_elements);
    case Array::kComplex64List:
      return ZeroCopyArray(*array.mutable_complex64_list()->mutable_value(),
                           py::dtype::of<std::complex<float>>(), shape,
                           num_elements);
    case Array::kComplex128List:
      return ZeroCopyArray(*array.mutable_complex128_list()->mutable_value(),
                           py::dtype::of<std::complex<double>>(), shape,
                           num_elements);
    case Array::kBfloat16List:
      return NarrowedArray<uint16_t>(
          *array.mutable_bfloat16_list()->mutable_value(), BFloat16Dtype(),
          shape, num_elements);
    default:
      return std::nullopt;
  }
}

}  // namespace

void DecodeValue(const federated_language::Type& type, py::handle leaves,
                 Value* out) {
  if (type.has_struct_()) {
    py::list list = leaves.cast<py::list>();
    const auto& element_types = type.struct_().element();
    if (static_cast<int>(list.size()) != element_types.size()) {
      throw py::value_error(absl::StrCat("Expected ", element_types.size(),
                                         " struct elements, found ",
                                         list.size(), "."));
    }
    auto* elements = out->mutable_struct_()->mutable_element();
    for (int i = 0; i < element_types.size(); ++i) {
      Value::Struct::Element* element = elements->Add();
      element->set_name(element_types[i].name());
      DecodeValue(element_types[i].value(), list[i], element->mutable_value());
    }
  } else if (type.has_federated()) {
    py::list list = leaves.cast<py::list>();
    Value::Federated* federated = out->mutable_federated();
    *federated->mutable_type() = type.federated();
    for (py::handle member_leaves : list) {
      DecodeValue(type.federated().member(), member_leaves,
                  federated->add_value());
    }
  } else if (type.has_tensor() && py::isinstance<py::array>(leaves)) {
    DecodeArray(type.tensor(), py::reinterpret_borrow<py::array>(leaves),
                out->mutable_array());
  } else {
    py::bytes serialized = leaves.cast<py::bytes>();
    if (!out->ParseFromString(absl::string_view(serialized))) {
      throw py::value_error(
          "Failed to parse serialized tensorflow_federated.v0.Value.");
    }
  }
}

py::object EncodeValue(const federated_language::Type& type, Value& value) {
  if (type.has_struct_()) {
    const auto& element_types = type.struct_().element();
    if (value.struct_().element_size() != element_types.size()) {
      throw py::value_error(absl::StrCat("Expected ", element_types.size(),
                                         " struct elements, found ",
                                         value.struct_().element_size(), "."));
    }
    py::list elements;
    for (int i = 0; i < element_types.size(); ++i) {
      elements.append(EncodeValue(
          element_types[i].value(),
          *value.mutable_struct_()->mutable_element(i)->mutable_value()));
    }
    return std::move(elements);
  }
  if (type.has_federated()) {
    py::list members;
    for (Value& member : *value.mutable_federated()->mutable_value()) {
      members.append(EncodeValue(type.federated().member(), member));
      if (type.federated().all_equal()) break;
    }
    return std::move(members);
  }
  if (type.has_tensor() && value.has_array()) {
    std::optional<py::array> array = EncodeArray(*value.mutable_array());
    if (array.has_value()) {
      value = Value();
      return *std::move(array);
    }
  }
  // Protobuf cannot serialize a value of 2 GiB or more.
  std::string serialized;
  if (!value.SerializeToString(&serialized)) {
    throw py::value_error("Failed to serialize tensorflow_federated.v0.Value.");
  }
  value = Value();
  return py::bytes(serialized);
}

}  // namespace confidential_federated_compute::program_executor_tee
