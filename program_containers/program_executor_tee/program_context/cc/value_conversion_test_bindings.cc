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

#include <pybind11/pybind11.h>

#include <string>

#include "federated_language/proto/computation.pb.h"
#include "program_executor_tee/program_context/cc/value_conversion.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::program_executor_tee {

namespace py = pybind11;

using ::tensorflow_federated::v0::Value;

namespace {

federated_language::Type ParseType(const std::string& type) {
  federated_language::Type type_pb;
  if (!type_pb.ParseFromString(type)) {
    throw py::value_error("Expected a serialized federated_language.Type.");
  }
  return type_pb;
}

}  // namespace

PYBIND11_MODULE(value_conversion_test_bindings, m) {
  m.doc() = "Test-only access to the C++ half of value_conversion.";

  // Builds a value of type `type` (a serialized `federated_language.Type`) from
  // `leaves`, and converts it straight back to leaves, with no executor in
  // between.
  m.def(
      "round_trip",
      [](const std::string& type, py::handle leaves) {
        federated_language::Type type_pb = ParseType(type);
        Value value;
        DecodeValue(type_pb, leaves, &value);
        return EncodeValue(type_pb, value);
      },
      py::arg("type"), py::arg("leaves"));

  // Builds a value of type `type` from `leaves`, and returns it serialized.
  m.def(
      "serialize",
      [](const std::string& type, py::handle leaves) {
        Value value;
        DecodeValue(ParseType(type), leaves, &value);
        return py::bytes(value.SerializeAsString());
      },
      py::arg("type"), py::arg("leaves"));
}

}  // namespace confidential_federated_compute::program_executor_tee
