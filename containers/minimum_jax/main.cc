// Copyright 2025 Google LLC.
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

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <iostream>
#include <string>
#include <vector>

namespace py = pybind11;

int main(int argc, char** argv) {
  std::vector<std::string> inputs{"Winter is coming", "The north remembers",
                                  "Winter came early", "My watch starts"};

  py::scoped_interpreter guard{};

  try {
    py::module_ tokens_lib = py::module_::import("tokens");

    py::object result_obj = tokens_lib.attr("find_most_frequent_token")(inputs);

    std::string most_frequent_token = result_obj.cast<std::string>();
    std::cout << "The most frequent token is " << most_frequent_token
              << std::endl;

  } catch (py::error_already_set& e) {
    std::cerr << "Python error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}