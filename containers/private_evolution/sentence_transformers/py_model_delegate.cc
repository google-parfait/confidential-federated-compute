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
#include "py_model_delegate.h"

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <optional>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace confidential_federated_compute::sentence_transformers {

PyModelDelegateFactory::PyModelDelegateFactory() {
  pybind11::initialize_interpreter();
}

PyModelDelegateFactory::~PyModelDelegateFactory() {
  pybind11::finalize_interpreter();
}

bool PyModelDelegate::InitializeModel(absl::string_view model_artifact_path) {
  pybind11::gil_scoped_acquire acquire;
  try {
    pybind11::module_ generate_embedding_lib =
        pybind11::module_::import("generate_embedding");

    auto init_result_py = generate_embedding_lib.attr("initialize_model")(
        std::string(model_artifact_path));
    bool result = init_result_py.cast<bool>();
    LOG(WARNING) << "Model initialization result: " << result;
    return result;
  } catch (pybind11::error_already_set& e) {
    LOG(WARNING) << "Model initialization failed." << e.what();
    return false;
  }
}

absl::StatusOr<std::vector<std::vector<float>>>
PyModelDelegate::GenerateEmbeddings(const std::vector<std::string>& inputs,
                                    std::optional<std::string> prompt) {
  pybind11::gil_scoped_acquire acquire;
  try {
    pybind11::module_ generate_embedding_lib =
        pybind11::module_::import("generate_embedding");
    pybind11::object embedding_py;
    if (prompt.has_value()) {
      embedding_py =
          generate_embedding_lib.attr("encode")(inputs, prompt.value());
    } else {
      embedding_py = generate_embedding_lib.attr("encode")(inputs);
    }
    auto result = embedding_py.cast<std::vector<std::vector<float>>>();
    LOG(WARNING) << "Embedding generated.";
    return result;
  } catch (pybind11::error_already_set& e) {
    std::string error_msg = e.what();
    LOG(ERROR) << "Generate embedding failed." << error_msg;
    return absl::InvalidArgumentError(error_msg);
  }
}

}  // namespace confidential_federated_compute::sentence_transformers
