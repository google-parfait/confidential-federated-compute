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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_EVOLUTION_SENTENCE_TRANSFORMERS_PY_MODEL_DELEGATE_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_EVOLUTION_SENTENCE_TRANSFORMERS_PY_MODEL_DELEGATE_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "containers/python/embedded_python.h"
#include "model_delegate.h"

namespace confidential_federated_compute::sentence_transformers {

// ModelDelegate that delegates all method calls to python code.
class PyModelDelegate : public ModelDelegate {
 public:
  bool InitializeModel(absl::string_view artifact_path) override;
  absl::StatusOr<std::vector<std::vector<float>>> GenerateEmbeddings(
      const std::vector<std::string>& inputs,
      std::optional<std::string> prompt) override;
};

// Factory class for creating ModelDelegate.
// This class also initialize/finalize the python interpreter, and loading
// python libraries. Create it on the main thread before starting the gRPC
// server. The binary must link the embedded_python_env target :py_env.
class PyRuntimeManager : public ModelDelegateFactory {
 public:
  PyRuntimeManager();
  ~PyRuntimeManager();
  // Import python libraries.
  // This method should be called before any method of PyModelDelegate is
  // called. Once the libraries are imported, the result will be cached so
  // subsequent import calls are lightweight. This method should run on the main
  // thread at startup: some Python libraries can only be imported on the main
  // thread, for example ones that install signal handlers.
  absl::Status ImportLib();
  std::unique_ptr<ModelDelegate> Create() override {
    return std::make_unique<PyModelDelegate>();
  }

 private:
  std::unique_ptr<python::EmbeddedInterpreter> interpreter_;
};

}  // namespace confidential_federated_compute::sentence_transformers

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_PRIVATE_EVOLUTION_SENTENCE_TRANSFORMERS_PY_MODEL_DELEGATE_H_
