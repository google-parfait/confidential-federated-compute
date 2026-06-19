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
#include "py_mauve_delegate.h"

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "fcp/protos/confidentialcompute/mauve_score_config.pb.h"

namespace confidential_federated_compute::mauve_score {

namespace {

// Flattens a vector of float vectors into a contiguous bytes buffer
// suitable for passing to Python via pybind11::bytes.
pybind11::bytes FlattenToBytes(
    const std::vector<std::vector<float>>& embeddings) {
  const size_t n_rows = embeddings.size();
  const size_t dim = embeddings[0].size();
  std::vector<float> flat(n_rows * dim);
  for (size_t i = 0; i < n_rows; i++) {
    std::copy(embeddings[i].begin(), embeddings[i].end(),
              flat.begin() + i * dim);
  }
  return pybind11::bytes(reinterpret_cast<const char*>(flat.data()),
                         flat.size() * sizeof(float));
}

}  // namespace

PyRuntimeManager::PyRuntimeManager() {
  pybind11::initialize_interpreter();
  // Release GIL so gRPC worker threads can acquire it.
  tstate_ = PyEval_SaveThread();
  LOG(INFO) << "Python interpreter initialized.";
}

PyRuntimeManager::~PyRuntimeManager() {
  PyEval_RestoreThread(tstate_);
  pybind11::finalize_interpreter();
  LOG(INFO) << "Python interpreter finalized.";
}

absl::Status PyRuntimeManager::ImportLib() {
  pybind11::gil_scoped_acquire acquire;
  try {
    pybind11::module_::import("compute_mauve_score_lib");
    LOG(INFO) << "compute_mauve_score_lib imported successfully.";
    return absl::OkStatus();
  } catch (pybind11::error_already_set& e) {
    std::string error =
        absl::StrCat("Import compute_mauve_score_lib failed: ", e.what());
    LOG(WARNING) << error;
    return absl::InternalError(error);
  }
}

absl::StatusOr<fcp::confidentialcompute::MauveScoreResult>
ComputeMauveViaPython(const std::vector<std::vector<float>>& real_embeddings,
                      const std::vector<std::vector<float>>& synth_embeddings) {
  pybind11::gil_scoped_acquire acquire;
  LOG(INFO) << "GIL acquired. Computing MAUVE score via Python.";

  try {
    pybind11::module_ lib =
        pybind11::module_::import("compute_mauve_score_lib");

    // Pass raw float bytes + dimensions to Python.
    // Python handles numpy array construction via np.frombuffer().reshape().
    pybind11::bytes real_bytes = FlattenToBytes(real_embeddings);
    pybind11::bytes synth_bytes = FlattenToBytes(synth_embeddings);
    int32_t embedding_dim = real_embeddings[0].size();
    if (embedding_dim != synth_embeddings[0].size()) {
      return absl::InvalidArgumentError(
          "Real and synthetic embedding dimensions must match.");
    }

    // Call Python function that builds numpy arrays, computes MAUVE,
    // and returns serialized proto bytes.
    pybind11::bytes serialized_result =
        lib.attr("compute_mauve_and_serialize_result")(
            real_bytes, pybind11::int_(embedding_dim), synth_bytes,
            pybind11::int_(embedding_dim));

    // Parse the serialized proto in C++.
    fcp::confidentialcompute::MauveScoreResult result;
    std::string serialized_str = serialized_result.cast<std::string>();
    if (!result.ParseFromString(serialized_str)) {
      return absl::InternalError("Failed to parse MauveScoreResult proto.");
    }

    LOG(INFO) << "MAUVE result: AUC=" << result.mauve_auc()
              << ", recall=" << result.recall()
              << ", precision=" << result.precision()
              << ", n_clusters=" << result.num_clusters();

    return result;
  } catch (pybind11::error_already_set& e) {
    std::string error_msg = e.what();
    LOG(ERROR) << "Python MAUVE computation failed: " << error_msg;
    return absl::InternalError(error_msg);
  }
}

}  // namespace confidential_federated_compute::mauve_score
