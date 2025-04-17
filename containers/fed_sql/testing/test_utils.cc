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
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"

namespace confidential_federated_compute::fed_sql::testing {

using ::tensorflow_federated::aggregation::CheckpointBuilder;
using ::tensorflow_federated::aggregation::CreateTestData;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;

std::string BuildFedSqlGroupByCheckpoint(
    std::initializer_list<uint64_t> key_col_values,
    std::initializer_list<uint64_t> val_col_values,
    const std::string& key_col_name = "key",
    const std::string& val_col_name = "val") {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> ckpt_builder = builder_factory.Create();

  absl::StatusOr<Tensor> key =
      Tensor::Create(DataType::DT_INT64,
                     TensorShape({static_cast<int64_t>(key_col_values.size())}),
                     CreateTestData<uint64_t>(key_col_values));
  absl::StatusOr<Tensor> val =
      Tensor::Create(DataType::DT_INT64,
                     TensorShape({static_cast<int64_t>(val_col_values.size())}),
                     CreateTestData<uint64_t>(val_col_values));
  CHECK_OK(key);
  CHECK_OK(val);
  CHECK_OK(ckpt_builder->Add(key_col_name, *key));
  CHECK_OK(ckpt_builder->Add(val_col_name, *val));
  auto checkpoint = ckpt_builder->Build();
  CHECK_OK(checkpoint.status());

  std::string checkpoint_string;
  absl::CopyCordToString(*checkpoint, &checkpoint_string);
  return checkpoint_string;
}

}  // namespace confidential_federated_compute::fed_sql::testing
