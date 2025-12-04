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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_METADATA_TESTING_TEST_UTILS_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_METADATA_TESTING_TEST_UTILS_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/tee_payload_metadata.pb.h"
#include "gmock/gmock.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace confidential_federated_compute::metadata::testing {

using ::fcp::confidentialcompute::BlobMetadata;
using ::tensorflow_federated::aggregation::Tensor;

MATCHER_P(LowerNBitsAreZero, n, "") {
  if (n < 0 || n > 64) {
    *result_listener << "n must be between 0 and 64, but is " << n;
    return false;
  }
  if (n == 0) return true;
  if (n == 64) return arg == 0;

  uint64_t mask = (1ULL << n) - 1;
  if ((arg & mask) != 0) {
    *result_listener << " has non-zero lower " << n << " bits";
    return false;
  }
  return true;
}

MATCHER_P6(EqualsEventTimeRange, start_year, start_month, start_day, end_year,
           end_month, end_day, "") {
  const auto& start_time = arg.start_event_time();
  const auto& end_time = arg.end_event_time();
  bool start_matches = start_time.year() == start_year &&
                       start_time.month() == start_month &&
                       start_time.day() == start_day;
  bool end_matches = end_time.year() == end_year &&
                     end_time.month() == end_month && end_time.day() == end_day;

  if (!start_matches) {
    *result_listener << " has start time " << start_time.year() << "-"
                     << start_time.month() << "-" << start_time.day();
  }
  if (!end_matches) {
    if (!start_matches) {
      *result_listener << " and";
    }
    *result_listener << " has end time " << end_time.year() << "-"
                     << end_time.month() << "-" << end_time.day();
  }
  return start_matches && end_matches;
}

// TODO: remove these when an upcoming change lands that makes these
// unnecessary (Tensor will have constructors to create 1D tensors directly)
Tensor BuildStringTensor(std::string name, std::vector<std::string> values);

Tensor BuildIntTensor(std::string name, std::initializer_list<int64_t> values);

std::string BuildCheckpointFromTensors(std::vector<Tensor> tensors);

std::pair<BlobMetadata, std::string> EncryptWithKmsKeys(
    std::string message, std::string associated_data, std::string public_key);

// Creates a checkpoint with a privacy ID and event times.
std::string BuildCheckpoint(std::string privacy_id_val,
                            std::vector<std::string> event_times);

// Creates an encrypted checkpoint with a privacy ID and event times.
std::pair<BlobMetadata, std::string> BuildEncryptedCheckpoint(
    std::string privacy_id_val, std::vector<std::string> event_times,
    std::string public_key, std::string associated_data);

}  // namespace confidential_federated_compute::metadata::testing

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_METADATA_TESTING_TEST_UTILS_H_
