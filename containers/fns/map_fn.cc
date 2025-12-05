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
#include "containers/fns/map_fn.h"

#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "containers/session.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "google/protobuf/any.pb.h"

namespace confidential_federated_compute::fns {

absl::StatusOr<fcp::confidentialcompute::WriteFinishedResponse> MapFn::Write(
    fcp::confidentialcompute::WriteRequest write_request,
    std::string unencrypted_data, Context& context) {
  KeyValue input;
  input.value.data = unencrypted_data;
  input.value.metadata = write_request.first_request_metadata();
  input.key = write_request.first_request_configuration();
  absl::StatusOr<KeyValue> output = Map(input, context);
  // TODO: Handle output encryption more generally. Currently Map is responsible
  // for encrypting the output properly.
  if (!output.ok()) {
    return ToWriteFinishedResponse(output.status());
  }

  fcp::confidentialcompute::ReadResponse read_response;
  read_response.set_data(std::move(output->value.data));
  *read_response.mutable_first_response_metadata() =
      std::move(output->value.metadata);
  *read_response.mutable_first_response_configuration() =
      std::move(output->key);
  read_response.set_finish_read(true);
  context.Emit(read_response);
  fcp::confidentialcompute::WriteFinishedResponse response;
  response.set_committed_size_bytes(unencrypted_data.size());
  return response;
}

}  // namespace confidential_federated_compute::fns
