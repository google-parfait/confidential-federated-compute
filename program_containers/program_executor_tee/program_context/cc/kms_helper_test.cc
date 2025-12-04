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
#include "program_executor_tee/program_context/cc/kms_helper.h"

#include <string>

#include "containers/crypto.h"
#include "containers/crypto_test_utils.h"
#include "fcp/protos/confidentialcompute/data_read_write.pb.h"

namespace confidential_federated_compute::program_executor_tee {
namespace {

using crypto_test_utils::MockSigningKeyHandle;
using ::fcp::confidentialcompute::outgoing::WriteRequest;
using ::testing::NiceMock;

constexpr char kKeyId[] = "test";

TEST(KmsHelperTest, CreateWriteRequestForRelease) {
  std::pair<std::string, std::string> public_private_key_pair =
      crypto_test_utils::GenerateKeyPair(kKeyId);
  NiceMock<MockSigningKeyHandle> mock_signing_key_handle;

  WriteRequest write_request;
  ASSERT_TRUE(CreateWriteRequestForRelease(
                  &write_request, mock_signing_key_handle,
                  public_private_key_pair.first, "my_key", "my_data",
                  "my_access_policy_hash", "src_state", "dst_state")
                  .ok());

  google::protobuf::Struct config_properties;
  auto blob_decryptor =
      std::make_unique<confidential_federated_compute::BlobDecryptor>(
          mock_signing_key_handle, config_properties,
          std::vector<absl::string_view>({public_private_key_pair.second}));
  auto plaintext_result = blob_decryptor->DecryptBlob(
      write_request.first_request_metadata(), write_request.data(), kKeyId);
  ASSERT_TRUE(plaintext_result.ok());
  ASSERT_EQ(*plaintext_result, "my_data");

  ASSERT_EQ(write_request.key(), "my_key");
  ASSERT_TRUE(write_request.commit());
}

}  // namespace
}  // namespace confidential_federated_compute::program_executor_tee