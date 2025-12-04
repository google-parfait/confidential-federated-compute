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

#include "absl/strings/string_view.h"
#include "cc/crypto/signing_key.h"
#include "fcp/base/monitoring.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "fcp/protos/confidentialcompute/data_read_write.pb.h"
#include "openssl/rand.h"

namespace confidential_federated_compute::program_executor_tee {

using ::fcp::confidential_compute::EncryptMessageResult;
using ::fcp::confidential_compute::MessageEncryptor;
using ::fcp::confidential_compute::OkpKey;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::outgoing::WriteRequest;

absl::Status CreateWriteRequestForRelease(
    WriteRequest* write_request, oak::crypto::SigningKeyHandle& signing_key,
    absl::string_view encryption_key, std::string key, std::string data,
    std::string access_policy_hash, std::optional<std::string> src_state,
    std::string dst_state) {
  FCP_ASSIGN_OR_RETURN(OkpKey okp_key, OkpKey::Decode(encryption_key));
  BlobHeader header;
  std::string blob_id(kBlobIdSize, '\0');
  (void)RAND_bytes(reinterpret_cast<unsigned char*>(blob_id.data()),
                   blob_id.size());
  header.set_blob_id(blob_id);
  header.set_key_id(okp_key.key_id);
  header.set_access_policy_sha256(access_policy_hash);
  std::string serialized_blob_header = header.SerializeAsString();

  MessageEncryptor message_encryptor;
  FCP_ASSIGN_OR_RETURN(
      EncryptMessageResult encrypted_message,
      message_encryptor.EncryptForRelease(
          data, encryption_key, serialized_blob_header, src_state, dst_state,
          [&signing_key](
              absl::string_view message) -> absl::StatusOr<std::string> {
            FCP_ASSIGN_OR_RETURN(auto signature, signing_key.Sign(message));
            return std::move(*signature.mutable_signature());
          }));

  BlobMetadata metadata;
  metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_NONE);
  metadata.set_total_size_bytes(encrypted_message.ciphertext.size());
  BlobMetadata::HpkePlusAeadMetadata* hpke_plus_aead_metadata =
      metadata.mutable_hpke_plus_aead_data();
  hpke_plus_aead_metadata->set_ciphertext_associated_data(
      std::string(serialized_blob_header));
  hpke_plus_aead_metadata->set_encrypted_symmetric_key(
      encrypted_message.encrypted_symmetric_key);
  hpke_plus_aead_metadata->set_encapsulated_public_key(
      encrypted_message.encapped_key);
  hpke_plus_aead_metadata->mutable_kms_symmetric_key_associated_data()
      ->set_record_header(std::string(serialized_blob_header));

  *write_request->mutable_first_request_metadata() = std::move(metadata);
  write_request->set_commit(true);
  write_request->set_data(encrypted_message.ciphertext);
  write_request->set_release_token(encrypted_message.release_token);
  write_request->set_key(key);

  return absl::OkStatus();
}

}  // namespace confidential_federated_compute::program_executor_tee