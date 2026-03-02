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

use attestation_transparency_service_proto::payload_transparency_proto::fcp::confidentialcompute::signed_payload::Signature;

/// A trait for creating fcp.confidentialcompute.SignedPayload signatures.
#[mockall::automock]
pub trait PayloadSigner {
    /// Signs the payload with the provided headers.
    ///
    /// `header` should be a serialized `Headers` message. The signing key
    /// algorithm header is automatically added.
    fn sign(&self, headers: &[u8], payload: &[u8]) -> anyhow::Result<Signature>;
}
