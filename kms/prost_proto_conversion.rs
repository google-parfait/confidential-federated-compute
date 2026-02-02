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

use prost::{DecodeError, Message};

/// An extension to the `Message` trait that provides conversion between
/// messages built using `cargo_build_script` and `prost_proto_library`.
pub trait ProstProtoConversionExt<T: Message + Default>: Message + Sized {
    fn convert(&self) -> Result<T, DecodeError> {
        T::decode(self.encode_to_vec().as_slice())
    }
}

/// The `cbs` module contains messages built using `cargo_build_script`.
mod cbs {
    pub use oak_proto_rust::oak::{
        attestation::v1::{Endorsements, Evidence, ReferenceValues},
        crypto::v1::EncryptedRequest,
        session::v1::{SessionRequest, SessionResponse},
    };
}

/// The `ppl` module contains messages built using `prost_proto_library`.
mod ppl {
    pub use access_policy_proto::reference_value_proto::oak::attestation::v1::ReferenceValues;
    pub use kms_proto::{
        crypto_proto::oak::crypto::v1::EncryptedRequest,
        endorsement_proto::oak::attestation::v1::Endorsements,
        evidence_proto::oak::attestation::v1::Evidence,
    };
    pub use session_v1_service_proto::session_proto::oak::session::v1::{
        SessionRequest, SessionResponse,
    };
}

impl ProstProtoConversionExt<cbs::EncryptedRequest> for ppl::EncryptedRequest {}
impl ProstProtoConversionExt<ppl::EncryptedRequest> for cbs::EncryptedRequest {}

impl ProstProtoConversionExt<cbs::Endorsements> for ppl::Endorsements {}
impl ProstProtoConversionExt<ppl::Endorsements> for cbs::Endorsements {}

impl ProstProtoConversionExt<cbs::Evidence> for ppl::Evidence {}
impl ProstProtoConversionExt<ppl::Evidence> for cbs::Evidence {}

impl ProstProtoConversionExt<cbs::ReferenceValues> for ppl::ReferenceValues {}
impl ProstProtoConversionExt<ppl::ReferenceValues> for cbs::ReferenceValues {}

impl ProstProtoConversionExt<cbs::SessionRequest> for ppl::SessionRequest {}
impl ProstProtoConversionExt<ppl::SessionRequest> for cbs::SessionRequest {}

impl ProstProtoConversionExt<cbs::SessionResponse> for ppl::SessionResponse {}
impl ProstProtoConversionExt<ppl::SessionResponse> for cbs::SessionResponse {}
