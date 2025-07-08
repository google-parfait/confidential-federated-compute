// Copyright 2025 The Trusted Computations Platform Authors.
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

use std::sync::Arc;

use oak_attestation_types::{attester::Attester, endorser::Endorser};
use oak_attestation_verification_types::util::Clock;
use oak_crypto::{encryptor::Encryptor, noise_handshake::OrderedCrypter, signer::Signer};
use oak_dice_attestation_verifier::DiceAttestationVerifier;
use oak_proto_rust::oak::attestation::v1::ReferenceValues;
use oak_session::{
    attestation::AttestationType,
    config::{EncryptorProvider, SessionConfig},
    encryptors::UnorderedChannelEncryptor,
    handshake::HandshakeType,
    session_binding::SignatureBinderBuilder,
};

const SESSION_ID: &str = "cfc_kms";

struct UnorderedEncryptorProvider;
impl EncryptorProvider for UnorderedEncryptorProvider {
    fn provide_encryptor(
        &self,
        crypter: OrderedCrypter,
    ) -> Result<Box<dyn Encryptor>, anyhow::Error> {
        TryInto::<UnorderedChannelEncryptor>::try_into((crypter, /* window_size= */ 64))
            .map(|v| Box::new(v) as Box<dyn Encryptor>)
    }
}

/// Creates a new SessionConfig for connections between KMS servers.
pub fn create_session_config(
    attester: &Arc<dyn Attester>,
    endorser: &Arc<dyn Endorser>,
    signer: Box<dyn Signer>,
    reference_values: ReferenceValues,
    clock: Arc<dyn Clock>,
) -> anyhow::Result<SessionConfig> {
    Ok(SessionConfig::builder(AttestationType::Bidirectional, HandshakeType::NoiseNN)
        .add_self_attester_ref(SESSION_ID.into(), attester)
        .add_self_endorser_ref(SESSION_ID.into(), endorser)
        .add_peer_verifier(
            SESSION_ID.into(),
            Box::new(DiceAttestationVerifier::create(reference_values, clock)),
        )
        // The communication channel is not guaranteed to be ordered.
        .set_encryption_provider(Box::new(UnorderedEncryptorProvider))
        .add_session_binder(
            SESSION_ID.into(),
            Box::new(
                SignatureBinderBuilder::default()
                    .signer(signer)
                    .build()
                    .map_err(anyhow::Error::msg)?,
            ),
        )
        .build())
}
