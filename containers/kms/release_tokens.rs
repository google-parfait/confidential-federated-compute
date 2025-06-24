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

//! Provides functions for verifying and decrypting release tokens, including
//! generating the transform signing key endorsement used to establish the
//! provenance of a release token.

use anyhow::{anyhow, bail, ensure, Context};
use bssl_crypto::{ec, ecdsa, hpke};
use coset::{
    cbor::value::Value,
    cwt::{ClaimName, ClaimsSet},
    iana, Algorithm, CborSerializable, CoseEncrypt0, CoseKey, CoseKeyBuilder, CoseSign1,
    CoseSign1Builder, HeaderBuilder, KeyOperation, KeyType, Label,
};
use hashbrown::HashMap;
use key_derivation::{derive_private_keys, HPKE_BASE_X25519_SHA256_AES128GCM, PUBLIC_KEY_CLAIM};
use storage_proto::confidential_federated_compute::kms::PipelineInvocationStateValue;

// Private COSE Header parameters; see
// https://github.com/google/federated-compute/blob/main/fcp/protos/confidentialcompute/cbor_ids.md.
pub const ENCAPSULATED_KEY_PARAM: i64 = -65537;
pub const RELEASE_TOKEN_SRC_STATE_PARAM: i64 = -65538;
pub const RELEASE_TOKEN_DST_STATE_PARAM: i64 = -65539;

/// Generates the signing key endorsement for a transform.
///
/// The endorsement is a CBOR Web Token (CWT; RFC 8392) signed by the cluster
/// key. It contains the transform signing key in a claim; the caller may
/// provide additional claims as desired (e.g. information about how the
/// transform matched the access policy).
pub fn endorse_transform_signing_key(
    transform_signing_key: &[u8],
    cluster_key: &ecdsa::PrivateKey<ec::P256>,
    mut claims: ClaimsSet,
) -> anyhow::Result<Vec<u8>> {
    // Add a claim containing the transform signing key. An uncompressed X9.62
    // public key is "0x04<x><y>". For P-256, x and y are 32 bytes each.
    ensure!(
        transform_signing_key.starts_with(b"\x04") && transform_signing_key.len() == 65,
        "transform_signing_key is not a X9.62-encoded ECDSA P-256 public key"
    );
    let cose_key = CoseKeyBuilder::new_ec2_pub_key(
        iana::EllipticCurve::P_256,
        transform_signing_key[1..33].into(),
        transform_signing_key[33..].into(),
    )
    .algorithm(iana::Algorithm::ES256)
    .add_key_op(iana::KeyOperation::Verify)
    .build()
    .to_vec()
    .map_err(anyhow::Error::msg)
    .context("failed to encode CoseKey")?;
    claims.rest.push((ClaimName::PrivateUse(PUBLIC_KEY_CLAIM), Value::from(cose_key)));

    CoseSign1Builder::new()
        .protected(HeaderBuilder::new().algorithm(iana::Algorithm::ES256).build())
        .payload(claims.to_vec().map_err(anyhow::Error::msg).context("failed to encode ClaimsSet")?)
        .create_signature(b"", |msg| cluster_key.sign_p1363(msg))
        .build()
        .to_vec()
        .map_err(anyhow::Error::msg)
        .context("failed to encode CoseSign1")
}

/// Verifies that a release token is valid and was signed by the signing key in
/// the endorsement.
///
/// Returns the release token's payload, the endorsement's claims, and a
/// function that verifies the endorsement's signature.
pub fn verify_release_token(
    token: &[u8],
    endorsement: &[u8],
) -> anyhow::Result<(
    CoseEncrypt0,
    ClaimsSet,
    impl Fn(&ecdsa::PublicKey<ec::P256>) -> anyhow::Result<()>,
)> {
    // Extract the transform's public key from the endorsement.
    let endorsement = CoseSign1::from_slice(endorsement)
        .map_err(anyhow::Error::msg)
        .context("failed to parse endorsement")?;
    let claims = ClaimsSet::from_slice(endorsement.payload.as_deref().unwrap_or_default())
        .map_err(anyhow::Error::msg)
        .context("failed to parse endorsement claims")?;
    let cose_key = claims
        .rest
        .iter()
        .find(|(name, _)| name == &ClaimName::PrivateUse(PUBLIC_KEY_CLAIM))
        .and_then(|(_, value)| value.as_bytes())
        .context("endorsement is missing public key")?;
    let cose_key = CoseKey::from_slice(cose_key)
        .map_err(anyhow::Error::msg)
        .context("failed to parse public key")?;
    ensure!(cose_key.kty == KeyType::Assigned(iana::KeyType::EC2), "unsupported public key type");
    ensure!(
        cose_key.alg == Some(Algorithm::Assigned(iana::Algorithm::ES256)),
        "unsupported public key algorithm"
    );
    ensure!(
        cose_key.key_ops.is_empty()
            || cose_key.key_ops.contains(&KeyOperation::Assigned(iana::KeyOperation::Verify)),
        "public key disallows verify operation"
    );
    let (mut crv, mut x, mut y) = (None, None, None);
    for (label, value) in cose_key.params {
        use iana::Ec2KeyParameter;
        match (label, value) {
            (Label::Int(l), v) if l == Ec2KeyParameter::Crv as i64 => crv = Some(v),
            (Label::Int(l), Value::Bytes(v)) if l == Ec2KeyParameter::X as i64 => x = Some(v),
            (Label::Int(l), Value::Bytes(v)) if l == Ec2KeyParameter::Y as i64 => y = Some(v),
            _ => {}
        }
    }
    ensure!(crv == Some(Value::from(iana::EllipticCurve::P_256 as i64)), "unsupported curve");
    ensure!(x.as_ref().is_some_and(|x| x.len() == 32), "invalid x coordinate");
    ensure!(y.as_ref().is_some_and(|y| y.len() == 32), "invalid y coordinate");
    let x962_public_key = [b"\x04", x.unwrap().as_slice(), y.unwrap().as_slice()].concat();
    let public_key = ecdsa::PublicKey::<ec::P256>::from_x962_uncompressed(&x962_public_key)
        .context("failed to parse public key")?;

    // Verify the release token's signature.
    let token = CoseSign1::from_slice(token)
        .map_err(anyhow::Error::msg)
        .context("failed to parse release token")?;
    ensure!(token.protected.header.alg == cose_key.alg, "release token algorithm mismatch");
    token
        .verify_signature(b"", |signature, data| {
            public_key
                .verify_p1363(data, signature)
                .map_err(|_| anyhow!("signature verification failed"))
        })
        .context("invalid release token signature")?;

    // Define a function to verify the endorsement's signature.
    ensure!(
        endorsement.protected.header.alg == Some(Algorithm::Assigned(iana::Algorithm::ES256)),
        "unsupported endorsement signature algorithm"
    );
    let verify_signature_fn = move |cluster_key: &ecdsa::PublicKey<ec::P256>| {
        endorsement
            .verify_signature(b"", |signature, data| {
                cluster_key
                    .verify_p1363(data, signature)
                    .map_err(|_| anyhow!("signature verification failed"))
            })
            .context("invalid endorsement signature")
    };

    let token_payload = CoseEncrypt0::from_slice(token.payload.as_deref().unwrap_or_default())
        .map_err(anyhow::Error::msg)
        .context("invalid release token payload")?;
    Ok((token_payload, claims, verify_signature_fn))
}

/// Decrypts and returns the protected contents of a release token.
///
/// Decryption will not be performed if the payload was encrypted with a key
/// derived from a node id that is not in `dst_node_ids`.
pub fn decrypt_release_token(
    token_payload: &CoseEncrypt0,
    dst_node_ids: &[Value],
    invocation_state: &PipelineInvocationStateValue,
    intermediate_key_id_prefix: &[u8],
) -> anyhow::Result<Vec<u8>> {
    // Determine the node id used to derive the encryption key. Key derivation
    // sets the key_id to the prefix followed by the node id as a big-endian
    // 32-bit integer.
    let node_id = token_payload
        .unprotected
        .key_id
        .as_slice()
        .strip_prefix(intermediate_key_id_prefix)
        .and_then(|id| Some(u32::from_be_bytes(id.try_into().ok()?)))
        .context("invalid key id")?;
    ensure!(
        dst_node_ids.contains(&Value::from(node_id)),
        "endorsement doesn't include dst_node_id {}",
        node_id
    );

    // Derive the decryption key.
    let intermediates_key = invocation_state
        .intermediates_key
        .as_ref()
        .context("PipelineInvocationState missing intermediates_key")?;
    let private_keys = derive_private_keys(
        intermediates_key.algorithm,
        intermediate_key_id_prefix,
        &intermediates_key.ikm,
        [node_id.to_be_bytes()],
    )?;
    let cose_key = CoseKey::from_slice(private_keys.first().map(Vec::as_slice).unwrap_or_default())
        .map_err(anyhow::Error::msg)
        .context("derive_private_keys produced invalid key")?;
    ensure!(
        token_payload.protected.header.alg == cose_key.alg,
        "release token has wrong algorithm"
    );

    // Decrypt the release token.
    let params = match intermediates_key.algorithm {
        HPKE_BASE_X25519_SHA256_AES128GCM => hpke::Params::new(
            hpke::Kem::X25519HkdfSha256,
            hpke::Kdf::HkdfSha256,
            hpke::Aead::Aes128Gcm,
        ),
        _ => bail!("unsupported release token algorithm"),
    };
    let private_key = cose_key
        .params
        .iter()
        .find(|(label, _)| label == &Label::Int(iana::OkpKeyParameter::D as i64))
        .and_then(|(_, value)| value.as_bytes())
        .context("derived key missing private key parameter")?;
    let encapsulated_key = token_payload
        .unprotected
        .rest
        .iter()
        .find(|(name, _)| name == &Label::Int(ENCAPSULATED_KEY_PARAM))
        .and_then(|(_, value)| value.as_bytes())
        .context("release token missing encapsulated key")?;
    ensure!(token_payload.ciphertext.is_some(), "release token missing ciphertext");
    token_payload.decrypt(b"", |ciphertext, aad| {
        hpke::RecipientContext::new(&params, private_key, encapsulated_key, b"")
            .and_then(|mut context| context.open(ciphertext, aad))
            .context("failed to decrypt release token")
    })
}

/// An atomic update to the state of a logical pipeline.
#[derive(Clone, Debug)]
pub struct LogicalPipelineUpdate<'a> {
    pub logical_pipeline_name: &'a str,
    pub src_state: Option<&'a [u8]>,
    pub dst_state: &'a [u8],
}

/// Computes the set of storage updates implied by the release tokens, as well
/// as the resulting state of each logical pipeline after the updates are
/// applied.
///
/// `release_tokens` is a stream of (logical pipeline name, token payload)
/// tuples.
///
/// If multiple release tokens affect the same logical pipeline, this function
/// will combine their mutations to produce a single update. For example,
/// updates A -> B, B -> C, and C -> D will be combined to A -> D. The release
/// tokens may be provided in any order.
pub fn compute_logical_pipeline_updates<'a>(
    release_tokens: impl IntoIterator<Item = (&'a str, &'a CoseEncrypt0)>,
) -> anyhow::Result<Vec<LogicalPipelineUpdate<'a>>> {
    // Collect the set of state changes for each logical pipeline. Note that
    // while the src state can be None, the dst cannot.
    type StateChange<'a> = (Option<&'a [u8]>, &'a [u8]);
    let mut state_changes: HashMap<&str, Vec<StateChange>> = HashMap::new();
    for (logical_pipeline_name, token_payload) in release_tokens {
        let (mut src_state, mut dst_state) = (None, None);
        for (label, value) in &token_payload.protected.header.rest {
            match (label, value) {
                (Label::Int(l), Value::Null) if l == &RELEASE_TOKEN_SRC_STATE_PARAM => {
                    src_state = Some(None);
                }
                (Label::Int(l), Value::Bytes(v)) if l == &RELEASE_TOKEN_SRC_STATE_PARAM => {
                    src_state = Some(Some(v.as_slice()));
                }
                (Label::Int(l), Value::Bytes(v)) if l == &RELEASE_TOKEN_DST_STATE_PARAM => {
                    dst_state = Some(v.as_slice());
                }
                _ => {}
            }
        }
        state_changes.entry(logical_pipeline_name).or_default().push((
            src_state.context("release token missing src state")?,
            dst_state.context("release token missing dst state")?,
        ));
    }

    // For each logical pipeline, find the Eulerian trail (if any): the sequence
    // of state changes that uses every edge (state change) exactly once. To
    // avoid ambiguity, we additionally require that this path is not a circuit:
    // its start and end states must be different.
    state_changes
        .into_iter()
        .map(|(logical_pipeline_name, transitions)| {
            let (src_state, dst_state) = find_eulerian_trail(&transitions)
                .with_context(|| format!("invalid state changes for {}", logical_pipeline_name))?;
            Ok(LogicalPipelineUpdate { logical_pipeline_name, src_state, dst_state })
        })
        .collect()
}

/// Returns the start and end of a Eulerian trail over the given directed
/// multigraph. This trail will not be a circuit.
fn find_eulerian_trail<'a>(
    edges: &[(Option<&'a [u8]>, &'a [u8])],
) -> anyhow::Result<(Option<&'a [u8]>, &'a [u8])> {
    // As described by https://en.wikipedia.org/wiki/Eulerian_path, the graph
    // will contain a Eulerian trail (that's not a circuit) if (a) it's
    // connected, (b) exactly one vertex has `outdegree - indegree = 1` and
    // exactly one vertex has `outdegree - indegree = -1`, and (c) all other
    // vertices have equal indegree and outdegree.

    // Process the list of edges into per-node information.
    #[derive(Default)]
    struct Entry<'a> {
        /// The nodes adjacent to this node, ignoring edge directionality.
        adjacent_nodes: Vec<Option<&'a [u8]>>,
        /// The node's `outdegree - indegree`.
        degree_difference: i32,
    }
    let mut nodes: HashMap<Option<&[u8]>, Entry> = HashMap::with_capacity(edges.len());
    for (src, dst) in edges {
        let entry = nodes.entry(*src).or_default();
        entry.adjacent_nodes.push(Some(dst));
        entry.degree_difference += 1;
        let entry = nodes.entry(Some(dst)).or_default();
        entry.adjacent_nodes.push(*src);
        entry.degree_difference -= 1;
    }

    // Run a DFS over the graph.
    let mut stack = vec![edges.first().context("no state changes found")?.0];
    while let Some(node) = stack.pop() {
        // `Vec::append` moves all elements, leaving `adjacent_nodes` empty.
        // Conveniently, this ensures that we don't traverse the same edge
        // twice. It also means that after the DFS completes, a node was reached
        // iff it has an empty adjacency list -- and the graph is connected iff
        // all adjacency lists are empty.
        stack.append(&mut nodes.get_mut(&node).unwrap().adjacent_nodes);
    }

    // Check the results to determine whether the graph contains a Eulerian
    // trail, and if so, what its start and end states are.
    let mut src = None;
    let mut dst = None;
    for (node, Entry { adjacent_nodes, degree_difference }) in nodes {
        ensure!(adjacent_nodes.is_empty(), "state changes do not form a connected graph");
        match degree_difference {
            -1 => {
                ensure!(dst.is_none(), "multiple dst states");
                dst = Some(node);
            }
            0 => {}
            1 => {
                ensure!(src.is_none(), "multiple src states");
                src = Some(node);
            }
            _ => bail!("state used multiple times"),
        }
    }
    if let (Some(src), Some(Some(dst))) = (src, dst) {
        Ok((src, dst))
    } else {
        Err(anyhow!("cycle in states"))
    }
}
