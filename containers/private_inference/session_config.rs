use oak_attestation_verification::{EventLogVerifier, SessionBindingPublicKeyPolicy};
use oak_attestation_verification_types::util::Clock;
use oak_crypto::certificate::certificate_verifier::CertificateVerifier;
use oak_crypto_tink::signature_verifier::SignatureVerifier;
use oak_proto_rust::attestation::CERTIFICATE_BASED_ATTESTATION_ID;
use oak_session::config::SessionConfigBuilder;
use oak_session::key_extractor::DefaultBindingKeyExtractor;
use oak_time::{Clock as OakClock, Instant};
use oak_time_std::instant::from_system_time;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

struct SystemClock;

impl Clock for SystemClock {
    fn get_milliseconds_since_epoch(&self) -> i64 {
        match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(n) => n.as_millis().try_into().unwrap(),
            Err(_) => 0,
        }
    }
}

impl OakClock for SystemClock {
    fn get_time(&self) -> Instant {
        from_system_time(SystemTime::now())
    }
}

/// Modifies a peer-unidirectional session config by adding a peer verifier.
///
/// # Safety
///
/// The returned config is an opaque raw pointer to a `SessionConfig` object.
/// The handle is intended to be reclaimed by passing it the FFI factory of an
///  oak_session::ffi::new_client_session.
///
/// tink_serialized_public_keyset_data and tink_serialized_public_keyset_len
/// must describe a valid  buffer. Data must not be modified during this
/// function call. It may be modified or discarded after, as this function
/// will make its own copy.
#[no_mangle]
pub unsafe extern "C" fn update_peer_unidirectional_session_config(
    builder: *mut SessionConfigBuilder,
    tink_serialized_public_keyset_data: *const u8,
    tink_serialized_public_keyset_len: usize,
) -> *mut SessionConfigBuilder {
    let builder = Box::from_raw(builder);

    // Safety: data and len assumed to describe a valid buffer, satisfying the
    // safety condition of std::slice::from_raw_parts.
    let keyset_slice: Vec<u8> = unsafe {
        let extern_slice: &[u8] = std::slice::from_raw_parts(
            tink_serialized_public_keyset_data,
            tink_serialized_public_keyset_len,
        );
        Vec::from(extern_slice) // Makes a copy.
    };

    let signature_verifier = SignatureVerifier::new(&keyset_slice);
    let certificate_verifier = CertificateVerifier::new(signature_verifier);
    let policy = SessionBindingPublicKeyPolicy::new(certificate_verifier);
    let attestation_verifier =
        EventLogVerifier::new(vec![Box::new(policy)], Arc::new(SystemClock {}));

    let builder = Box::new(builder.add_peer_verifier_with_key_extractor(
        CERTIFICATE_BASED_ATTESTATION_ID.to_string(),
        Box::new(attestation_verifier),
        Box::new(DefaultBindingKeyExtractor {}),
    ));

    Box::into_raw(builder)
}
