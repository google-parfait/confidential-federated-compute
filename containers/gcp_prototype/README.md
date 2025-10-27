# GCP Confidential Space Oak Session Prototype

This directory contains a prototype C++/Rust application demonstrating a secure client-server communication pattern using the [Oak Sessions library](https://github.com/project-oak/oak/tree/main/oak_sessions) over gRPC, designed to run within [GCP Confidential Space](https://cloud.google.com/confidential-computing/confidential-space/docs). It features attestation using real GCP Confidential Space tokens and performs client-side verification of both the token claims and session binding.

## How it Works

The prototype consists of a server (`main`) designed to run within a Confidential Space VM and a test client (`test_client`) that connects to it.

1.  **Secure Channel:** Communication uses the [Noise Protocol Framework (NN pattern)](https://noiseprotocol.org/noise.html#interactive-handshake-patterns-fundamental) implemented by Oak Sessions, providing end-to-end encryption and forward secrecy over a standard gRPC bidirectional stream.
2.  **Server Attestation:**
    * Upon connection, the server generates a fresh P-256 key pair (using Rust crypto via FFI).
    * It requests an attestation token (JWT format) from the Confidential Space agent running on the host, providing its public key (Base64-encoded) as the `eat_nonce`.
    * It includes this JWT in its initial handshake message (`AttestResponse`) to the client within a custom assertion map entry keyed `"custom_assertion"`.
    * During the handshake, Oak Sessions requires the server to "bind" the session ID by signing it with the private key corresponding to the public key sent in the nonce. This signature is sent in the `HandshakeResponse`'s `assertion_bindings` map. This proves possession of the private key associated with the attestation.
3.  **Client Verification:**
    * The client receives the server's `AttestResponse` containing the JWT assertion.
    * The Oak client session invokes the configured verifier (our Rust `FfiVerifier`).
    * The Rust `FfiVerifier` calls back to the C++ `verify_jwt_f` function, passing the JWT bytes and a pointer to the `MyVerifier` instance.
    * **JWT Verification (`MyVerifier::VerifyJwt`):**
        * Fetches Google's or Intel's public signing keys (JWKS) via the standard endpoint.
        * Uses the Tink library to verify the JWT signature against the fetched keys.
        * Verifies standard claims: `iss` (issuer), `aud` (audience), `typ` (type), and timestamps (`iat`, `nbf`, `exp`) against expected values.
        * Parses the full JWT payload using `nlohmann::json`.
        * Extracts and logs specific Confidential Computing claims: `hwmodel`, `secboot`, `dbgstat`, `swversion`, `oemid`, the nested container `image_digest`, and TCB status/dates.
    * **Policy Enforcement (`MyVerifier::EnforcePolicy`):**
        * Compares extracted claims against an `AttestationPolicy` struct.
        * Currently enforces:
            * `hwmodel` must match `INTEL_TDX` (for ITA provider).
            * `secboot` must be `true` (if `policy.require_secboot_enabled` is true).
            * `dbgstat` must be `"disabled"` (if `policy.require_debug_disabled` is true).
            * `image_digest` must match the value provided via the `--expected_image_digest` flag (if provided).
            * TCB dates must be at least as recent as the provided minimums (using strict `absl::Time` comparison).
        * If any required policy check fails, verification fails, an error is returned to Rust, and the connection is aborted by the Oak client session.
    * **Nonce Extraction (`MyVerifier::ExtractNonce`):** If verification succeeds, extracts the server's public key (raw bytes) from the `eat_nonce` claim and returns it to Rust.
    * **Session Binding Verification (Rust `FfiVerifiedAssertion::verify_binding`):**
        * The extracted public key is stored by the Rust verifier logic.
        * When the server sends its binding signature in the `HandshakeResponse`, the Rust code uses the `p256`/`ecdsa` crates to verify this signature against the session ID using the stored public key.
        * If the binding verification fails, an error is returned, and the connection is aborted by the Oak client session.
4.  **Application Data:** Once the handshake and all verifications succeed, the client sends an encrypted "Client says hi!" message, and the server responds with an encrypted "Server says hi back!".
5.  **FFI:** C++ and Rust code interact via C Foreign Function Interface (FFI) for:
    * Server: C++ calls Rust (`generate_key_pair`, `create_server_session_config`) to generate keys and create the `SessionConfig`, passing the token and receiving an opaque key handle. Rust handles key management and signing internally.
    * Client: C++ calls Rust (`create_client_session_config`) to create the `SessionConfig`, passing a context pointer (`MyVerifier*`) and a callback function pointer (`verify_jwt_f`). Rust calls back to C++ (`verify_jwt_f`) for JWT/policy verification and public key extraction. Rust handles binding verification internally.

## Implemented Features

* **Oak Noise Session:** Secure channel establishment using NN handshake pattern.
* **Real Attestation Token:** Server fetches JWT from GCP Confidential Space agent (supports both GCA and ITA providers).
* **Key Generation & Nonce:** Server uses a fresh P-256 public key (generated in Rust) as the JWT `eat_nonce`.
* **Session Binding:** Server signs the session ID using the corresponding private key (handled in Rust); client verifies this signature using the public key from the nonce (verification in Rust).
* **JWT Verification:** Client verifies JWT signature and standard claims using Tink in C++.
* **Claim Extraction:** Client extracts and logs key Confidential Computing claims (`hwmodel`, `secboot`, `dbgstat`, `swversion`, `oemid`, `image_digest`, TCB statuses and dates) in C++.
* **Attestation Policy Enforcement:** Client enforces checks for `hwmodel`, `secboot`, `dbgstat`, `image_digest`, and **minimum TCB dates** based on configurable `AttestationPolicy` in C++.
* **Build Integration:** A single `bazel run` command (`load_and_print_digest_runner`) builds the server, loads it into Docker, and prints the correct image manifest digest required by the client policy.
* **C++/Rust Integration:** Demonstrates using FFI for key generation, configuration, callbacks, and passing opaque handles between C++ and Rust components.
* **Code Refactoring:** Shared session logic (handshake, message pumping) factored out into `session_utils` library.

## How to Build and Run

### Prerequisites

* Bazel (or Bazelisk)
* Docker (running and accessible by the user running Bazel)
* Access to a GCP project configured for Confidential Space.
* `docker` CLI installed and configured (for `load_and_print_digest_runner` and manual push steps).

### Server

1.  **Build, Load Image, and Get Digest:**
    Run the following command from the `containers/gcp_prototype` directory. It builds the server container, loads it into your local Docker daemon using `docker load`, inspects the loaded image to find the manifest digest, and prints it.
    ```bash
    bazel run //:load_and_print_digest_runner
    ```

    Look for the output line:
    `Server Image Docker Digest: sha256:xxxxxxxx...`
    **Copy this digest value (the manifest digest).**

2.  **Tag and Push Image:**
    Tag the loaded image (`gcp_prototype:latest`) and push it to a container registry (like Artifact Registry) accessible by your Confidential Space deployment. Replace `$PROJECT`, `$LOCATION`, and `$REPO` with your specific values.

    ```bash
    # Example for Artifact Registry:
    export PROJECT=your-gcp-project-id
    export LOCATION=us-central1 # or your region
    export REPO=your-artifact-registry-repo-name
    export IMAGE_TAG=us-docker.pkg.dev/$PROJECT/$REPO/gcp_prototype:latest

    docker tag gcp_prototype:latest $IMAGE_TAG
    docker push $IMAGE_TAG
    ```
    *(Verify that the digest reported by `docker push` matches the one printed by the Bazel command).*

3.  **Deploy to Confidential Space:**
    Launch a Confidential Space VM using the image you just pushed (e.g., `$IMAGE_TAG`). Ensure the VM's firewall rules allow ingress traffic on TCP port 8000. Note the external IP address of the deployed VM.

### Client

1.  **Run the Test Client:**
    Execute the client using `bazel run`. Provide the server VM's external IP address and the image manifest digest you copied earlier. You can also optionally enforce minimum TCB dates.
    ```bash
    bazel run //:test_client -- \
        --server_address=<SERVER_EXTERNAL_IP> \
        --expected_image_digest=<COPIED_SHA256_DIGEST> \
        --dump_jwt=true
    ```
    (Replace `<SERVER_EXTERNAL_IP>` and `<COPIED_SHA256_DIGEST>`)

    If successful, the client logs will show the extracted claims, policy check results, successful binding verification in Rust, and the decrypted "Server says hi back!" message.

## Missing Features / Next Steps (b/452094015)

* **Endorsement Verification:** The client currently includes only a stub (`MyVerifier::Verify`) for verifying platform endorsements via the Oak interface. A robust implementation should validate the certificate chains and hardware measurements provided in the `Endorsements` proto, likely by integrating with a service like **Intel Trust Authority (ITA)** or performing manual checks against known roots and measurements.
* **Root of Trust Anchoring:** The client currently fetches and trusts Google's or Intel's JWKS keys via HTTPS without pinning or further validation. A production system might require stricter validation (e.g., certificate pinning, offline fetching, or using a trusted distributor) of the keys used to sign the attestation token.
* **Mutual Attestation:** Implement client-side attestation if required by the use case (would involve client key generation, nonce handling, and server-side verification).
* **Error Handling:** Improve error propagation, particularly across the FFI boundary and in gRPC handlers (e.g., return specific gRPC error codes instead of `LOG(FATAL)` or relying solely on `CHECK_OK`). Use `absl::Status` more consistently.
