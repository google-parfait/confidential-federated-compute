# Components for GCP Confidential Space Offloading

This directory contains C++/Rust components for offloading computations (currently batched Large Language Model (LLM) inference) from an on-premise TEE runing under Oak to a GCP Confidential Space TEE.
It uses the [Oak Sessions library](https://github.com/project-oak/oak/tree/main/oak_sessions) over gRPC, designed to run with the server in [GCP Confidential Space](https://cloud.google.com/confidential-computing/confidential-space/docs) and the client within an **Oak TEE**.
It features attestation using real GCP Confidential Space tokens and performs client-side verification of both the token claims and session binding.

## Architecture

The system consists of three main components (as well as the suporting building blocks):

1.  **GCP Server (Confidential Space TEE):** Runs the `batched_inference_gcp_main` binary. **Hosts the LLM model and performs inference using** `llama.cpp` **upon receiving an encrypted request.** Authenticates itself to the client via the Oak Session handshake.
2.  **Oak Client (Oak TEE):** Runs the `batched_inference_oak_main` binary inside an Oak-compatible container. It is fully isolated and has **no direct network access**. **It acts as a gRPC server to the Host for receiving prompts.** implementing FCP Confidential Transform.
3.  **Host Proxy (Untrusted):** Runs on the host machine that launches the Oak TEE. It provides a gRPC proxy service that forwards encrypted messages between the isolated Oak Client and the external GCP Server. Not included here (it's environment-specific).

## How it Works

The prototype consists of a server binary designed to run within a Confidential Space TEE and a client binary runing in an Oak TEE that connects to it via the Host Proxy.

1.  **Secure Channel:** Communication uses the [Noise Protocol Framework (NN pattern)](https://noiseprotocol.org/noise.html#interactive-handshake-patterns-fundamental) implemented by Oak Sessions, providing end-to-end encryption and forward secrecy over a standard gRPC bidirectional stream.
2.  **Server Attestation:**
    * Upon connection, the server generates a fresh P-256 key pair (using Rust crypto via FFI).
    * It requests an attestation token (JWT format) from the Confidential Space agent, providing its public key (Base64-encoded) as the `eat_nonce`.
    * It includes this JWT in its initial handshake message (`AttestResponse`) within a custom assertion map entry keyed `"custom_assertion"`.
    * **Session Binding:** During the handshake, Oak Sessions requires the server to "bind" the session ID by signing it with the private key corresponding to the public key in the nonce. This signature is sent in the `HandshakeResponse`'s `assertion_bindings` map, proving possession of the private key associated with the attestation.
3.  **Client Verification (Hardened Offline):**
    * The client runs in a hardened, offline environment, using **Root of Trust** (Intel Trust Authority JWKS) and **Policy** files, currently baked immutably into its container image at build time, to verify the server's attestation report.
    * **JWT Verification:** On startup, it uses these local files to cryptographically verify the server's token signature and standard claims (`iss`, `aud`, `exp`, `nbf`).
    * **Policy Enforcement:** It extracts and enforces specific Confidential Computing claims against the baked-in policy:
        * `hwmodel` must match `INTEL_TDX`.
        * `secboot` must be enabled.
        * `dbgstat` must be `disabled`.
        * `image_digest` (the nested container digest) must match the expected server hash.
        * TCB status must be `UpToDate` and meet minimum date requirements.
    * **Binding Verification:** If the JWT is valid, the client extracts the public key from the `eat_nonce` claim. The Rust session layer then uses this key to verify the session binding signature from the `HandshakeResponse`, ensuring the established Noise session is bound to the attested workload.
4.  **Prompt Offload & Inference:** Once the handshake and all verifications succeed, the client is ready to forward inference calls to the server. When the client receives inference requests via the Confidnetial Transform API, it encrypts this message and sends it to the GCP Server for execution. The GCP Server decrypts the message, **runs the prompt through the LLM** (`llama.cpp`), and sends the encrypted **generated response** back to the client, which relays the reply again using the Confidnetial Transform protocol to its caller.
5.  **FFI:** C++ and Rust code interact via C Foreign Function Interface (FFI) for key management, session configuration, and callbacks.
6.  **Result Delivery:** The Oak Client decrypts the response and returns it to the host.

## Implemented Features

* **Oak Noise Session:** Secure channel establishment using NN handshake pattern.
* **Host Proxy:** Untrusted host relays encrypted traffic between isolated TEE client and external server.
* **Real Attestation Token:** Server fetches JWT from GCP Confidential Space agent.
* **Session Binding:** Server proves possession of the attestation-bound key.
* **Hardened Offline Verification:** Client performs full cryptographic verification using only baked-in keys and policy.
* **Build Integration:** Bazel rules to bake JWKS and generated policy protos into the client container bundle.
* **LLM Inference:** Integrated `llama.cpp` for CPU- and GPU-based inference (tested with Gemma-270M).

## How to Build and Run

### Prerequisites

* Bazel (or Bazelisk)
* Docker (running and accessible by the user running Bazel)
* Access to a GCP project configured for Confidential Space.
* `docker` CLI installed and configured.
* **Hugging Face Token:** Required to fetch model weights during the build. Set it in your environment or pass it to Bazel.

### Server

1.  **Build, Load Image, and Get Digest:**
    Run the following command from the `containers/gcp` directory. **You must provide your Hugging Face token to fetch the model weights.** It builds the server container, loads it into your local Docker daemon, and prints the manifest digest.
    ```bash
    bazelisk run --repo_env=HF_TOKEN=hf_YOUR_TOKEN :load_and_print_digest_runner_batched_inference_gcp_ita
    ```
    **Copy this digest value.**

2.  **Tag and Push Image:**
    Tag and push the loaded image (`batched_inference:ita_latest`) to your Artifact Registry.
    ```bash
    docker tag batched_inference:ita_latest $IMAGE_TAG
    docker push $IMAGE_TAG
    ```

3.  **Deploy to Confidential Space:**
    Launch a Confidential Space VM using the image you just pushed, ensuring TCP port 8000 is open.

### Client (Oak TEE)

1.  **Build Client Bundle:**
    Build the offline client bundle from the `containers/gcp` directory, providing the server's digest as a build flag. This bakes the digest into the client's immutable policy.
    ```bash
    bazel build :batched_inference_oak_ita_oci_runtime_bundle --//:server_digest=sha256:YOUR_COPIED_DIGEST
    ```
    The output will be a tarball (e.g., `bazel-bin/containers/gcp/batched_inference_oak_ita_oci_runtime_bundle.tar`).

2.  **Run with Host Proxy:**
    Execute your host binary (which must launch the Oak TEE, pass the client bundle to it, and set up port forwarding), pointing it to the running GCP server, **and include the prompt**.
    ```bash
    ./your_host_binary \
      --client_container_path=/path/to/batched_inference_oak_ita_oci_runtime_bundle.tar \
      --gcp_server_address=<GCP_SERVER_IP>:8000 \
      --prompt="Tell me a joke about semi-conductors."
    ```
