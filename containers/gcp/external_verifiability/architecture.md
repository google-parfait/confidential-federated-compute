# External Verifiability — Architecture

A data processing pipeline defined by an FCP data access policy may specify
a transformation that uses the containers in this directory to offload
processing from an Oak-based TEE VM to a GCP-based Confidential Space TEE VM.
This document describes how to audit the containers and source code that are
used in this offloading process. For instructions on how to audit active access
policies more generally, please see
[inspecting_endorsements.md](../../docs/inspecting_endorsements.md).

This document covers the architecture of the GCP offloading pipeline from
the perspective of external verification. It explains what the two container
images are, how they communicate, how they are built, and how their
cryptographic identities are established.

For step-by-step auditing instructions, see
[instructions.md](instructions.md).

## System Overview

The GCP offloading pipeline consists of two separately-built TEE containers
that communicate over an end-to-end encrypted channel:

```
┌──────────────────────┐        ┌────────────────────────────────────┐
│  Oak Client (TEE)    │◄──────►│  GCP Server (Confidential Space)   │
│                      │  Oak   │                                    │
│  batched_inference_  │ Session│  batched_inference_gcp_main        │
│  oak_main            │  (E2E) │                                    │
│                      │        │                                    │
│  ┌───────────────┐   │        │  ┌─────────────────────────┐       │
│  │ policy.proto  │   │        │  │ llama.cpp + Model GGUF  │       │
│  │ Intel JWKS    │   │        │  │ Intel TDX attestation   │       │
│  └───────────────┘   │        │  └─────────────────────────┘       │
└──────────────────────┘        └────────────────────────────────────┘
```

Both containers are built from code in the `containers/gcp/` directory in this repository.

NOTE: At the time of this writing, the operations being offloaded to GCP server
involve only LLM inference, and the names of the containers as well as the
description that follows reflect that. That said, the offloading architecture
employed here and the mechanisms used for attestation and attestation
verification are general, and do not depend on the specific type of processing
being offloaded.

## The Oak Client Container

**Binary:** `batched_inference_oak_main`

**Where it runs:** Inside an [Oak](https://github.com/project-oak/oak) TEE.
The container has no direct network access — all external traffic is relayed
by a host-side proxy that provides a gRPC communication channel to the GCP
server container.

**What it does:** it implements the `Confidential Transform` API
and is generally used as one of the transforms in a larger data processing
pipeline described by a [data access policy](../../../docs/README.md).
It acts as a bridge between the data processing pipeline and the
GCP inference server:

1. Receives encrypted user data from the pipeline.
2. Decrypts the data and formulates inference requests inside the TEE.
3. Forwards them over an end-to-end encrypted and attested
   [Oak Session](https://github.com/project-oak/oak/tree/main/oak_sessions)
   channel to the GCP server.
4. Receives LLM responses.
5. Re-encrypts and returns them to the pipeline.

It is called the "client" container because it is a client of the GCP
server — the actual LLM inference happens on GCP.

### Digest Identity (Client)

The client container's cryptographic identity is the **raw SHA-256 hash of
the `container.tar` file** (`sha256sum container.tar`). This hash appears
in the [FCP](https://github.com/google-parfait/federated-compute)
Data Access Policy and is what the
[KMS](https://github.com/google-parfait/confidential-federated-compute/tree/main/kms)
verification infrastructure checks.

There is no OCI manifest digest involved — the tarball is not pushed to a
container registry.

### Build Inputs Embedded at Build Time (Client)

The client container embeds several artifacts at build time that are
critical for its security posture:

- **Intel Trust Authority JWKS** — fetched from
  `https://portal.trustauthority.intel.com/certs` during the build and
  embedded in the container at `/etc/pki/intel/jwks.json`. Used to verify
  the attestation evidence received from the GCP server and endorsed by the
  Intel Trust Authority (ITA).
- **Attestation Policy** (`policy.textproto`) — generated from
  [`server_image_registry.json`](../server_image_registry.json) by [`generate_policy.py`](../generate_policy.py), and embedded in the
  container at `/etc/confidential/policy.textproto`. Contains the approved
  GCP server container digests and security constraints.

### CI and Provenance (Client)

Built by the `reusable_build.yaml` GitHub Actions workflow. The workflow produces two distinct [in-toto](https://in-toto.io/) attestations via the GitHub Attestation API. Both attest over the same subject (the tarball binary, identified by its `sha256` digest), but carry different predicates:

1. **SLSA provenance** (predicate type `https://slsa.dev/provenance/v1`): Attested via [`actions/attest-build-provenance@v1`](https://github.com/actions/attest-build-provenance), which automatically generates the SLSA v1 provenance schema by introspecting the GitHub Actions runner environment.
2. **Custom Client Metadata** (predicate type `https://batched-inference.google.com/client-metadata/v1`): Attested via [`actions/attest@v1`](https://github.com/actions/attest), a generic in-toto attestation action that wraps the embedded `policy.textproto` configuration as a custom predicate.

Both attestations are available as DSSE bundles in the GitHub Attestation Store and are also published to the [Sigstore Rekor](https://docs.sigstore.dev/logging/overview/) transparency log.

## The GCP Server Container

**Binary:** `batched_inference_gcp_main`

**Image:** `us-docker.pkg.dev/private-inference/offloading/batched_inference`
(subject to change)

**Where it runs:** In a
[GCP Confidential Space](https://cloud.google.com/confidential-computing/confidential-space/docs)
TEE VM that uses Intel TDX, and has access to an NVIDIA H100 GPU
in Confidential Compute (CC) mode.

**What it does:** Uses the `llama.cpp` library and contains the Gemma
model weights (GGUF
format). Performs the actual LLM inference. The Oak client container
connects to it via an end-to-end encrypted Oak Session channel, verifying the
server's Intel Trust Authority (ITA) attestation token when the channel is
established. Subsequent inference requests are only sent over an attested and
encrypted connection.

### Digest Identity (Server)

The server container's cryptographic identity is the **Artifact Registry
manifest digest** — the SHA-256 of the OCI manifest JSON produced by
`docker push`. This is distinct from the **image ID** (the SHA-256 of
the image's JSON configuration), which the manifest references. Both
appear in the ITA token's `submods.container` claims, but the
attestation policy verifies `image_digest` (the manifest hash) because
it is the top-level content-addressable identifier assigned by the
registry.

> **Note on digest format:** The manifest digest is NOT the same as
> `sha256sum` of a `docker save` tarball. The manifest digest is the
> SHA-256 of the OCI manifest JSON, computed by the Docker client on
> **GitHub's runner** during `docker push`. Google's Artifact Registry
> simply stores what was pushed — it does not compute or influence the
> digest. The SLSA attestation covers the digest that the Docker client
> computed on GitHub's infrastructure. A `docker save` export produces a
> different tarball format (uncompressed layers, different structure),
> which is why its `sha256sum` differs. This is purely a Docker format
> distinction — both digests refer to the same image content, they are
> just computed over different serialization formats.

### CI and Provenance (Server)

Built by the `gcp_server_build.yaml` GitHub Actions workflow. The workflow produces two distinct [in-toto](https://in-toto.io/) attestations via the GitHub Attestation API. Both attest over the same subject (the container image, identified by its manifest digest), but carry different predicates:

1. **SLSA provenance** (predicate type `https://slsa.dev/provenance/v1`): Attested via [`actions/attest-build-provenance@v1`](https://github.com/actions/attest-build-provenance) using `subject-name` (the Artifact Registry image path) and `subject-digest` (the manifest digest from `docker push`).
2. **Custom Server Metadata** (predicate type `https://batched-inference.google.com/server-metadata/v1`): Attested via [`actions/attest@v1`](https://github.com/actions/attest), a generic in-toto attestation action that wraps the model name, attestation flavor, and ALTS configuration as a custom predicate.

Like the client container, both attestations are available as DSSE bundles in the GitHub Attestation Store and are also published to the Sigstore Rekor transparency log.

The server container image is pushed to a **private** Artifact Registry
and is not publicly downloadable (due to the fact that the model weights
have licensing constraints). However, both the SLSA provenance and the custom metadata are publicly
queryable via the GitHub Attestation API, enabling verification without
downloading the image binary.

## Attestation Policy

The attestation policy (`policy.textproto`) is the binding between the
two containers. It is generated at client build time from
[`server_image_registry.json`](../server_image_registry.json) and specifies:

| Field | Purpose |
|-------|--------|
| `verifier_type` | Must be `ITA` (Intel Trust Authority) |
| `allow_debug` | Must be `false` — no debug Confidential Space images |
| `skip_secboot` | Must be `false` — Secure Boot must be enforced |
| `expected_image_digest` | SHA-256 manifest digest(s) of approved server images |
| `expected_project_id` | GCP project that hosts the server (optional) |
| `expected_service_account` | Service account identity of the server VM (optional) |
| `max_sw_tcb_age_days` | Maximum age of the software TCB |
| `max_hw_tcb_age_days` | Maximum age of the hardware TCB |
| `min_swversion` | Minimum Confidential Space image version (e.g. `260500`) |

> [!NOTE]
> **Historical Advisory:** In earlier phases of rolling out these GCP-based
> containers to production, we erroneously endorsed development versions of the
> Oak client container that were configured for use with pre-release debug
> versions of Confidential Space images and did not implement all the constraints
> listed in the table above. The containers with this attestation policy were
> never actually used in production, even though they may have been published as
> authorized transformations in some data access policies.

**Key source files:**

- Policy schema:
  [`attestation_policy.proto`](../attestation_policy.proto) — the
  protobuf definition of all policy fields.
- Policy generator:
  [`generate_policy.py`](../generate_policy.py) — reads
  [`server_image_registry.json`](../server_image_registry.json), filters entries by model/attestation
  flavor/age, and writes the `policy.textproto` file.
- Client verifier:
  [`attestation_token_verifier.cc`](../attestation_token_verifier.cc) —
  the C++ implementation that verifies the ITA JWT signature using
  embedded Intel JWKS, extracts claims from the token, and enforces
  all policy constraints via `EnforcePolicy()`.

## Server Attestation via Intel Trust Authority

The GCP server container runs inside an
[Intel TDX](https://www.intel.com/content/www/us/en/developer/tools/trust-domain-extensions/overview.html)
Trust Domain on a
[GCP Confidential Space](https://cloud.google.com/confidential-computing/confidential-space/docs)
VM with NVIDIA H100 GPUs. The server is configured to use
[Intel Trust Authority (ITA)](https://www.intel.com/content/www/us/en/security/trust-authority.html)
as its attestation verifier.

**How attestation works (at runtime, on every Oak Session):**

When the server receives a new Oak Session handshake from a client, it
requests a fresh attestation token (JWT) from the GCP Confidential Space
attestation agent. The session's Noise protocol public key is included as
the `eat_nonce`, which forces a fresh ITA token for every session (tokens
have a 5-minute TTL and cannot be reused across sessions because each
session generates a unique nonce). The agent collects evidence from the
Intel TDX hardware and forwards it to Intel Trust Authority. ITA verifies
the TDX attestation quote (signed by Intel's hardware keys), and if valid,
signs and returns a JWT containing claims that **report the actual state**
of the server's execution environment. Within a single established
session, the attestation is done once at handshake time — subsequent
requests within that session reuse the already-verified connection.

| Claim | What ITA reports |
|-------|------------------|
| `hwmodel` | The hardware model. For ITA-attested deployments: `INTEL_TDX`. For GCA-attested deployments: `GCP_INTEL_TDX`. Our production deployments use ITA, so the expected value is `INTEL_TDX`. |
| `swname` | The software platform name. Must be `CONFIDENTIAL_SPACE` — ITA populates this only after validating the TDX RTMRs against Google's registered Confidential Space image reference values (see [ITA GCP CS integration](https://docs.trustauthority.intel.com/main/articles/articles/ita/integrate-gcp-cs.html)). |
| `swversion` | The Confidential Space image version (e.g. `260500`). |
| `secboot` | Whether Secure Boot is enabled (`true` / `false`) |
| `dbgstat` | The debug status (e.g., `disabled`, `disabled-since-boot`, or `enabled`) |
| `submods.container.image_digest` | The Artifact Registry manifest digest of the running server container |
| `tdx.cvm_compliance_status` | The CVM compliance status. Must be `gcp_compliant_cvm` — ITA sets this when the MRTD matches endorsed GCE guest firmware values. This confirms the firmware layer, while `swname` confirms the full Confidential Space image (see [ITA EAT profile](https://portal.trustauthority.intel.com/eat_profile.html#cvm_compliance_status)). |
| `tdx.gcp_attester_tcb_status` | The software TCB status and date |
| `tdx.attester_tcb_status` | The hardware TCB status and date |
| `submods.gce.project_id` | The GCP project ID the VM belongs to |
| `google_service_accounts` | The service account(s) the VM runs under |
| `submods.confidential_space.support_attributes` | Confidential Space operational attributes (string list) |
| `submods.confidential_space.monitoring_enabled.memory` | Whether memory monitoring is enabled (`true` / `false`) |

**ITA reports facts — it does not enforce policy.** ITA will truthfully
report `dbgstat: enabled` if debugging is enabled; it does not refuse to
issue a token. It is the **client's** responsibility to check these
claims against its embedded policy and reject connections that violate
the policy. See the next section for what the client enforces.

ITA signs the token with Intel's private key. The client verifies the
signature using the Intel JWKS embedded in the container at build time
(fetched from `https://portal.trustauthority.intel.com/certs` during
the GitHub Actions build).

### How ITA Ensures the VM Runs Confidential Space

A key question for auditors is: how do these checks ensure the client is
talking to a genuine Confidential Space TDX VM, and not just any TDX VM
running arbitrary software that fabricates `submods.confidential_space`
claims?

The answer lies in the TDX measurement registers. At VM boot, the TDX
hardware measures the guest firmware into the **MRTD** register. The
rest of the boot chain — the bootloader, kernel, and the entire
filesystem merkle tree root hash of the Confidential Space image — is
measured into the **RTMRs** (Runtime Measurement Registers).

These measurements are included in the TDX attestation quote that ITA
verifies against reference values published by Google in the public
GCS bucket
[`gs://gce_tcb_integrity`](https://console.cloud.google.com/storage/browser/gce_tcb_integrity):

- [`intel_rims/`](https://console.cloud.google.com/storage/browser/gce_tcb_integrity/intel_rims)
  — Confidential Space image reference values. Contains the full kernel
  command lines (with dm-verity filesystem root hashes) mapped to image
  names and versions. ITA checks the **RTMRs** against these. Only if
  they match does ITA populate `swname = "CONFIDENTIAL_SPACE"` —
  confirming the VM runs a genuine Confidential Space image.
- [`ovmf_x64_csm/tdx/`](https://console.cloud.google.com/storage/browser/gce_tcb_integrity/ovmf_x64_csm/tdx)
  — GCE guest firmware endorsements, keyed by MRTD hex digest. ITA
  checks the **MRTD** against these endorsed firmware values. If it
  matches, ITA sets `tdx.cvm_compliance_status = "gcp_compliant_cvm"`.

The Confidential Space attestation agent runs inside the measured VM
image. Before launching the workload container, it measures events into
the RTMRs that describe the container's identity and launch arguments.
ITA parses these measured events and — because it has already verified
that the filesystem measurement corresponds to a valid Confidential
Space image — trusts their contents and uses them to populate the
`submods.container` and `submods.confidential_space` fields in the JWT.
A different TDX VM image would have different MRTD and RTMR
measurements, and ITA would either refuse to issue a token or issue one
without the Confidential Space-specific claims. Therefore, a rogue VM
cannot fabricate these claims.

The Confidential Space images used in production
[are available in the GCP image registry](https://cloud.google.com/confidential-computing/confidential-space/docs/confidential-space-images)
as `confidential-space-XXXXX` where `XXXXX` corresponds to the software
version (`swversion` in the attestation policy; see the claims above).

> **Why ITA and not Google Cloud Attestation?** The
> `attestation_policy.proto` supports both `ITA` and `GCA` as verifier
> types. GCA (Google Cloud Attestation) uses Google's own attestation
> service, which would mean trusting Google to honestly report the
> server's state. For all production builds, for external verifiability
> purposes — where the explicit threat model is that Google is the
> party being audited — we use Intel Trust Authority (ITA).
> This means the attestation endorsement comes from Intel, not Google.

## Client-Side Attestation Verification

The Oak client container performs **offline verification** of
the server's attestation token. The client runs in a fully isolated TEE
with no direct network access except for the direct channel to the
server container on GCP, so it cannot fetch verification materials
at runtime directly from Intel's servers. In order to keep it simple,
everything the container needs is embedded into the container at build
time.

**What the client verifies on every connection:**

1. **JWT signature verification.** The client verifies the server's
   attestation token (JWT) signature using the Intel JWKS embedded at
   `/etc/pki/intel/jwks.json`. This proves the token was issued by Intel
   Trust Authority, not forged by Google or anyone else.

2. **Standard JWT claims.** The client checks `iss` (issuer), `aud`
   (audience), `exp` (expiration), and `nbf` (not before) to ensure the
   token is fresh and addressed to the correct audience.

3. **Confidential Computing claims.** The client extracts and enforces
   claims from the attestation token against the embedded
   `policy.textproto`:

   | Claim | Check | What it prevents |
   |-------|-------|-----------------|
   | `hwmodel` | Must be `INTEL_TDX` | Server running on non-TDX hardware |
   | `swname` | Must be `CONFIDENTIAL_SPACE` | Non-Confidential Space VM (see [How ITA Ensures the VM Runs Confidential Space](#how-ita-ensures-the-vm-runs-confidential-space)) |
   | `tdx.cvm_compliance_status` | Must be `gcp_compliant_cvm` | VM image not registered as a valid Confidential Space image |
   | `swversion` | Must be ≥ `min_swversion` in policy | Outdated Confidential Space image with known vulnerabilities |
   | `secboot` | Must be `true` | Tampered boot chain |
   | `dbgstat` | Must be `disabled` | Memory inspection via debug mode |
   | `submods.confidential_space.monitoring_enabled.memory` | Must be `false` | Memory monitoring bypassing TEE isolation |
   | `submods.container.image_digest` | Must match `expected_image_digest` in policy | Wrong or tampered server container |
   | TCB status | Must be `UpToDate`, meet `max_sw_tcb_age_days` / `max_hw_tcb_age_days` | Stale or unpatched platform |

4. **Session binding.** Upon connection, the server generates a fresh
   P-256 key pair and includes its public key as the `eat_nonce` in the
   attestation token request. After the client verifies the JWT, it
   extracts the public key from the `eat_nonce` claim. The server then
   signs the Noise session ID with the corresponding private key, and
   the client verifies this signature. This cryptographically binds the
   established encrypted session to the specific attested workload —
   preventing a MITM from relaying an attestation token from a different
   machine.

5. **Only after all checks pass** does the client decrypt user data and
   forward prompts to the server over the encrypted Oak Session channel.

## What the Auditor Verifies vs. What the Runtime Verifies

An external auditor's job is fundamentally different from the client's
runtime verification. Understanding this boundary is critical:

**The auditor verifies (offline, from public information):**

| What | How |
|------|-----|
| The client container was built from specific source code | SLSA provenance via GitHub Attestation API |
| The server container was built from specific source code | SLSA provenance via GitHub Attestation API |
| The source code does what it claims | Read the public repository at the attested commit |
| The Intel JWKS fetch URL is genuine | Inspect `MODULE.bazel` at the attested commit |
| The attestation policy has correct constraints | Read custom metadata from the attestation bundle |
| The verification logic is correctly implemented | Read the client verification code at the attested commit |

**The runtime verifies (the client does this on every Oak Session):**

| What | How |
|------|-----|
| The server runs on real Intel TDX hardware | ITA attestation token |
| The VM is a genuine Confidential Space image | ITA attestation token (`swname` via RTMR validation) |
| Confidential Computing mode is enabled | ITA attestation token |
| The Confidential Space image version is current | Client checks `swversion` ≥ `min_swversion` |
| Secure Boot is enforced | ITA attestation token |
| Debug mode is disabled | ITA attestation token |
| Memory monitoring is disabled | ITA attestation token (`monitoring_enabled.memory`) |
| The server container digest matches the policy | Client checks `image_digest` claim |
| The encrypted session is bound to the attested server | Session binding signature |

**What the auditor does NOT need to verify:**

- The auditor does **not** verify that the GCP Confidential Space VM
  image is legitimate — Intel Trust Authority does this at runtime by
  checking the TDX MRTD and RTMR measurements against reference values
  (see [How ITA Ensures the VM Runs Confidential Space](#how-ita-ensures-the-vm-runs-confidential-space)).
- The auditor does **not** verify that TDX hardware is genuine — Intel's
  attestation infrastructure handles this.
- The auditor does **not** need to trust that Google's Confidential
  Space image is legitimate — Intel independently verifies the platform
  against reference values published in the public GCS bucket
  [`gs://gce_tcb_integrity/intel_rims`](https://console.cloud.google.com/storage/browser/gce_tcb_integrity/intel_rims).
- The auditor does **not** need to download or inspect the server
  container image — they verify the source code that produced it.

The auditor's role is to verify that the **code and configuration are
correct** — that the verification logic the client runs is sound, that
the policy constraints are appropriate, and that the build provenance
is legitimate. The actual runtime enforcement is handled by the client container
(using Intel's attestation) on every single connection.

## Digest Flow

The complete chain of trust has two phases: build-time and runtime.

**Build-time (establishing the chain):**

```
GCP Server image built and pushed by GitHub Actions
  → docker push computes manifest digest on GitHub's runner
  → SLSA provenance attested (digest → commit → workflow run)
  → manifest digest recorded in server_image_registry.json (manually)
  → generate_policy.py reads registry, produces policy.textproto
  → Oak client container.tar built with embedded policy + Intel JWKS
  → sha256sum of container.tar attested via SLSA provenance
  → container.tar digest recorded in
    FCP Data Access Policy
  → access policy endorsed and published to Rekor transparency log
```

**Runtime (enforcing the chain):**

```
Client TEE boots
  → loads embedded policy.textproto and Intel JWKS
  → connects to GCP server via Oak Session handshake
  → server requests ITA attestation token (fresh P-256 key as nonce)
  → ITA verifies TDX hardware, Confidential Space image, container digest
  → ITA signs and returns JWT to server
  → server sends JWT + session binding signature to client
  → client verifies JWT signature using embedded Intel JWKS
  → client checks all policy constraints (hardware, debug, digest, TCB)
  → client verifies session binding (nonce key matches session key)
  → only then: client decrypts user data and sends prompts to server
```

## Attestation Storage Architecture

The evidence required for external verification is stored across multiple independent systems to ensure availability and tamper-resistance:

1. **GitHub Attestation Store:** The primary storage for both SLSA provenance and custom metadata DSSE bundles. It is queryable via the GitHub API (`GET /repos/.../attestations/sha256:{digest}`).
2. **Sigstore / Fulcio:** Acts as the Certificate Authority. It issues the short-lived X.509 certificate that binds GitHub's OIDC identity (the specific workflow run) to the ephemeral key used to sign the attestation bundle.
3. **Sigstore / Rekor:** An append-only transparency log. Every attestation (both SLSA and custom metadata) is anchored here with an inclusion proof, providing a tamper-proof public record.

## Distributed Trust Model

The system distributes trust across independent parties so that no
single entity (including Google) can subvert it:

### Intel Trust Authority

**Trusted for:** Runtime attestation of the server's execution
environment.

ITA verifies that the server is running on genuine TDX hardware, with
Confidential Computing enabled, Secure Boot enforced, debug mode
disabled, and a known Confidential Space image. ITA also reports the
exact container image digest running inside the VM.

**What Google cannot do:** Google cannot forge an ITA attestation token
(Intel signs them). Google cannot make ITA report a different type of
Confidential Space image than what is actually running. Google cannot
make ITA claim the server is running on TDX hardware when it is not.

**What ITA does NOT verify:** ITA does not verify that the container
image's *source code* is correct — it only reports the container's
digest. The source-to-digest link comes from SLSA provenance (GitHub's
domain).

### GitHub (CI and Source Hosting)

**Trusted for:** Build provenance and source code transparency.

GitHub Actions runs the CI builds on GitHub's own infrastructure (not
Google's). SLSA provenance, attested via `actions/attest-build-provenance`,
cryptographically links each container digest to the exact source commit
and workflow run that produced it. The provenance is signed by
Sigstore/Fulcio using GitHub's OIDC identity.

**What Google cannot do:** Google cannot forge SLSA provenance (GitHub
signs it). Google cannot run the CI build on its own servers and claim
it ran on GitHub. Google cannot modify the source code without it being
visible in the public git history.

### Sigstore / Rekor

**Trusted for:** Tamper-proof public record of all attestations.

The Rekor transparency log is append-only and publicly auditable.
Provenance attestations and data access policy endorsements are anchored
in Rekor, making it impossible to retroactively modify or delete them.

**What Google cannot do:** Google cannot remove or alter a Rekor entry
after it has been logged.

### What Google Controls

Google writes the source code, operates the GCP infrastructure (VMs,
Artifact Registry, networking), initiates commits, and triggers builds.
But the attestation chain is designed so that Google's actions are
**observable and constrained**:

- Every code change is visible in the public GitHub repository.
- Every build is traced to a specific commit via SLSA provenance.
- Every deployed container's digest is reported by the Confidential
  Space attestation agent running inside an Intel TDX-verified VM.

## Build Non-Reproducibility

The external verifiability architecture for GCP-based deployments
described here is built on the assumption that neither the client, nor
server container builds are guaranteed to be reproducible:

- The client container fetches Intel JWKS dynamically at build time via
  `curl_file` in `MODULE.bazel`. If Intel rotates keys between builds,
  the fetched content changes and the container hash changes.
- The server container depends on CUDA, model weights, and other
  dependencies that are potentially not fully hermetic.

Consequently, the external validation mechanism described here does NOT
rely on rebuilding from source and comparing hashes. Instead, SLSA
provenance ties each specific built binary to a specific source commit,
without requiring anyone to recreate the build.

The provenance chain is: digest → SLSA attestation → source
commit → public source code. An auditor follows this chain to verify
what code produced a given container, then reads that code to verify
its behavior.

## Related Documentation

- [instructions.md](instructions.md) — Step-by-step external auditing
  workflow.
- [../README.md](../README.md) — GCP containers build and run
  instructions.
