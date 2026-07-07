# External Verifiability — Auditing Instructions

A data processing pipeline defined by an FCP data access policy may specify
a transformation that uses the containers in this directory to offload
processing from an Oak-based TEE VM to a GCP-based Confidential Space TEE VM.
This document provides a step-by-step workflow for an external auditor to
verify that the GCP offloading pipeline containers were built from the
public source code without tampering. For instructions on how to audit active
access policies more generally, please see
[inspecting_endorsements.md](../../docs/inspecting_endorsements.md).

For background on the system architecture, see
[architecture.md](architecture.md).

## Prerequisites

- Python 3.6+
- A copy of this repository (or at minimum the
  `containers/gcp/` directory)
- Set up the virtual environment and install dependencies:
  ```console
  $ python3 -m venv venv
  $ source venv/bin/activate
  $ pip install -r requirements.txt
  ```

## Overview

The auditing workflow follows a chain of trust from the published data
access policy down to the source code of both containers:

```
Step 0: Obtain client container digest from data access policy
  ↓
Step 1: Trace client digest → source commit and verify provenance cryptographically
  ↓
Step 2: Inspect the client container (JWKS, policy, verification code)
  ↓
Step 3: Extract server container digest(s) from the policy
  ↓
Step 4: Trace server digest → source commit and verify provenance cryptographically
  ↓
Step 5: Inspect the server container (no egress, no persistent storage)
```

After completing all steps, the auditor has cryptographic proof that both
containers were built from specific, publicly auditable source commits on
GitHub Actions — not on Google's infrastructure.

---

## Step 0. Obtain the Client Container Digest

**Goal:** Get the SHA-256 digest of the Oak client container that is
authorized to process user data.

**How:** Follow the existing documentation in
[inspecting_endorsements.md](../../docs/inspecting_endorsements.md) to
query the Rekor transparency log for endorsed data access policies. Use
the `explain_fcp_attestation_record` tool
(`tools/explain_fcp_attestation_record/`) to parse the binary access
policy and extract the authorized container digests.

**Output:** A SHA-256 hex string, e.g.:
```
44516b40fd067cf3c01f1dc1c3890d2002add6d4fd21c5532bc5740346915138
```

---

## Step 1. Trace Client Container Digest to Source Commit

> **NOTE:** While these instructions focus on using the tool on the GCP
> offloading client container, the same `trace_digest.py` tool can be used
> to find and inspect the source code for any other endorsed container that
> may occur in an access policy.

**Goal:** Given the client container digest, find the exact source commit
it was built from and cryptographically verify the associated SLSA provenance evidence.

**Tool:** Use `trace_digest.py` from the `containers/gcp` directory:

```console
$ source venv/bin/activate
$ python3 trace_digest.py 44516b40fd067cf3c01f1dc1c3890d2002add6d4fd21c5532bc5740346915138
======================================================================
 EXTERNAL VERIFICATION START: sha256:44516b40fd067cf3c01f1dc1c3890d2002add6d4fd21c5532bc5740346915138
======================================================================

[*] STEP 1/4: Querying Rekor Transparency Log...
  -> Rekor indicates 5 valid certificate(s). First GITHUB_SHA: 55cbbc38dee3eda06eba3a9152c5b967d996f909
  -> Rekor indicates WORKFLOW_URI: https://github.com/.../build.yaml@refs/heads/main
  -> Identified as client container.

[*] STEP 2/4: Fetching Attestation Bundles...
  -> Found 5 attestation(s) via GitHub API.
  ...

[*] STEP 3/4: Cryptographic Verification via Sigstore...
  -> Attestation 1: Cryptographic signatures (Fulcio, Rekor, DSSE, OIDC Identity) VERIFIED.
  ...

[*] STEP 3.5/4: Independent Rekor Transparency Log Cross-Check...
  -> Rekor confirms 5 transparency log entry/entries for this digest.
  -> Attestation 1 payload strictly matched to Rekor transparency log.
  ...

[*] STEP 4/4: Extracting Provenance & Metadata...
  -> Subject: container.tar (sha256:44516b40...)

======================================================================
 [V] VERIFICATION SUCCESS 
======================================================================
Source Commit(s):
  - 55cbbc38dee3eda06eba3a9152c5b967d996f909
    https://github.com/google-parfait/confidential-federated-compute/tree/55cbbc38dee3eda06eba3a9152c5b967d996f909/containers/gcp

Custom Metadata:
  - Type: https://batched-inference.google.com/client-metadata/v1
      attestation_policy: {'expected_image_digest': ['sha256:...'], 'verifier_type': 'ITA', ...}
```

**What happens under the hood:**

1. **Querying Rekor Transparency Log:** The script queries Rekor to identify the container type (client or server) based on the workflow URI in the certificate extensions.
2. **Fetching Attestation Bundles:** The script fetches attestations from the GitHub Attestations API.
3. **Cryptographic Verification:** The script executes a **complete cryptographic verification suite** against the DSSE bundles using the official `sigstore` verification engine:
   - **Fulcio Certificate Chain**: Validates the short-lived X.509 leaf certificate and its temporal validity.
   - **DSSE Envelope Signature**: Verifies the ECDSA signature over the payload.
   - **OIDC Identity**: Asserts that the SAN of the leaf certificate matches the trusted repository identity (`google-parfait/confidential-federated-compute`).
   - **Independent Rekor Cross-Check**: It independently verifies the Inclusion Proof against the Rekor transparency log for *every* attestation.
4. **Extracting Provenance & Metadata:** Upon successful verification, the script decodes the payloads to extract the source commit, verify the `subject.digest.sha256` matches the requested digest, and automatically extracts any attached **custom metadata** (such as the attestation policy details) for easy inspection.

**What to check:**

Ensure you see the following in the tool's output:
- `  -> Attestation 1: Cryptographic signatures (Fulcio, Rekor, DSSE, OIDC Identity) VERIFIED.`

**No authentication required.** The GitHub Attestation API is public for
public repositories.

---

## Step 2. Inspect the Client Container Code

**Goal:** Confirm that the client container enforces the correct security constraints, connects to the genuine Intel Trust Authority, and properly implements the attestation verification logic.

**Convenience Note:**
The `trace_digest.py` tool from Step 1 automatically extracts and prints the custom attestation metadata, which contains the finalized policy embedded in the container (including the expected server digests) as well as the JWKS endpoint. However, manual source inspection provides the ultimate ground truth.

**Instructions:**
First, open your web browser and navigate to the GitHub repository at the exact source commit hash printed at the end of Step 1. Because the cryptographic provenance ties the container directly to this commit, you can trust that this is the exact code running in the container.

Then, inspect the source code to manually verify the following properties:

1. **The Intel JWKS Endpoint:**
   - Find the `curl_file` rule in `MODULE.bazel` (or equivalent build rule) that fetches the Intel JWKS.
   - Verify the URL is Intel's published ITA endpoint: `https://portal.trustauthority.intel.com/certs`
   - *Why this works:* Since the build ran on GitHub Actions, Google could not intercept this fetch — the JWKS in the container is exactly what Intel served.

2. **The Attestation Policy Logic:**
   Inspect the attestation policy generation pipeline at the attested
   commit:
   - Review `server_image_registry.json` — the source of truth for
     approved server container digests.
   - Review `generate_policy.py` — the script that reads the registry
     and generates the final `policy.textproto`.
   - Check the `BUILD` file to see how `generate_policy.py` is invoked
     during the build.
   - Verify that the resulting policy fields match the expected values
     listed in the [Attestation Policy](architecture.md#attestation-policy)
     section of architecture.md.

3. **The Verification Code:**
   - Inspect the source code to verify that it performs the necessary
     verifications described in
     [Client-Side Attestation Verification](architecture.md#client-side-attestation-verification).

---

## Step 3. Extract Server Container Digests

**Goal:** Extract the GCP server container image digest(s) that the
client is configured to accept.

**Process:** You can obtain these digests in two ways:
1. **From Step 1 Output:** Review the custom metadata output printed by `trace_digest.py` in Step 1. The `attestation_policy` object contains the `expected_image_digest` list.
2. **From Source Inspection (Step 2):** Read the `server_image_registry.json` file at the attested commit to manually verify the registry.

These are Artifact Registry manifest digests in the format `sha256:...`.

Example:
```json
"expected_image_digest": ["sha256:fe822f41abc123..."]
```

These digests are the inputs to the next server tracing step.

---

## Step 4. Trace Server Digest to Source Commit

**Goal:** Given a server container digest extracted from the policy, find the exact
source commit it was built from and cryptographically verify the server's SLSA provenance.

**Tool:** Use the same `trace_digest.py` script:

```console
$ python3 trace_digest.py fe822f41abc123...
```

The script uses the same flow as for client digests. It first queries the Rekor transparency log to auto-detect the container type. Then, it queries the GitHub Attestation API and automatically verifies the server's DSSE bundle.

**What happens under the hood:** The Rekor auto-detection step identifies the digest as a server container. The GitHub Attestation
API is then queried with the server's manifest digest. The SLSA provenance
links the digest to a source commit and a `gcp_server_build.yaml`
workflow run, while checking the ECDSA signature and the OIDC SAN identity.

**What to check:**

Look at the output of `trace_digest.py` during the server trace. Ensure you see:
- `  -> Attestation 1: Cryptographic signatures (Fulcio, Rekor, DSSE, OIDC Identity) VERIFIED.`
- The SLSA provenance `subject.digest.sha256` exactly matches the expected server digest.

**Note:** The server container image itself is hosted in a private
Artifact Registry and is not downloadable (it contains model weights
with licensing constraints). However, the SLSA provenance is publicly
queryable — you do not need access to the image to verify its
provenance.

## Step 5. Inspect the Server Container Code

**Goal:** Confirm that the server container behaves securely and respects data privacy once it receives the decrypted user data from the client container.

**Approach (Source Inspection):**

Using the exact source commit identified in Step 4, inspect the server container's source code on GitHub.

**What to check in the source code:**

1. **No External Egress:** Verify that the server container does not establish external network connections to send data outside the Confidential Space environment. The container should only respond to the client container.
2. **No Persistent Storage:** Verify that the server does not log sensitive user data to disk, write to external databases, or persist data across requests in a way that could leak information between users or after the session ends.
3. **Data Handling:** Ensure that the server container only processes the prompts to run the inference model and returns the result without performing any unauthorized side-effects.

---

## Summary

After completing all steps, the auditor has established:

1. **The client container digest** came from a published, endorsed data
   access policy.
2. **The client container** was built from a specific public source
   commit on GitHub Actions — not on Google's infrastructure.
3. **The Intel JWKS** used by the client is fetched from Intel's
   published endpoint during the GitHub Actions build.
4. **The attestation policy** enforces appropriate security constraints
   and is readable in the public source.
5. **The server container digest** is embedded in the client's policy
   and traceable to its own source commit.
6. **The server container** was also built from a public source commit
   on GitHub Actions.
