# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared core logic for fetching and verifying SLSA provenance via Sigstore."""

import urllib.request
import urllib.error
import json
import base64
import sys
import subprocess
import tempfile
import os
import hashlib
import re

class VerificationError(Exception):
    """Raised when a verification check fails."""
    pass

def fail(reason):
    print(f"\n" + "="*70)
    print(f" [X] VERIFICATION FAILED ")
    print(f"="*70)
    print(f"Reason: {reason}\n")
    sys.exit(1)

def success(commits, custom_metadata, workflows):
    print(f"\n" + "="*70)
    print(f" [V] VERIFICATION SUCCESS ")
    print(f"="*70)
    if commits:
        print(f"Source Commit(s):")
        for c in commits:
            print(f"  - {c}")
            print(f"    https://github.com/google-parfait/confidential-federated-compute/tree/{c}/containers/gcp")
    if custom_metadata:
        print(f"\nCustom Metadata:")
        for ptype, pdata in custom_metadata:
            print(f"  - Type: {ptype}")
            for k, v in pdata.items():
                print(f"      {k}: {v}")
    if workflows:
        print(f"\nGitHub Actions Run ID(s):")
        for w in workflows:
            print(f"  - {w}")
    print()

def fetch_attestations_from_github(digest):
    """
    Queries the GitHub API and returns the raw list of attestation dictionaries.

    Args:
        digest (str): The raw SHA256 hex digest (without 'sha256:' prefix).

    Returns:
        list[dict]: A list of attestation bundles.

    Raises:
        Exception: If the GitHub API is unreachable or returns invalid data.
    """
    url = f"https://api.github.com/repos/google-parfait/confidential-federated-compute/attestations/sha256:{digest}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        raise Exception(f"HTTP Error {e.code}: {e.reason}. Could not find attestations. Was this image built by CI?")
    except Exception as e:
        raise Exception(f"Error fetching attestations: {e}")

    if not isinstance(data, dict):
        raise Exception(f"GitHub API returned unexpected data type: {type(data).__name__}. Expected a JSON object.")

    attestations = data.get("attestations")
    if not attestations or not isinstance(attestations, list):
        raise Exception("API returned HTTP 200 but no valid 'attestations' list was found in the response.")

    return attestations

def verify_attestation_signatures(attestations, digest, repository="google-parfait/confidential-federated-compute", offline=False):
    """
    Writes bundles to disk and uses sigstore CLI to cryptographically verify them.

    Args:
        attestations (list[dict]): A list of attestation dictionaries.
        digest (str): The raw SHA256 hex digest (without 'sha256:' prefix).
        repository (str, optional): The expected GitHub repository.
        offline (bool, optional): If True, passes --offline to sigstore CLI,
            which verifies the bundle strictly using its embedded Rekor transparency
            log entry and Fulcio certificate without making external network requests.

    Returns:
        None (Fails process on validation error).
    """
    for idx, attestation in enumerate(attestations):
        if not isinstance(attestation, dict):
            fail(f"Attestation {idx+1} is not a valid JSON object.")
        bundle = attestation.get("bundle")
        if not bundle or not isinstance(bundle, dict):
            fail(f"Attestation {idx+1} is missing a valid 'bundle' object.")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
            json.dump(bundle, tf)
            tf_name = tf.name

        try:
            cmd = [
                sys.executable, "-m", "sigstore", "verify", "github",
                "--bundle", tf_name,
                "--repository", repository,
                f"sha256:{digest}"
            ]
            if offline:
                cmd.insert(5, "--offline")


            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                fail(f"Cryptographic signature check FAILED for attestation {idx+1}.\n\nSigstore output:\n{result.stderr}")

            print(f"  -> Attestation {idx+1}: Cryptographic signatures (Fulcio, Rekor, DSSE, OIDC Identity) VERIFIED.")
        except FileNotFoundError:
            fail("sigstore module not found. Did you run 'pip install -r requirements.txt'?")
        except Exception as e:
            fail(f"Error running sigstore: {e}")
        finally:
            os.remove(tf_name)

# ---------------------------------------------------------------------------
# Rekor transparency log helpers & client fallback (Rekor-to-Oak bridge)
# ---------------------------------------------------------------------------

def get_rekor_entries_for_digest(digest):
    """
    Given a sha256 digest, query the Rekor index and fetch all associated log entries.
    Returns a list of raw JSON entry dictionaries.

    Raises on any HTTP or parsing error — callers are responsible for handling failures.
    """
    if digest.startswith("sha256:"):
        digest = digest[7:]

    rekor_url = "https://rekor.sigstore.dev/api/v1/index/retrieve"
    rekor_req = urllib.request.Request(
        rekor_url,
        data=json.dumps({"hash": f"sha256:{digest}"}).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(rekor_req, timeout=10) as resp:
        rekor_entry_ids = json.loads(resp.read().decode())

    if not rekor_entry_ids or not isinstance(rekor_entry_ids, list):
        raise Exception("Rekor index returned no entry IDs or invalid format.")

    entries = []
    for entry_id in rekor_entry_ids:
        req2 = urllib.request.Request(
            f"https://rekor.sigstore.dev/api/v1/log/entries/{entry_id}",
            headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(req2, timeout=10) as resp2:
            entry_data = json.loads(resp2.read().decode())

        if not isinstance(entry_data, dict):
            raise Exception(f"Rekor entry {entry_id} response is not a valid JSON object.")
        if entry_id not in entry_data:
            raise Exception(f"Rekor entry {entry_id} response is missing the entry ID key.")

        entries.append(entry_data[entry_id])

    return entries

def extract_cert_metadata(entry):
    """
    Parses a Rekor entry to extract the GITHUB_SHA and WORKFLOW_URI
    from the Fulcio certificate extensions.
    Returns a tuple (GITHUB_SHA, WORKFLOW_URI).
    """
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend

    try:
        body = json.loads(base64.b64decode(entry['body']).decode('utf-8'))
        # NOTE: Takes only the FIRST signature's verifier certificate (index [0]).
        # Sigstore DSSE entries in practice have exactly one signature per entry.
        cert_pem = base64.b64decode(body['spec']['signatures'][0]['verifier'])
    except Exception as e:
        raise Exception(f"Failed to parse DSSE envelope or find verifier certificate: {e}")

    try:
        cert = x509.load_pem_x509_certificate(cert_pem, default_backend())
    except Exception as e:
        raise Exception(f"Failed to load X.509 certificate: {e}")

    github_sha = None
    workflow_uri = None

    def decode_asn1_string(der_bytes):
        if not isinstance(der_bytes, bytes):
            return str(der_bytes)
        # Sigstore extensions are usually ASN.1 UTF8String (0x0C) or OCTET STRING (0x04)
        if der_bytes[0] in (0x0c, 0x04):
            length = der_bytes[1]
            offset = 2
            if length >= 128:
                num_bytes = length & 0x7f
                length = int.from_bytes(der_bytes[offset:offset+num_bytes], 'big')
                offset += num_bytes
            return der_bytes[offset:offset+length].decode('utf-8', errors='ignore')
        return der_bytes.decode('utf-8', errors='ignore')

    for ext in cert.extensions:
        oid = ext.oid.dotted_string
        try:
            val_str = decode_asn1_string(ext.value.value)

            if oid == "1.3.6.1.4.1.57264.1.10": # Provider-generic commit SHA
                github_sha = val_str
            elif oid == "1.3.6.1.4.1.57264.1.3" and not github_sha: # Deprecated GitHub SHA
                github_sha = val_str
            elif oid == "1.3.6.1.4.1.57264.1.18": # Caller Repository Workflow
                workflow_uri = val_str
        except AttributeError:
            pass
        except UnicodeDecodeError:
            pass

    if not github_sha:
        raise Exception("Could not find GITHUB_SHA in certificate extensions (missing OID .1.10 or .1.3).")
    if not workflow_uri:
        raise Exception("Could not find WORKFLOW_URI in certificate extensions (missing OID .1.18).")

    return github_sha, workflow_uri

def fetch_oak_bundle_by_sha(github_sha, target_digest):
    """
    Given a GITHUB_SHA, enumerate the Oak GCS bucket for all packages built at that commit,
    download their attestation bundles, and return the one matching the target digest.
    """
    if target_digest.startswith("sha256:"):
        target_digest = target_digest[7:]

    # GCS JSON API endpoint to list objects with a specific prefix
    list_url = f"https://storage.googleapis.com/storage/v1/b/oak-bins/o?prefix=provenance/{github_sha}/"
    req = urllib.request.Request(list_url, headers={"Accept": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None # Bucket or prefix not found
        raise Exception(f"HTTP Error querying Oak bucket: {e.code} {e.reason}")

    items = data.get("items", [])
    if not items:
        return None

    for item in items:
        if not item["name"].endswith("attestation.jsonl"):
            continue

        # Download the bundle
        dl_url = item["mediaLink"]
        try:
            with urllib.request.urlopen(dl_url, timeout=10) as res:
                bundle = json.loads(res.read().decode())

            # Extract DSSE payload and find subject digest
            dsse = bundle.get("dsseEnvelope", {})
            payload_b64 = dsse.get("payload", "")
            if not payload_b64:
                continue

            payload = json.loads(base64.b64decode(payload_b64).decode('utf-8'))
            # NOTE: Takes only the FIRST subject (index [0]) from the in-toto statement.
            # SLSA provenance bundles for container images have exactly one subject
            # (the container digest). This is safe for our use case.
            subject = payload.get("subject", [{}])[0]
            subject_digest = subject.get("digest", {}).get("sha256", "")

            if subject_digest == target_digest:
                return bundle
        except Exception as e:
            print(f"Warning: Failed to fetch or parse {item['name']}: {e}")
            continue

    return None

def fetch_rekor_payload_hashes(digest):
    """
    Queries the Rekor transparency log and returns a set of valid payload hashes.

    Args:
        digest (str): The raw SHA256 hex digest (without 'sha256:' prefix).

    Returns:
        set[str]: A set of valid payload hashes confirmed by Rekor.
    """
    # Use shared helper for the HTTP fetch (avoids duplicating the Rekor query code).
    # All validation of entry contents is done inline below — every check from the
    # original implementation is preserved.
    try:
        rekor_entries = get_rekor_entries_for_digest(digest)
        if not rekor_entries:
            fail("Rekor returned no entries or invalid format. Transparency log cross-check failed.")

        print(f"  -> Rekor confirms {len(rekor_entries)} transparency log entry/entries for this digest.")
        rekor_payload_hashes = set()

        for idx, entry_obj in enumerate(rekor_entries):
            entry_id = f"entry[{idx}]"

            if not isinstance(entry_obj, dict):
                fail(f"Rekor {entry_id} content is not a valid JSON object.")

            body_b64 = entry_obj.get("body")
            if not body_b64 or not isinstance(body_b64, str):
                fail(f"Rekor {entry_id} is missing a valid 'body' string.")

            try:
                body_json = base64.b64decode(body_b64).decode('utf-8')
                body_data = json.loads(body_json)
            except Exception as e:
                fail(f"Failed to decode Rekor {entry_id} body: {e}")

            if not isinstance(body_data, dict):
                fail(f"Rekor {entry_id} decoded body is not a JSON object.")

            spec = body_data.get("spec")
            if not isinstance(spec, dict):
                fail(f"Rekor {entry_id} is missing a valid 'spec' object.")

            kind = body_data.get("kind")
            if not kind or not isinstance(kind, str):
                fail(f"Rekor {entry_id} is missing a valid 'kind' string.")

            if kind == "dsse":
                payload_hash_obj = spec.get("payloadHash")
                if not isinstance(payload_hash_obj, dict):
                    fail(f"Rekor {entry_id} (dsse) is missing a valid 'payloadHash' object.")
                phash = payload_hash_obj.get("value")
                if not phash or not isinstance(phash, str):
                    fail(f"Rekor {entry_id} (dsse) is missing a valid payloadHash 'value' string.")
            else:
                data_obj = spec.get("data")
                if not isinstance(data_obj, dict):
                    fail(f"Rekor {entry_id} (kind={kind}) is missing a valid 'data' object.")
                hash_obj = data_obj.get("hash")
                if not isinstance(hash_obj, dict):
                    fail(f"Rekor {entry_id} (kind={kind}) is missing a valid 'hash' object.")
                phash = hash_obj.get("value")
                if not phash or not isinstance(phash, str):
                    fail(f"Rekor {entry_id} (kind={kind}) is missing a valid hash 'value' string.")

            rekor_payload_hashes.add(phash)
    except SystemExit:
        raise
    except Exception as e:
        fail(f"Rekor cross-check failed or is unavailable: {e}")

    return rekor_payload_hashes

def decode_dsse_payload(attestation, idx):
    """
    Extracts and decodes the DSSE payload from a single attestation.

    Args:
        attestation (dict): A single attestation dictionary.
        idx (int): The index of the attestation (for error reporting).

    Returns:
        tuple[str, dict]: The raw payload JSON string, and the parsed payload dictionary.
    """
    if not isinstance(attestation, dict):
        fail(f"Attestation {idx+1} is not a valid JSON object.")

    b = attestation.get("bundle")
    if not b or not isinstance(b, dict):
        fail(f"Attestation {idx+1} is missing a valid 'bundle' object.")

    dsse_envelope = b.get("dsseEnvelope")
    if not dsse_envelope or not isinstance(dsse_envelope, dict):
        fail(f"Attestation {idx+1} is missing a valid 'dsseEnvelope' inside bundle.")

    payload_b64 = dsse_envelope.get("payload")
    if not payload_b64 or not isinstance(payload_b64, str):
        fail(f"Attestation {idx+1} is missing a valid 'payload' string in dsseEnvelope. Critical structural failure.")

    try:
        payload_json = base64.b64decode(payload_b64).decode('utf-8')
        payload = json.loads(payload_json)
    except Exception as e:
        fail(f"Failed to decode DSSE payload in attestation {idx+1}: {e}")

    if not isinstance(payload, dict):
        fail(f"DSSE payload in attestation {idx+1} is not a valid JSON object (got {type(payload).__name__}).")

    return payload_json, payload

def cross_check_payloads_against_rekor(attestations, rekor_payload_hashes):
    """
    Decodes each DSSE payload and verifies its hash exists in Rekor.

    Args:
        attestations (list[dict]): A list of attestation dictionaries.
        rekor_payload_hashes (set[str]): A set of valid payload hashes from Rekor.

    Returns:
        None (Fails process on validation error).
    """
    for idx, attestation in enumerate(attestations):
        payload_json, _ = decode_dsse_payload(attestation, idx)

        calculated_phash = hashlib.sha256(payload_json.encode('utf-8')).hexdigest()
        if calculated_phash not in rekor_payload_hashes:
            fail(f"Attestation {idx+1} payload hash ({calculated_phash}) NOT FOUND in Rekor transparency log. Possible tampering.")
        print(f"  -> Attestation {idx+1} payload strictly matched to Rekor transparency log.")

def extract_provenance_metadata(attestations, digest):
    """
    Validates subject consistency and extracts commits, workflows, custom metadata.

    Args:
        attestations (list[dict]): A list of attestation dictionaries.
        digest (str): The raw SHA256 hex digest (without 'sha256:' prefix).

    Returns:
        tuple: (subject_name, subject_digest, unique_commits_list, custom_metadata_list, workflows_list)
    """
    unique_commits = set()
    workflows = []
    custom_metadata = []

    subject_name = None
    subject_digest = None

    for idx, attestation in enumerate(attestations):
        _, payload = decode_dsse_payload(attestation, idx)

        # Validate subject in EVERY attestation and enforce consistency.
        subject_list = payload.get("subject")
        if not subject_list or not isinstance(subject_list, list) or len(subject_list) == 0:
            fail(f"Attestation {idx+1} is missing a valid 'subject' list. Critical metadata missing.")

        matching_subject = None
        for subj_idx, subj in enumerate(subject_list):
            if not isinstance(subj, dict):
                fail(f"Attestation {idx+1} subject entry {subj_idx} is not a valid JSON object (got {type(subj).__name__}).")
            subj_digest = subj.get("digest")
            if isinstance(subj_digest, dict) and subj_digest.get("sha256") == digest:
                matching_subject = subj
                break

        if not matching_subject:
            fail(f"Attestation {idx+1} subject list does not contain the requested digest sha256:{digest}.")

        this_name = matching_subject.get("name")
        if not this_name:
            fail(f"Attestation {idx+1} matching subject is missing a 'name'.")

        subj_digest_obj = matching_subject.get("digest")
        if not isinstance(subj_digest_obj, dict):
            fail(f"Attestation {idx+1} matching subject is missing a valid 'digest' object.")

        extracted_sha256 = subj_digest_obj.get("sha256")
        if not extracted_sha256 or not isinstance(extracted_sha256, str):
            fail(f"Attestation {idx+1} matching subject is missing a valid 'sha256' string in digest.")

        if subject_name is None:
            # First attestation — establish the reference subject.
            subject_name = this_name
            subject_digest = extracted_sha256
            print(f"  -> Subject: {subject_name} (sha256:{subject_digest})")
        else:
            # Subsequent attestations — enforce consistency.
            if this_name != subject_name:
                fail(f"Subject name mismatch: attestation 1 has '{subject_name}' but attestation {idx+1} has '{this_name}'.")

            if extracted_sha256 != subject_digest:
                fail(f"Subject digest mismatch: attestation 1 has '{subject_digest}' but attestation {idx+1} has '{extracted_sha256}'.")

        predicate_type = payload.get("predicateType")
        if not predicate_type or not isinstance(predicate_type, str):
            fail(f"Attestation {idx+1} is missing a valid 'predicateType' string. Critical metadata missing.")

        predicate = payload.get("predicate")
        if not predicate or not isinstance(predicate, dict):
            fail(f"Attestation {idx+1} is missing a valid 'predicate' object. Critical metadata missing.")

        if "slsa.dev/provenance" in predicate_type:
            build_def = predicate.get("buildDefinition")
            if not build_def or not isinstance(build_def, dict):
                fail(f"SLSA provenance attestation {idx+1} is missing a valid 'buildDefinition' object.")

            resolved_deps = build_def.get("resolvedDependencies")
            if not resolved_deps or not isinstance(resolved_deps, list):
                fail(f"SLSA provenance attestation {idx+1} has missing or invalid 'resolvedDependencies' list. Cannot find gitCommit.")

            commits_found = 0
            for dep_idx, dep in enumerate(resolved_deps):
                if not isinstance(dep, dict):
                    fail(f"SLSA provenance attestation {idx+1} resolvedDependencies entry {dep_idx} is not a valid JSON object (got {type(dep).__name__}).")
                dep_digest = dep.get("digest")
                if isinstance(dep_digest, dict) and "gitCommit" in dep_digest:
                    commit_val = dep_digest["gitCommit"]
                    if not isinstance(commit_val, str) or not commit_val.strip():
                        fail(f"SLSA provenance attestation {idx+1} has invalid or empty 'gitCommit'.")
                    unique_commits.add(commit_val.strip())
                    commits_found += 1

            if commits_found == 0:
                fail(f"SLSA provenance attestation {idx+1} did not contain any 'gitCommit' in its dependencies.")

            run_details = predicate.get("runDetails")
            if not run_details or not isinstance(run_details, dict):
                fail(f"SLSA provenance attestation {idx+1} is missing a valid 'runDetails' object.")

            metadata = run_details.get("metadata")
            if not metadata or not isinstance(metadata, dict):
                fail(f"SLSA provenance attestation {idx+1} is missing a valid 'metadata' object in runDetails.")

            workflow_id = metadata.get("invocationId")
            if not workflow_id or not isinstance(workflow_id, str):
                fail(f"SLSA provenance attestation {idx+1} is missing a valid 'invocationId' string (Workflow Run ID).")
            workflows.append(workflow_id)
        else:
            custom_metadata.append((predicate_type, predicate))

    if not unique_commits:
        fail("Parsed attestations did not contain SLSA provenance (missing gitCommit).")
    if not custom_metadata:
        print("  -> Note: Parsed attestations did not contain any custom metadata.")

    return subject_name, subject_digest, list(unique_commits), custom_metadata, workflows

def _compare_single_bundle(gh_bundle, oak_bundle):
    """Compare two SLSA bundles field-by-field. Raises VerificationError on mismatch.

    Checks ALL of the following fields — none are skipped:
      1. dsseEnvelope.payload (byte-identical base64 content)
      2. dsseEnvelope.payloadType
      3. dsseEnvelope.signatures (count AND each sig's bytes)
      4. verificationMaterial (normalized: empty strings/lists/dicts stripped)
      5. mediaType
    If ANY check fails, raises VerificationError immediately with a specific message.
    If ALL pass, returns normally (implicit success).
    """
    def _strip_empty_defaults(obj):
        """Recursively remove keys whose values are or become empty strings, lists, or dicts."""
        if isinstance(obj, dict):
            stripped = {}
            for k, v in obj.items():
                sv = _strip_empty_defaults(v)
                if sv != "" and sv != [] and sv != {}:
                    stripped[k] = sv
            return stripped
        if isinstance(obj, list):
            return [_strip_empty_defaults(item) for item in obj]
        return obj

    gh_dsse = gh_bundle.get("dsseEnvelope", {})
    oak_dsse = oak_bundle.get("dsseEnvelope", {})

    # 1. Payloads must be byte-identical (the actual in-toto statement)
    if gh_dsse.get("payload") != oak_dsse.get("payload"):
        raise VerificationError("GitHub and Oak SLSA attestation payloads differ. Possible tampering.")

    # 2. Payload type must match
    if gh_dsse.get("payloadType") != oak_dsse.get("payloadType"):
        raise VerificationError("GitHub and Oak SLSA attestation payloadType differs.")

    # 3. Signature bytes must match (ignore empty optional fields like keyid)
    gh_sigs = gh_dsse.get("signatures", [])
    oak_sigs = oak_dsse.get("signatures", [])
    if len(gh_sigs) != len(oak_sigs):
        raise VerificationError(f"GitHub has {len(gh_sigs)} signature(s) but Oak has {len(oak_sigs)}.")
    for sig_idx in range(len(gh_sigs)):
        if gh_sigs[sig_idx].get("sig") != oak_sigs[sig_idx].get("sig"):
            raise VerificationError(f"GitHub and Oak SLSA signature[{sig_idx}] bytes differ. Possible tampering.")

    # 4. Verification material must match (normalized to strip empty defaults)
    gh_vm = _strip_empty_defaults(gh_bundle.get("verificationMaterial", {}))
    oak_vm = _strip_empty_defaults(oak_bundle.get("verificationMaterial", {}))
    if gh_vm != oak_vm:
        raise VerificationError("GitHub and Oak SLSA verification material differs. Possible tampering.")

    # 5. Media type
    if gh_bundle.get("mediaType") != oak_bundle.get("mediaType"):
        raise VerificationError("GitHub and Oak SLSA bundle mediaType differs.")

def _compare_attestation_sources(github_attestations, oak_attestations):
    """
    Verify that the Oak bucket SLSA provenance bundle is a complete, identical
    copy of the corresponding GitHub SLSA provenance bundle. This validates that
    if GitHub becomes unavailable, the Oak bundle is a 100% reliable fallback.

    The Oak bucket only contains the SLSA provenance attestation (custom metadata
    is GitHub-only and supplementary — not required for provenance verification).
    """
    # NOTE: We only compare ONE Oak bundle (index [0]). The caller (fetch_and_verify)
    # stops at the first Oak bundle it finds (via break), so oak_attestations always
    # has exactly one entry. This single Oak bundle is compared against ALL GitHub
    # SLSA provenance bundles to find its match — it must match at least one.
    oak_bundle = oak_attestations[0].get("bundle")
    if not oak_bundle or not isinstance(oak_bundle, dict):
        fail("Oak attestation is missing a valid 'bundle' object for comparison.")

    mismatch_reasons = []
    # Iterate through ALL GitHub attestations, filtering for SLSA provenance type.
    # Non-SLSA attestations (e.g. custom metadata) are skipped via continue.
    # Each SLSA provenance bundle is compared against the single Oak bundle.
    # On the FIRST match, we return success. If NO match is found after checking
    # all SLSA bundles, we hard-fail with the accumulated mismatch reasons.
    for idx, att in enumerate(github_attestations):
        gh_bundle = att.get("bundle")
        if not gh_bundle or not isinstance(gh_bundle, dict):
            continue

        dsse = gh_bundle.get("dsseEnvelope", {})
        payload_b64 = dsse.get("payload", "")
        if not payload_b64:
            continue

        try:
            payload = json.loads(base64.b64decode(payload_b64).decode('utf-8'))
            predicate_type = payload.get("predicateType", "")
        except Exception:
            continue

        if "slsa.dev/provenance" not in predicate_type:
            continue

        # Found a SLSA provenance from GitHub. Compare content field-by-field.
        try:
            _compare_single_bundle(gh_bundle, oak_bundle)
            gh_sigs = gh_bundle.get("dsseEnvelope", {}).get("signatures", [])
            print(f"  -> DSSE payload: identical.")
            print(f"  -> DSSE payloadType: identical.")
            print(f"  -> DSSE signatures ({len(gh_sigs)}): identical.")
            print(f"  -> Verification material: identical.")
            print(f"  -> SLSA provenance bundle: GitHub and Oak content fully verified as identical.")
            return
        except VerificationError as e:
            mismatch_reasons.append(str(e))

    if not mismatch_reasons:
        fail("GitHub attestations do not contain an SLSA provenance bundle for comparison with Oak.")
    else:
        fail(f"CRITICAL: Oak bundle did not match any of the {len(mismatch_reasons)} GitHub SLSA provenance bundles. Mismatches: {mismatch_reasons}")

def fetch_and_verify(digest):
    """
    Main entrypoint: orchestrates fetching, signature verification, Rekor
    cross-checking, and metadata extraction.

    For client containers, both GitHub and Oak sources are queried and compared.
    If one source is unavailable, the other is used with a warning.
    For server containers, only GitHub is queried (Oak fallback not yet implemented).

    Args:
        digest (str): The target digest (can optionally include 'sha256:' prefix).

    Returns:
        tuple: Outputs from extract_provenance_metadata.
    """
    # Auto-adjust the digest for all downstream functions.
    digest_clean = digest[7:] if digest.startswith("sha256:") else digest

    # Validate digest is a well-formed hex string.
    if not re.fullmatch(r'[0-9a-fA-F]{64}', digest_clean):
        fail(f"Invalid digest format: '{digest_clean}'. Expected a 64-character lowercase hex SHA256 string.")

    print(f"======================================================================")
    print(f" EXTERNAL VERIFICATION START: sha256:{digest_clean}")
    print(f"======================================================================\n")

    # ---- STEP 1: Query Rekor to determine container type ----
    print(f"[*] STEP 1/4: Querying Rekor Transparency Log...")
    try:
        rekor_entries = get_rekor_entries_for_digest(digest_clean)
    except Exception as e:
        fail(f"Failed to query Rekor transparency log: {e}")

    if not rekor_entries:
        fail("No Rekor transparency log entries found for this digest.")

    valid_cert_metadata = []
    for entry in rekor_entries:
        try:
            sha, uri = extract_cert_metadata(entry)
            valid_cert_metadata.append((sha, uri))
        except Exception:
            continue

    if not valid_cert_metadata:
        fail("Could not extract GITHUB_SHA and WORKFLOW_URI from any Rekor certificate extension.")

    # Determine container type. All Rekor entries for a given digest must agree
    # on whether it's a client or server build. If they disagree, something is
    # seriously wrong (CI misconfiguration, or compromised entries).
    first_sha, first_uri = valid_cert_metadata[0]
    is_client_set = {"gcp_server_build.yaml" not in uri for _, uri in valid_cert_metadata}
    if len(is_client_set) > 1:
        fail("Ambiguous container type: Rekor entries disagree on client vs server build.")
    is_client = is_client_set.pop()
    container_type = "client" if is_client else "server"
    print(f"  -> Rekor indicates {len(valid_cert_metadata)} valid certificate(s). First GITHUB_SHA: {first_sha}")
    print(f"  -> Rekor indicates WORKFLOW_URI: {first_uri}")
    print(f"  -> Identified as {container_type} container.\n")

    # ---- STEP 2: Fetch attestation bundles from available sources ----
    print(f"[*] STEP 2/4: Fetching Attestation Bundles...")

    # Try GitHub API
    github_attestations = None
    try:
        github_attestations = fetch_attestations_from_github(digest_clean)
        print(f"  -> Found {len(github_attestations)} attestation(s) via GitHub API.")
    except Exception as e:
        print(f"  -> WARNING: GitHub API unavailable: {e}")

    # For client containers, also try Oak bucket.
    # We iterate through unique SHAs from Rekor and stop at the FIRST one that
    # has a matching bundle in the Oak bucket (break on first success). This is
    # intentional: we only need ONE Oak bundle to cross-check against GitHub.
    # The Oak bundle is later verified field-by-field in _compare_attestation_sources,
    # and independently verified by Sigstore (Step 3) and Rekor (Step 3.5).
    oak_attestations = None
    if is_client:
        seen_shas = set()
        for sha, uri in valid_cert_metadata:
            if sha in seen_shas:
                continue
            seen_shas.add(sha)
            try:
                bundle = fetch_oak_bundle_by_sha(sha, digest_clean)
                if bundle:
                    oak_attestations = [{"bundle": bundle}]
                    print(f"  -> Found 1 attestation bundle in Oak bucket (via SHA {sha}).")
                    break
            except Exception as e:
                print(f"  -> WARNING: Oak bucket fetch failed for SHA {sha}: {e}")
        if not oak_attestations:
            print(f"  -> WARNING: No matching bundle found in Oak bucket for any of the {len(seen_shas)} unique SHA(s).")

    # Decision: which source(s) to use
    is_offline = False
    if github_attestations and oak_attestations:
        # Both available — compare full SLSA bundle content
        print(f"  -> Both sources available. Cross-checking SLSA bundle consistency...")
        _compare_attestation_sources(github_attestations, oak_attestations)
        attestations = github_attestations  # Use GitHub as primary (has custom metadata too)
        print()
    elif github_attestations:
        if is_client:
            print(f"  -> WARNING: Only GitHub source available for client container (Oak bucket unavailable).")
        attestations = github_attestations
        print()
    elif oak_attestations:
        print(f"  -> WARNING: Only Oak bucket source available (GitHub API unavailable).")
        print(f"  -> Note: Custom metadata attestation is GitHub-only and will not be available.")
        attestations = oak_attestations
        is_offline = True
        print()
    else:
        if is_client:
            fail("Neither GitHub API nor Oak bucket returned valid attestations. Cannot proceed.")
        else:
            fail("GitHub API unavailable and no alternative source exists for server containers yet.")

    # ---- STEP 3: Cryptographic verification ----
    print(f"[*] STEP 3/4: Cryptographic Verification via Sigstore...")
    verify_attestation_signatures(attestations, digest_clean, offline=is_offline)

    # Rekor cross-check: always run this regardless of source. Rekor is independent
    # of GitHub and Step 1 already proved it's reachable. The Oak fallback path
    # should get MORE verification, not less.
    print(f"\n[*] STEP 3.5/4: Independent Rekor Transparency Log Cross-Check...")
    rekor_hashes = fetch_rekor_payload_hashes(digest_clean)
    cross_check_payloads_against_rekor(attestations, rekor_hashes)

    # ---- STEP 4: Extract provenance metadata ----
    print(f"\n[*] STEP 4/4: Extracting Provenance & Metadata...")
    return extract_provenance_metadata(attestations, digest_clean)
