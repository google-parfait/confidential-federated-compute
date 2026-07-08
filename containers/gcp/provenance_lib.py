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
    """
    url = f"https://api.github.com/repos/google-parfait/confidential-federated-compute/attestations/sha256:{digest}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        fail(f"HTTP Error {e.code}: {e.reason}. Could not find attestations. Was this image built by CI?")
    except Exception as e:
        fail(f"Error fetching attestations: {e}")

    if not isinstance(data, dict):
        fail(f"GitHub API returned unexpected data type: {type(data).__name__}. Expected a JSON object.")

    attestations = data.get("attestations")
    if not attestations or not isinstance(attestations, list):
        fail("API returned HTTP 200 but no valid 'attestations' list was found in the response.")

    return attestations

def verify_attestation_signatures(attestations, digest, repository="google-parfait/confidential-federated-compute"):
    """
    Writes bundles to disk and uses sigstore CLI to cryptographically verify them.

    Args:
        attestations (list[dict]): A list of attestation dictionaries.
        digest (str): The raw SHA256 hex digest (without 'sha256:' prefix).
        repository (str, optional): The expected GitHub repository.

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

def fetch_rekor_payload_hashes(digest):
    """
    Queries the Rekor transparency log and returns a set of valid payload hashes.

    Args:
        digest (str): The raw SHA256 hex digest (without 'sha256:' prefix).

    Returns:
        set[str]: A set of valid payload hashes confirmed by Rekor.
    """
    rekor_url = "https://rekor.sigstore.dev/api/v1/index/retrieve"
    rekor_req = urllib.request.Request(
        rekor_url,
        data=json.dumps({"hash": f"sha256:{digest}"}).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST"
    )

    rekor_payload_hashes = set()
    try:
        with urllib.request.urlopen(rekor_req, timeout=10) as resp:
            rekor_entries = json.loads(resp.read().decode())
        if not rekor_entries or not isinstance(rekor_entries, list):
            fail("Rekor returned no entries or invalid format. Transparency log cross-check failed.")

        print(f"  -> Rekor confirms {len(rekor_entries)} transparency log entry/entries for this digest.")
        for entry_id in rekor_entries:
            print(f"     Fetching Rekor entry: https://rekor.sigstore.dev/api/v1/log/entries/{entry_id}")
            req2 = urllib.request.Request(
                f"https://rekor.sigstore.dev/api/v1/log/entries/{entry_id}",
                headers={"Accept": "application/json"}
            )
            with urllib.request.urlopen(req2, timeout=10) as resp2:
                entry_data = json.loads(resp2.read().decode())

                if not isinstance(entry_data, dict):
                    fail(f"Rekor entry {entry_id} response is not a valid JSON object.")
                if entry_id not in entry_data:
                    fail(f"Rekor entry {entry_id} response is missing the entry ID key.")

                entry_obj = entry_data[entry_id]
                if not isinstance(entry_obj, dict):
                    fail(f"Rekor entry {entry_id} content is not a valid JSON object.")

                body_b64 = entry_obj.get("body")
                if not body_b64 or not isinstance(body_b64, str):
                    fail(f"Rekor entry {entry_id} is missing a valid 'body' string.")

                try:
                    body_json = base64.b64decode(body_b64).decode('utf-8')
                    body_data = json.loads(body_json)
                except Exception as e:
                    fail(f"Failed to decode Rekor entry {entry_id} body: {e}")

                if not isinstance(body_data, dict):
                    fail(f"Rekor entry {entry_id} decoded body is not a JSON object.")

                spec = body_data.get("spec")
                if not isinstance(spec, dict):
                    fail(f"Rekor entry {entry_id} is missing a valid 'spec' object.")

                kind = body_data.get("kind")
                if not kind or not isinstance(kind, str):
                    fail(f"Rekor entry {entry_id} is missing a valid 'kind' string.")

                if kind == "dsse":
                    payload_hash_obj = spec.get("payloadHash")
                    if not isinstance(payload_hash_obj, dict):
                        fail(f"Rekor entry {entry_id} (dsse) is missing a valid 'payloadHash' object.")
                    phash = payload_hash_obj.get("value")
                    if not phash or not isinstance(phash, str):
                        fail(f"Rekor entry {entry_id} (dsse) is missing a valid payloadHash 'value' string.")
                else:
                    data_obj = spec.get("data")
                    if not isinstance(data_obj, dict):
                        fail(f"Rekor entry {entry_id} (kind={kind}) is missing a valid 'data' object.")
                    hash_obj = data_obj.get("hash")
                    if not isinstance(hash_obj, dict):
                        fail(f"Rekor entry {entry_id} (kind={kind}) is missing a valid 'hash' object.")
                    phash = hash_obj.get("value")
                    if not phash or not isinstance(phash, str):
                        fail(f"Rekor entry {entry_id} (kind={kind}) is missing a valid hash 'value' string.")

                rekor_payload_hashes.add(phash)
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

def fetch_and_verify(digest):
    """
    Main entrypoint: orchestrates fetching, signature verification, Rekor
    cross-checking, and metadata extraction.

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

    print(f"[*] STEP 1/3: Querying GitHub Attestation API...")
    attestations = fetch_attestations_from_github(digest_clean)
    print(f"  -> Found {len(attestations)} attestation(s).\n")

    print(f"[*] STEP 2/3: Cryptographic Verification via Sigstore...")
    verify_attestation_signatures(attestations, digest_clean)

    print(f"\n[*] STEP 2.5/3: Independent Rekor Transparency Log Cross-Check...")
    rekor_hashes = fetch_rekor_payload_hashes(digest_clean)
    cross_check_payloads_against_rekor(attestations, rekor_hashes)

    print(f"\n[*] STEP 3/3: Extracting Provenance & Metadata...")
    return extract_provenance_metadata(attestations, digest_clean)
