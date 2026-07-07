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

def fetch_and_verify(digest):
    """
    Fetches attestations, verifies them via sigstore, and extracts metadata.
    Returns:
        subject_name (str)
        subject_digest (str)
        commits (list of str)
        custom_metadata (list of tuples (predicate_type, predicate))
        workflows (list of str)
    """
    if digest.startswith("sha256:"):
        digest = digest[7:]

    print(f"======================================================================")
    print(f" EXTERNAL VERIFICATION START: sha256:{digest}")
    print(f"======================================================================\n")

    print(f"[*] STEP 1/3: Querying GitHub Attestation API...")
    url = f"https://api.github.com/repos/google-parfait/confidential-federated-compute/attestations/sha256:{digest}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        fail(f"HTTP Error {e.code}: {e.reason}. Could not find attestations. Was this image built by CI?")
    except Exception as e:
        fail(f"Error fetching attestations: {e}")

    attestations = data.get("attestations", [])
    if not attestations:
        fail("API returned HTTP 200 but no attestations were found in the response.")

    print(f"  -> Found {len(attestations)} attestation(s).\n")

    print(f"[*] STEP 2/3: Cryptographic Verification via Sigstore...")

    for idx, attestation in enumerate(attestations):
        bundle = attestation.get("bundle", {})

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
            json.dump(bundle, tf)
            tf_name = tf.name

        try:
            cmd = [
                sys.executable, "-m", "sigstore", "verify", "github",
                "--bundle", tf_name,
                "--repository", "google-parfait/confidential-federated-compute",
                f"sha256:{digest}"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                fail(f"Cryptographic signature check FAILED for attestation {idx+1}.\n\nSigstore output:\n{result.stderr}")

            print(f"  -> Attestation {idx+1}: Cryptographic signatures (Fulcio, Rekor, DSSE, OIDC Identity) VERIFIED.")
        except FileNotFoundError:
            fail("sigstore module not found. Did you run 'pip install -r requirements.txt'?")
        except Exception as e:
            fail(f"Error running sigstore: {e}")
        finally:
            os.remove(tf_name)

    print(f"\n[*] STEP 3/3: Extracting Provenance & Metadata...")
    unique_commits = set()
    workflows = []
    custom_metadata = []
    subject_printed = False

    subject_name = "UNKNOWN"
    subject_digest = "UNKNOWN"

    for idx, attestation in enumerate(attestations):
        b = attestation.get("bundle", {})
        dsse_envelope = b.get("dsseEnvelope", {})
        payload_b64 = dsse_envelope.get("payload", "")

        if not payload_b64:
            continue

        try:
            payload_json = base64.b64decode(payload_b64).decode('utf-8')
            payload = json.loads(payload_json)
        except Exception as e:
            fail(f"Failed to decode DSSE payload in attestation {idx+1}: {e}")

        if not subject_printed:
            subject = payload.get("subject", [{}])[0]
            subject_name = subject.get("name", "UNKNOWN")
            subject_digest = subject.get("digest", {}).get("sha256", "UNKNOWN")
            print(f"  -> Subject: {subject_name} (sha256:{subject_digest})")
            subject_printed = True

        predicate_type = payload.get("predicateType", "")
        predicate = payload.get("predicate", {})

        if "slsa.dev/provenance" in predicate_type:
            build_def = predicate.get("buildDefinition", {})
            resolved_deps = build_def.get("resolvedDependencies", [])

            commit = None
            for dep in resolved_deps:
                if "gitCommit" in dep.get("digest", {}):
                    commit = dep["digest"]["gitCommit"]
                    break

            if commit:
                unique_commits.add(commit)

            workflow_id = predicate.get("runDetails", {}).get("metadata", {}).get("invocationId")
            if workflow_id:
                workflows.append(workflow_id)
        else:
            custom_metadata.append((predicate_type, predicate))

    if not unique_commits and not custom_metadata:
        fail("No valid SLSA provenance (gitCommit) or custom metadata found in the verified attestations.")

    return subject_name, subject_digest, list(unique_commits), custom_metadata, workflows
