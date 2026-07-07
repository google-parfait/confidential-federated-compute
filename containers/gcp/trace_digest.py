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

"""Traces a container digest to its source commit using GitHub Attestations.

This script queries the GitHub Attestations API for a given container digest,
decodes the SLSA provenance from the returned DSSE envelope, and extracts the
source commit and GitHub Actions run ID that produced it.

Usage:
    python3 trace_digest.py <sha256_digest>
"""

import argparse
import urllib.request
import urllib.error
import json
import base64
import sys

def trace_digest(digest):
    print(f"[*] Tracing breadcrumbs for digest: {digest}")

    # Check for prefix
    if digest.startswith("sha256:"):
        digest = digest[7:]

    print(f"[*] 1. Querying GitHub Attestation API for digest: sha256:{digest}")

    url = f"https://api.github.com/repos/google-parfait/confidential-federated-compute/attestations/sha256:{digest}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        print(f"[!] HTTP Error {e.code}: {e.reason}")
        print(f"[!] Could not find attestations for digest. Are you sure it was built by CI?")
        sys.exit(1)
    except Exception as e:
        print(f"[!] Error fetching attestations: {e}")
        sys.exit(1)

    attestations = data.get("attestations", [])
    if not attestations:
        print(f"[!] API returned success but no attestations found.")
        sys.exit(1)

    print(f"[*]    -> Found {len(attestations)} attestation(s) in GitHub.")

    unique_commits = set()
    workflows = []

    for idx, attestation in enumerate(attestations):
        bundle = attestation.get("bundle", {})
        dsse_envelope = bundle.get("dsseEnvelope", {})
        payload_b64 = dsse_envelope.get("payload", "")

        if not payload_b64:
            continue

        payload_json = base64.b64decode(payload_b64).decode('utf-8')
        payload = json.loads(payload_json)

        if idx == 0:
            # Print subject info just once based on the first attestation
            subject = payload.get("subject", [{}])[0]
            subject_name = subject.get("name", "UNKNOWN")
            subject_digest = subject.get("digest", {}).get("sha256", "UNKNOWN")
            print(f"[*] 2. Decoding DSSE envelope payload (SLSA provenance)...")
            print(f"[*]    -> Subject: {subject_name} (sha256:{subject_digest})")
            if subject_digest != digest:
                print(f"[!] Warning: Provenance subject digest ({subject_digest}) does not match requested digest ({digest}).")
            else:
                print(f"[*]    -> Cryptographic digest match verified internally.")
            print(f"[*] 3. Extracted Build Details from {len(attestations)} attestation(s):")

        predicate = payload.get("predicate", {})
        build_def = predicate.get("buildDefinition", {})
        resolved_deps = build_def.get("resolvedDependencies", [])

        commit = "UNKNOWN"
        for dep in resolved_deps:
            if "gitCommit" in dep.get("digest", {}):
                commit = dep["digest"]["gitCommit"]
                break

        if commit != "UNKNOWN":
            unique_commits.add(commit)

        workflow_id = predicate.get("runDetails", {}).get("metadata", {}).get("invocationId", "UNKNOWN")
        if workflow_id != "UNKNOWN":
            workflows.append(workflow_id)

    if not unique_commits:
        print(f"[!] Could not locate gitCommit in any resolvedDependencies.")
        sys.exit(1)

    for commit in unique_commits:
        print(f"[*]    -> Source Commit: {commit}")

    print(f"[*]    -> Found {len(workflows)} GitHub Actions Run(s) (e.g., {workflows[0] if workflows else 'UNKNOWN'})")

    # 4. Print final URL
    print(f"\n" + "="*70)
    print(f"TRACE COMPLETE. INSPECT SOURCE AT:")
    for commit in unique_commits:
        print(f"https://github.com/google-parfait/confidential-federated-compute/tree/{commit}/containers/gcp")
    print(f"="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trace a container digest to its source commit via GitHub Attestations.",
        epilog='''example:
  python3 trace_digest.py 2dce970207711ffeb036de533243295752d04862687210aabe77c0decdc57d56''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("digest", type=str, help="The hex SHA256 digest of the container (client or server).")

    args = parser.parse_args()
    trace_digest(args.digest)
