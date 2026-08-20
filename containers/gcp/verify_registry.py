#!/usr/bin/env python3
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

"""Offline verification tool for server_image_registry.json.

Cryptographically verifies that all container image digests in the registry
have valid SLSA provenance bundles, signed Fulcio certificates, and Rekor
inclusion proofs using the embedded bundles in the registry file.

This tool operates 100% offline without making any network calls.

Usage:
    python3 verify_registry.py
    python3 verify_registry.py --registry=server_image_registry.json
"""

import argparse
import json
import os
import sys
import tempfile
import provenance_lib


def verify_entry_offline(entry, idx, total, repository="google-parfait/confidential-federated-compute", verbose=False):
    """Verifies a single registry entry offline using its embedded provenance bundle."""
    required_fields = ["model", "attestation", "digest", "tag", "created", "provenance"]
    for field in required_fields:
        if field not in entry:
            provenance_lib.fail(f"Registry entry {idx+1}/{total} is missing required field '{field}'.")

    model = entry["model"]
    attestation_type = entry["attestation"]
    digest = entry["digest"]
    tag = entry["tag"]
    created = entry["created"]
    provenance = entry["provenance"]

    if not isinstance(provenance, list) or len(provenance) == 0:
        provenance_lib.fail(f"Registry entry {idx+1}/{total} ({model} - {digest}) has empty or invalid 'provenance' list.")

    digest_clean = digest[7:] if digest.startswith("sha256:") else digest

    print(f"[*] [{idx+1}/{total}] Verifying {model} ({attestation_type}) | sha256:{digest_clean[:16]}... (created: {created})")

    # 1. Cryptographically verify the embedded Sigstore bundle offline
    provenance_lib.verify_attestation_signatures(
        provenance,
        digest_clean,
        repository=repository,
        offline=True
    )

    # 2. Extract and validate metadata (subject, gitCommit, workflow) from the bundle
    (subject_name, subject_digest, unique_commits, custom_metadata, workflows) = (
        provenance_lib.extract_provenance_metadata(provenance, digest_clean)
    )

    if subject_digest != digest_clean:
        provenance_lib.fail(
            f"Subject digest in attestation ({subject_digest}) does not match registry digest ({digest_clean})."
        )

    print(f"    -> Provenance Verified: commit={unique_commits[0] if unique_commits else 'N/A'}, workflow={workflows[0] if workflows else 'N/A'}")
    if verbose:
        print(f"       Subject: {subject_name}")
        print(f"       Unique Commits: {unique_commits}")
        print(f"       Workflows: {workflows}")


def test_policy_generation(registry_path, verbose=False):
    """Verifies that generate_policy.py can parse the registry without errors."""
    print("\n[*] Validating policy generation for all supported flavors...")
    import subprocess
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gen_policy_script = os.path.join(script_dir, "generate_policy.py")

    test_configs = [
        ("ITA", "gemma4_e4b", "ita_alts"),
        ("GCA", "gemma4_e4b", "gca_alts"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        for verifier_type, model, attestation in test_configs:
            out_file = os.path.join(tmpdir, f"policy_{verifier_type.lower()}.textproto")
            cmd = [
                sys.executable, gen_policy_script,
                f"--registry={registry_path}",
                f"--output={out_file}",
                f"--verifier_type={verifier_type}",
                f"--model={model}",
                f"--attestation={attestation}",
                "--max_sw_tcb_age_days=540",
                "--max_hw_tcb_age_days=540",
                "--min_swversion=260500",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                provenance_lib.fail(f"generate_policy.py failed for {verifier_type}/{model}/{attestation}:\n{res.stderr}")
            if verbose:
                print(f"    -> Policy ({verifier_type}/{model}/{attestation}): OK")
    print("    -> Policy generation: OK")


def main():
    parser = argparse.ArgumentParser(
        description="Verify SLSA provenance and cryptographic signatures in server_image_registry.json offline."
    )
    parser.add_argument(
        "--registry",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "server_image_registry.json"),
        help="Path to server_image_registry.json."
    )
    parser.add_argument(
        "--repository",
        default="google-parfait/confidential-federated-compute",
        help="Expected GitHub repository in the Fulcio certificate."
    )
    parser.add_argument(
        "--skip_policy_check",
        action="store_true",
        help="Skip policy generation validation."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output."
    )
    args = parser.parse_args()

    if not os.path.exists(args.registry):
        provenance_lib.fail(f"Registry file not found at: {args.registry}")

    try:
        with open(args.registry, "r") as f:
            registry_data = json.load(f)
    except Exception as e:
        provenance_lib.fail(f"Failed to parse registry JSON file: {e}")

    images = registry_data.get("images", [])
    if not images or not isinstance(images, list):
        provenance_lib.fail("Registry contains no images or 'images' is not a list.")

    print("=" * 70)
    print(" OFFLINE SERVER IMAGE REGISTRY VERIFICATION")
    print(f" Registry: {args.registry}")
    print(f" Entries:  {len(images)}")
    print("=" * 70 + "\n")

    for idx, entry in enumerate(images):
        verify_entry_offline(entry, idx, len(images), repository=args.repository, verbose=args.verbose)

    if not args.skip_policy_check:
        test_policy_generation(args.registry, verbose=args.verbose)

    print("\n" + "=" * 70)
    print(f" [V] ALL {len(images)} REGISTRY ENTRIES CRYPTOGRAPHICALLY VERIFIED (OFFLINE)")
    print("=" * 70)


if __name__ == "__main__":
    main()
