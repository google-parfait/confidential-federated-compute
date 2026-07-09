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

"""Updates server_image_registry.json with a cryptographically verified digest.

This script fetches the SLSA provenance and custom metadata attestations for
a given digest, cryptographically verifies them using Sigstore, extracts the
build parameters (model, alts, attestation type), and appends the entry
to server_image_registry.json.

Setup:
    python3 -m venv venv
    source venv/bin/activate
    pip install sigstore==4.4.0

Usage:
    python3 update_server_registry.py <sha256_digest>
    python3 update_server_registry.py <sha256_digest> --overwrite
"""

import argparse
import provenance_lib
import json
import os
import datetime

REGISTRY_PATH = "server_image_registry.json"

def update_registry(digest, overwrite=False):
    (subject_name, subject_digest, commits, custom_metadata, workflows), raw_attestations = provenance_lib.fetch_and_verify(digest)

    tag = subject_name
    extracted_custom_metadata = None
    for ptype, pdata in custom_metadata:
        if ptype == "https://batched-inference.google.com/server-metadata/v1":
            extracted_custom_metadata = pdata
            print(f"  -> Extracted Custom Metadata: {json.dumps(extracted_custom_metadata)}")
            break

    if not tag or tag == "UNKNOWN":
        provenance_lib.fail("Could not find SLSA provenance with a subject name (tag).")
    if not extracted_custom_metadata:
        provenance_lib.fail("Could not find custom build metadata attestation. This is required to determine the model and configuration.")

    model = extracted_custom_metadata.get("model")
    alts = extracted_custom_metadata.get("alts")
    base_attestation = extracted_custom_metadata.get("attestation")

    if not model or base_attestation is None:
        provenance_lib.fail(f"Custom metadata is missing required fields (model, attestation). Got: {extracted_custom_metadata}")

    if str(alts).lower() == "true":
        suffix = f"{base_attestation}_alts"
    else:
        suffix = f"{base_attestation}"

    created = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if digest.startswith("sha256:"):
        digest_val = digest
    else:
        digest_val = f"sha256:{digest}"

    entry = {
        "model": model,
        "attestation": suffix,
        "digest": digest_val,
        "tag": tag,
        "created": created
    }

    # Extract the first verified SLSA provenance bundle for offline verification.
    for idx, att in enumerate(raw_attestations):
        _, payload = provenance_lib.decode_dsse_payload(att, idx)
        predicate_type = payload.get("predicateType", "")
        if "slsa.dev/provenance" in predicate_type:
            entry["provenance"] = [
                {"predicateType": predicate_type, "bundle": att["bundle"]}
            ]
            break

    if "provenance" not in entry:
        provenance_lib.fail("No SLSA provenance bundle found in verified attestations. Cannot store provenance backup.")

    registry_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), REGISTRY_PATH)

    try:
        with open(registry_file, 'r') as f:
            registry = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        registry = {"images": []}

    found_existing = False
    for img in registry.get("images", []):
        if img.get("digest") == digest_val:
            found_existing = True
            mismatches = []
            if img.get("model") != entry["model"]: mismatches.append(f"model (registry={img.get('model')}, new={entry['model']})")
            if img.get("attestation") != entry["attestation"]: mismatches.append(f"attestation (registry={img.get('attestation')}, new={entry['attestation']})")
            if img.get("tag") != entry["tag"]: mismatches.append(f"tag (registry={img.get('tag')}, new={entry['tag']})")

            if mismatches:
                provenance_lib.fail(f"Registry entry for {digest_val} has mismatching metadata: {', '.join(mismatches)}")

            if "provenance" not in img or not img["provenance"]:
                print(f"\n[*] Digest {digest_val} exists but is missing provenance. Backfilling.")
                img["provenance"] = entry["provenance"]
            else:
                if overwrite:
                    print(f"\n[*] Digest {digest_val} already has provenance. Overwriting due to --overwrite.")
                    img["provenance"] = entry["provenance"]
                else:
                    print(f"\n[*] WARNING: Digest {digest_val} already has provenance. Skipping. Use --overwrite to replace.")
                    return

            entry_to_print = img
            break

    if not found_existing:
        registry.setdefault("images", []).append(entry)
        entry_to_print = entry

    with open(registry_file, 'w') as f:
        json.dump(registry, f, indent=2)
        f.write('\n')

    print(f"\n======================================================================")
    print(f" [V] REGISTRY UPDATE SUCCESS")
    print(f"======================================================================")
    print(json.dumps(entry_to_print, indent=2))
    print(f"\n[*] Updated {os.path.normpath(REGISTRY_PATH)}")
    print(f"[*] Please run: git add {os.path.normpath(REGISTRY_PATH)} && git commit -m \"Update server registry\"")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update server_image_registry.json with a verified digest.")
    parser.add_argument("digest", type=str, help="The hex SHA256 digest of the server container.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing provenance if present.")
    args = parser.parse_args()
    update_registry(args.digest, args.overwrite)
