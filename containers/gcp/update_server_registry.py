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

Unless --no-prune is given, every run also prunes entries whose `created`
timestamp is older than --max_age_days (default: 60). This is the only place
where server images expire: generate_policy.py bakes every registry entry into
the client policy, so that the client build depends only on the committed
sources. Run the script without a digest to prune only.

Usage:
    bazelisk run //:update_server_registry -- <sha256_digest>
    bazelisk run //:update_server_registry -- <sha256_digest> --overwrite
    bazelisk run //:update_server_registry -- <sha256_digest> --no-prune
    bazelisk run //:update_server_registry -- --max_age_days=60  # prune only
"""

import argparse
import provenance_lib
import json
import os
import datetime

REGISTRY_PATH = "server_image_registry.json"
DEFAULT_MAX_AGE_DAYS = 60

def prune_registry(registry, max_age_days, now=None):
    """Returns a copy of registry without images created more than max_age_days ago.

    The input registry is not modified. Entries without a parseable `created`
    timestamp are kept, since their age is unknown.
    """
    if max_age_days <= 0:
        raise ValueError(f"max_age_days must be positive, got {max_age_days}")
    now = now or datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - datetime.timedelta(days=max_age_days)
    kept = []
    for img in registry.get("images", []):
        try:
            created = datetime.datetime.fromisoformat(img["created"].replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError):
            print(f"[*] WARNING: {img.get('digest')} has no valid 'created' timestamp; keeping it.")
            kept.append(img)
            continue
        if created.tzinfo is None:
            created = created.replace(tzinfo=datetime.timezone.utc)
        if created < cutoff:
            print(f"[*] Pruning {img.get('digest')} (created {img['created']}): older than {max_age_days} days.")
        else:
            kept.append(img)
    return {**registry, "images": kept}

def prune_registry_file(max_age_days):
    """Prunes stale entries from server_image_registry.json without adding a digest."""
    if "BUILD_WORKSPACE_DIRECTORY" in os.environ:
        registry_file = os.path.join(os.environ["BUILD_WORKSPACE_DIRECTORY"], REGISTRY_PATH)
    else:
        registry_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), REGISTRY_PATH)

    with open(registry_file, 'r') as f:
        registry = json.load(f)

    pruned = prune_registry(registry, max_age_days)
    if len(pruned["images"]) == len(registry.get("images", [])):
        print(f"\n[*] No entries older than {max_age_days} days. Nothing written.")
        return

    with open(registry_file, 'w') as f:
        json.dump(pruned, f, indent=2)
        f.write('\n')

    print(f"\n[*] Updated {os.path.normpath(REGISTRY_PATH)}")
    print(f"[*] Please run: git add {os.path.normpath(REGISTRY_PATH)} && git commit -m \"Update server registry\"")

def update_registry(digest, overwrite=False, prune=True, max_age_days=DEFAULT_MAX_AGE_DAYS):
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

    if "BUILD_WORKSPACE_DIRECTORY" in os.environ:
        registry_file = os.path.join(os.environ["BUILD_WORKSPACE_DIRECTORY"], REGISTRY_PATH)
    else:
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

    if prune:
        registry = prune_registry(registry, max_age_days)

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
    parser = argparse.ArgumentParser(description="Update server_image_registry.json with a verified digest and prune stale entries.")
    parser.add_argument("digest", type=str, nargs="?", help="The hex SHA256 digest of the server container. If omitted, only prune stale entries.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing provenance if present.")
    parser.add_argument("--prune", action=argparse.BooleanOptionalAction, default=True,
                        help="Also remove registry entries older than --max_age_days (default: enabled; use --no-prune to keep them).")
    parser.add_argument("--max_age_days", type=int, default=DEFAULT_MAX_AGE_DAYS,
                        help=f"Age in days beyond which --prune removes entries (default: {DEFAULT_MAX_AGE_DAYS}).")
    args = parser.parse_args()
    if args.max_age_days <= 0:
        parser.error("--max_age_days must be positive.")
    if args.digest:
        update_registry(args.digest, args.overwrite, args.prune, args.max_age_days)
    elif args.prune:
        prune_registry_file(args.max_age_days)
    else:
        parser.error("nothing to do: pass a digest to add, or drop --no-prune to prune stale entries.")
