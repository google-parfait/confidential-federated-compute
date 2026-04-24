"""Generates an attestation policy textproto from a server image registry.

Filters registry entries by model, attestation flavor, and max age, then
writes a policy file with the matching digests.

Usage:
    python3 generate_policy.py \
        --registry=server_image_registry.json \
        --output=policy.textproto \
        --verifier_type=ITA \
        --model=gemma4_e4b \
        --attestation=ita_alts \
        --max_age_days=60
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone


def main():
    parser = argparse.ArgumentParser(
        description="Generate attestation policy from server image registry.")
    parser.add_argument("--registry", required=True,
                        help="Path to server_image_registry.json.")
    parser.add_argument("--output", required=True,
                        help="Output policy textproto path.")
    parser.add_argument("--verifier_type", required=True,
                        choices=["ITA", "GCA"],
                        help="Attestation verifier type.")
    parser.add_argument("--model", default="",
                        help="Filter by model name (empty = all).")
    parser.add_argument("--attestation", default="",
                        help="Filter by attestation flavor (empty = all).")
    parser.add_argument("--max_age_days", type=int, default=60,
                        help="Max age in days for registry entries.")
    args = parser.parse_args()

    with open(args.registry) as f:
        registry = json.load(f)

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.max_age_days)
    digests = []

    for entry in registry.get("images", []):
        if args.model and entry.get("model") != args.model:
            continue
        if args.attestation and entry.get("attestation") != args.attestation:
            continue
        try:
            created = datetime.fromisoformat(
                entry["created"].replace("Z", "+00:00"))
            if created < cutoff:
                print(f"Skipping stale: {entry['digest'][:20]}... "
                      f"(created {entry['created']})", file=sys.stderr)
                continue
        except (KeyError, ValueError):
            pass  # Accept entries without valid dates
        digests.append(entry["digest"])

    with open(args.output, "w") as f:
        f.write(f"verifier_type: {args.verifier_type}\n")
        f.write("allow_debug: true\n")
        f.write("allow_outdated_hw_tcb: true\n")
        for d in digests:
            f.write(f'expected_image_digest: "{d}"\n')

    print(f"Policy ({args.verifier_type}): {len(digests)} digest(s), "
          f"max_age={args.max_age_days}d", file=sys.stderr)
    for i, d in enumerate(digests):
        print(f"  [{i+1}] {d}", file=sys.stderr)


if __name__ == "__main__":
    main()
