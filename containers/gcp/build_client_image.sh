#!/bin/bash
# build_client_image.sh — Build a batched inference GCP client (Oak) container.
#
# Reads approved server digests from server_image_registry.json and bakes them
# into the client's attestation policy. Filters by model, attestation flavor,
# and max server image age.
#
# Usage:
#   ./build_client_image.sh --model=gemma4_e4b
#   ./build_client_image.sh --model=gemma4_31b --attestation=gca --no-alts
#   ./build_client_image.sh --model=gemma4_e4b --max_age_days=30

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────
ATTESTATION="ita"
ALTS=true
MODEL=""
MAX_AGE_DAYS=60
EXTRA_BAZEL_ARGS=""

# ─── Parse flags ─────────────────────────────────────────────────────────────
for arg in "$@"; do
  case "$arg" in
    --model=*)        MODEL="${arg#*=}" ;;
    --attestation=*)  ATTESTATION="${arg#*=}" ;;
    --alts)           ALTS=true ;;
    --no-alts)        ALTS=false ;;
    --max_age_days=*) MAX_AGE_DAYS="${arg#*=}" ;;
    --server_digest=*) EXTRA_BAZEL_ARGS="--//:server_digest=${arg#*=}" ;;
    --help|-h)
      echo "Usage: $0 --model=<model_name> [--attestation=ita|gca] [--alts|--no-alts]"
      echo ""
      echo "Flags:"
      echo "  --model          Model name (required). E.g., gemma4_31b, gemma4_e4b"
      echo "  --attestation    Attestation provider: ita or gca (default: ita)"
      echo "  --alts           Use ALTS transport (default: true)"
      echo "  --no-alts        Disable ALTS transport"
      echo "  --max_age_days   Max age of server images to accept (default: 60)"
      echo "  --server_digest  Manual override: bypass registry, use this single digest"
      exit 0
      ;;
    *)
      echo "ERROR: Unknown flag: $arg" >&2
      echo "Run '$0 --help' for usage." >&2
      exit 1
      ;;
  esac
done

# ─── Validate ────────────────────────────────────────────────────────────────
if [[ -z "$MODEL" ]]; then
  echo "NOTE: --model not specified. Accepting all models from registry." >&2
fi

ATTESTATION=$(echo "$ATTESTATION" | tr '[:upper:]' '[:lower:]')
if [[ "$ATTESTATION" != "ita" && "$ATTESTATION" != "gca" ]]; then
  echo "ERROR: --attestation must be 'ita' or 'gca', got '$ATTESTATION'" >&2
  exit 1
fi

# ─── Derive target and filter names ─────────────────────────────────────────
# Client target name: batched_inference_oak_{ita|gca}_oci_runtime_bundle
CLIENT_TARGET=":batched_inference_oak_${ATTESTATION}_oci_runtime_bundle"

# Server attestation filter must match what the server was built with.
# ITA client talks to ITA servers, GCA client talks to GCA servers.
if $ALTS; then
  SERVER_ATTESTATION="${ATTESTATION}_alts"
else
  SERVER_ATTESTATION="${ATTESTATION}"
fi

# ─── Build ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "══════════════════════════════════════════════════════════════"
echo "  Building client container"
echo "══════════════════════════════════════════════════════════════"
echo "  Model filter:       $MODEL"
echo "  Attestation:        $ATTESTATION (server filter: $SERVER_ATTESTATION)"
echo "  Max server age:     ${MAX_AGE_DAYS} days"
echo "  Bazel target:       $CLIENT_TARGET"
if [[ -n "$EXTRA_BAZEL_ARGS" ]]; then
  echo "  Manual override:    $EXTRA_BAZEL_ARGS"
fi
echo "══════════════════════════════════════════════════════════════"
echo ""

BAZEL_CMD=(
  bazelisk build "$CLIENT_TARGET"
  "--//:server_attestation=$SERVER_ATTESTATION"
  "--//:server_max_age_days=$MAX_AGE_DAYS"
)

if [[ -n "$MODEL" ]]; then
  BAZEL_CMD+=("--//:server_model=$MODEL")
fi

if [[ -n "$EXTRA_BAZEL_ARGS" ]]; then
  BAZEL_CMD+=("$EXTRA_BAZEL_ARGS")
fi

echo ">>> Running: ${BAZEL_CMD[*]}"
echo ""
"${BAZEL_CMD[@]}"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  SUCCESS"
echo "══════════════════════════════════════════════════════════════"
echo "  Client bundle built: $CLIENT_TARGET"
echo "  Approved servers:    model=$MODEL attestation=$SERVER_ATTESTATION max_age=${MAX_AGE_DAYS}d"
echo ""

# Show baked-in server digests from the generated policy.
POLICY_FILE="bazel-bin/policy_${ATTESTATION}.textproto"
if [[ -f "$POLICY_FILE" ]]; then
  DIGESTS=$(grep 'expected_image_digest' "$POLICY_FILE" | sed 's/.*"\(.*\)".*/\1/')
  COUNT=$(echo "$DIGESTS" | grep -c . || true)
  echo "  Baked-in server digests ($COUNT):"
  echo "$DIGESTS" | while read -r d; do
    echo "    - $d"
  done
else
  echo "  (Could not find $POLICY_FILE to list digests)"
fi
echo "══════════════════════════════════════════════════════════════"

