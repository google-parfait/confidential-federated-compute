#!/bin/bash
# build_and_push_server_image.sh — Build, tag, and push a batched inference GCP server image.
#
# Usage:
#   ./build_and_push_server_image.sh --model=gemma4_31b
#   ./build_and_push_server_image.sh --model=gemma4_e4b --attestation=gca --no-alts
#   ./build_and_push_server_image.sh --model=gemma4_31b --gcs_bucket=gs://other-bucket
#   ./build_and_push_server_image.sh --model=gemma4_31b --destination=us-docker.pkg.dev/my-project/my-repo

set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────
ATTESTATION="ita"
ALTS=true
MODEL=""
DESTINATION="us-docker.pkg.dev/private-inference/offloading"
GCS_BUCKET=""
UPDATE_REGISTRY=true
DRY_RUN=false

# ─── Parse flags ─────────────────────────────────────────────────────────────
for arg in "$@"; do
  case "$arg" in
    --model=*)        MODEL="${arg#*=}" ;;
    --attestation=*)  ATTESTATION="${arg#*=}" ;;
    --alts)           ALTS=true ;;
    --no-alts)        ALTS=false ;;
    --destination=*)  DESTINATION="${arg#*=}" ;;
    --gcs_bucket=*)   GCS_BUCKET="${arg#*=}" ;;
    --update_registry) UPDATE_REGISTRY=true ;;
    --no-update_registry) UPDATE_REGISTRY=false ;;
    --dry-run)        DRY_RUN=true ;;
    --help|-h)
      echo "Usage: $0 --model=<model_name> [--attestation=ita|gca] [--alts|--no-alts] [--destination=<registry>]"
      echo ""
      echo "Flags:"
      echo "  --model          Model name (required). E.g., gemma4_31b, gemma4_e4b"
      echo "  --attestation    Attestation provider (default: ita)"
      echo "  --alts           Use ALTS transport (default: true)"
      echo "  --no-alts        Disable ALTS transport"
      echo "  --destination    Container registry path (default: us-docker.pkg.dev/private-inference/offloading)"
      echo "  --gcs_bucket     Override GCS bucket for model weights (e.g., gs://my-bucket)"
      echo "  --update_registry    Update server_image_registry.json after push (default: true)"
      echo "  --no-update_registry Skip registry update"
      echo "  --dry-run        Print what would be done without executing"
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
  echo "ERROR: --model is required." >&2
  echo "Available models: gemma4_31b, gemma4_e4b" >&2
  echo "Example: $0 --model=gemma4_31b" >&2
  exit 1
fi

# ─── Derive names ────────────────────────────────────────────────────────────
if [[ "$ALTS" == true ]]; then
  SUFFIX="${ATTESTATION}_alts"
else
  SUFFIX="${ATTESTATION}"
fi

RUNNER_TARGET=":load_and_print_digest_runner_batched_inference_gcp_${SUFFIX}_${MODEL}"
LOCAL_TAG="batched_inference_gcp_${SUFFIX}_${MODEL}:latest"
REMOTE_TAG="${DESTINATION}/batched_inference:${SUFFIX}_${MODEL}_latest"

echo "══════════════════════════════════════════════════════════════"
echo "  Model:        $MODEL"
echo "  Attestation:  $ATTESTATION"
echo "  ALTS:         $ALTS"
echo "  Suffix:       $SUFFIX"
echo "  GCS bucket:   ${GCS_BUCKET:-<default from MODULE.bazel>}"
echo "  Runner:       $RUNNER_TARGET"
echo "  Local tag:    $LOCAL_TAG"
echo "  Remote tag:   $REMOTE_TAG"
echo "══════════════════════════════════════════════════════════════"

if [[ "$DRY_RUN" == true ]]; then
  echo "[DRY RUN] Would execute:"
  echo "  1. bazelisk run $RUNNER_TARGET"
  echo "  2. docker tag $LOCAL_TAG $REMOTE_TAG"
  echo "  3. docker push $REMOTE_TAG"
  exit 0
fi

# ─── Step 1: Build and load the image ────────────────────────────────────────
echo ""
echo ">>> Step 1/4: Building and loading image via bazel runner..."
echo ""

BUILD_LOG=$(mktemp)
trap "rm -f $BUILD_LOG" EXIT

# Cache-bust with timestamp to force re-evaluation of the genrule.
# Pass through GCS bucket override if specified.
GCS_BUCKET_ARG=""
if [[ -n "$GCS_BUCKET" ]]; then
  GCS_BUCKET_ARG="--repo_env=GCS_MODEL_BUCKET=${GCS_BUCKET}"
fi

if ! bazelisk run \
    --action_env="BUILD_TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)" \
    --lockfile_mode=refresh \
    ${GCS_BUCKET_ARG} \
    "$RUNNER_TARGET" 2>&1 | tee "$BUILD_LOG"; then
  echo "" >&2
  echo "ERROR: Bazel runner failed. See output above." >&2
  exit 1
fi

# ─── Step 2: Verify the image was loaded ─────────────────────────────────────
echo ""
echo ">>> Step 2/3: Verifying image was loaded..."
echo ""

if ! docker image inspect "$LOCAL_TAG" > /dev/null 2>&1; then
  echo "ERROR: Image '$LOCAL_TAG' not found in local Docker after runner completed." >&2
  echo "The runner may have succeeded but the image tag doesn't match." >&2
  echo "Check 'docker images' output." >&2
  exit 1
fi

echo "OK: Image '$LOCAL_TAG' found in local Docker."

# ─── Step 3: Tag and push to remote registry ─────────────────────────────────
echo ""
echo ">>> Step 3/3: Tagging and pushing image to remote registry..."
echo ""

docker tag "$LOCAL_TAG" "$REMOTE_TAG"
echo "Tagged: $LOCAL_TAG -> $REMOTE_TAG"

PUSH_LOG=$(mktemp)
trap "rm -f $BUILD_LOG $PUSH_LOG" EXIT

if ! docker push "$REMOTE_TAG" 2>&1 | tee "$PUSH_LOG"; then
  echo "ERROR: docker push failed." >&2
  echo "Make sure you're authenticated: gcloud auth configure-docker us-docker.pkg.dev" >&2
  exit 1
fi

# The authoritative digest is the one reported by `docker push`.
# Line looks like: "tag_name: digest: sha256:abc123... size: 1234"
DIGEST=$(grep -oP 'digest: \Ksha256:[a-f0-9]+' "$PUSH_LOG" | tail -n 1 || true)
if [[ -z "$DIGEST" ]]; then
  # Fallback: query the registry directly.
  DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' "$REMOTE_TAG" 2>/dev/null | grep -oP 'sha256:[a-f0-9]+' || true)
fi
if [[ -z "$DIGEST" ]]; then
  echo "WARNING: Could not extract pushed digest. Check 'docker push' output above." >&2
  DIGEST="UNKNOWN"
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  SUCCESS"
echo "══════════════════════════════════════════════════════════════"
echo "  Model:        $MODEL"
echo "  Attestation:  ${SUFFIX}"
echo "  Local tag:    $LOCAL_TAG"
echo "  Remote tag:   $REMOTE_TAG"
echo "  Pushed digest: $DIGEST"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo ""
echo "To use this digest in the client build:"
echo "  bazel build :batched_inference_oak_ita_oci_runtime_bundle \\"
echo "      --//:server_digest=\"$DIGEST\""
echo ""

# ─── Step 4: Update server_image_registry.json ───────────────────────────────
REGISTRY="$(cd "$(dirname "$0")" && pwd)/server_image_registry.json"
if [[ "$UPDATE_REGISTRY" != true ]]; then
  echo "Registry update skipped (--no-update_registry)."
elif [[ "$DIGEST" == "UNKNOWN" ]]; then
  echo "WARNING: Skipping registry update (digest unknown)." >&2
else
  CREATED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  python3 -c "
import json, sys
entry = {
    'model': sys.argv[1],
    'attestation': sys.argv[2],
    'digest': sys.argv[3],
    'tag': sys.argv[4],
    'created': sys.argv[5],
}
try:
    with open(sys.argv[6]) as f:
        registry = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    registry = {'images': []}
registry['images'].append(entry)
with open(sys.argv[6], 'w') as f:
    json.dump(registry, f, indent=2)
    f.write('\\n')
print(json.dumps(entry, indent=2))
" "$MODEL" "$SUFFIX" "$DIGEST" "$REMOTE_TAG" "$CREATED" "$REGISTRY"
  echo ""
  echo "Updated $REGISTRY"
fi
