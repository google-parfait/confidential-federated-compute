#!/bin/bash
cd $(dirname "$0")/..
readonly WORKSPACE_DIR="$PWD"
readonly SCRIPTS_DIR="${WORKSPACE_DIR}/scripts"

readonly DOCKER_IMAGE_NAME='confidential-federated-compute-development'

# Enable BuildKit for improved performance and storage management.
#
# See https://docs.docker.com/develop/develop-images/build_enhancements/.
#
# TODO(#3225): Re-enable buildkit support when it is compatible with rootless Docker.
# export DOCKER_BUILDKIT=1

(
  cd "${WORKSPACE_DIR}/development"

  docker build \
    --cache-from="${DOCKER_IMAGE_NAME}" \
    --tag="${DOCKER_IMAGE_NAME}" \
    . 1>&2
)

# Write the id (which corresponds to the SHA256 hash) of the image to stdout so
# it can be used in other scripts.
readonly DOCKER_IMAGE_ID="$(docker images --format='{{.ID}}' --no-trunc "${DOCKER_IMAGE_NAME}")"
echo "${DOCKER_IMAGE_ID}"
