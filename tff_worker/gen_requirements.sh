#!/usr/bin/env bash

# Get absolute path to the directory this script is in.
readonly SCRIPT=$(readlink -f $0)
readonly SCRIPT_DIR="$(dirname "${SCRIPT}")"
readonly SCRIPT_DOCKER_IMAGE_NAME="gen-req"

docker build \
  "${SCRIPT_DIR}/requirements_gen" \
  -t ${SCRIPT_DOCKER_IMAGE_NAME}

readonly NEW_DOCKER_IMAGE_ID="$(docker images --format='{{.ID}}' --no-trunc "${SCRIPT_DOCKER_IMAGE_NAME}")"

# The requirements.txt that is output has a warning that setuptools is not
# pinned. This is not a problem for our usage because the base python image has
# a version of setuptools included.
docker run \
  -v ${SCRIPT_DIR}:${SCRIPT_DIR} \
  -w ${SCRIPT_DIR} \
  ${NEW_DOCKER_IMAGE_ID} \
  pip-compile --generate-hashes requirements.in
