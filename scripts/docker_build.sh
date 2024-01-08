#!/bin/bash
#
# Copyright 2023 Google LLC.
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
#
# Builds a docker container for use in performing reproducible builds.
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
