#!/bin/bash
#
# Copyright 2023 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# A script that runs the build process for the entire repository.
# This script can be run locally or in CI.

set -e

cd $(dirname "$0")/..

readonly DOCKER_IMAGE_ID='europe-west2-docker.pkg.dev/oak-ci/oak-development/oak-development@sha256:7b6e401df8e90fec2597806a8c912649b9802de83abe9f6724c3dffe7772f07d'
declare -ar DOCKER_RUN_FLAGS=(
  '--rm'
  "--volume=$PWD/.cargo-cache:/root/.cargo"
  "--volume=$PWD:/workspace"
  '--workdir=/workspace'
)

docker run "${DOCKER_RUN_FLAGS[@]}" "${DOCKER_IMAGE_ID}" sh -c 'cargo build && cargo test'
./kokoro/bazel_build.sh

if [ "$1" == "release" ]; then
  docker run "${DOCKER_RUN_FLAGS[@]}" "${DOCKER_IMAGE_ID}" \
      cargo build --release \
          -p ledger_enclave_app \
          -p square_enclave_app \
          -p sum_enclave_app

  # KOKORO_ARTIFACTS_DIR may be unset if this script is run manually; it'll
  # always be set during CI builds.
  if [[ ! -z "${KOKORO_ARTIFACTS_DIR}" ]]; then
    mkdir -p "${KOKORO_ARTIFACTS_DIR}/binaries"
    cp -v \
        target/x86_64-unknown-none/release/ledger_enclave_app \
        target/x86_64-unknown-none/release/square_enclave_app \
        target/x86_64-unknown-none/release/sum_enclave_app \
        "${KOKORO_ARTIFACTS_DIR}/binaries/"
  fi
fi
