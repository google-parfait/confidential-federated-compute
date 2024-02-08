#!/bin/bash
#
# Copyright 2024 Google LLC.
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
# A script that runs the portions of continuous integration that use cargo,
# including building and testing all targets in the workspace. If `release` is
# passed, also builds the binaries in release mode and exports them to
# BINARY_OUTPUTS_DIR.
set -e

cd $(dirname -- "$0")/..

source scripts/cargo_common.sh

build_docker_image

docker run "${DOCKER_RUN_FLAGS[@]}" "${DOCKER_IMAGE_NAME}" sh -c 'cargo build && cargo test'

if [ "$1" == "release" ]; then
  docker run "${DOCKER_RUN_FLAGS[@]}" "${DOCKER_IMAGE_NAME}" \
      cargo build --release \
          -p ledger_enclave_app \
          -p square_enclave_app \
          -p sum_enclave_app

  # BINARY_OUTPUTS_DIR may be unset if this script is run manually; it'll
  # always be set during CI builds.
  if [[ -n "${BINARY_OUTPUTS_DIR}" ]]; then
    mkdir -p "${BINARY_OUTPUTS_DIR}"
    cp -v \
        target/x86_64-unknown-none/release/ledger_enclave_app \
        target/x86_64-unknown-none/release/square_enclave_app \
        target/x86_64-unknown-none/release/sum_enclave_app \
        "${BINARY_OUTPUTS_DIR}/"
  fi
fi
