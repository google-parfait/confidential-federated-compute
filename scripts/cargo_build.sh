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

cd $(dirname "$0")/..

readonly DOCKER_IMAGE_NAME=confidential-federated-compute-rust
# At least some versions of rootless docker don't work without a context dir.
readonly DOCKER_CONTEXT_DIR="$(mktemp -d)"
trap 'rm -rf -- "${DOCKER_CONTEXT_DIR}"' EXIT
docker build --cache-from $DOCKER_IMAGE_NAME --tag $DOCKER_IMAGE_NAME -f - "${DOCKER_CONTEXT_DIR}" <<EOF
FROM rust@sha256:4013eb0e2e5c7157d5f0f11d83594d8bad62238a86957f3d57e447a6a6bdf563
RUN rustup default nightly-2023-11-15
RUN rustup target add x86_64-unknown-none
RUN curl -LSso protoc.zip https://github.com/protocolbuffers/protobuf/releases/download/v25.2/protoc-25.2-linux-x86_64.zip && \
    echo "78ab9c3288919bdaa6cfcec6127a04813cf8a0ce406afa625e48e816abee2878 protoc.zip" | sha256sum -c && \
    unzip -q protoc.zip -d /usr/local/protobuf && \
    rm protoc.zip
ENV PROTOC /usr/local/protobuf/bin/protoc
EOF

declare -ar DOCKER_RUN_FLAGS=(
  '--rm'
  '--tty'
  '--env=CARGO_HOME=/root/.cargo'
  '--env=TERM'
  "--volume=$PWD/.cargo-cache:/root/.cargo"
  "--volume=$PWD:/workspace"
  '--workdir=/workspace'
)

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
