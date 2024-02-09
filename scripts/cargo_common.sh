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
# Common settings and functions for cargo docker images.
set -e

f() {
  readonly DOCKER_IMAGE_NAME="confidential-federated-compute-rust"

  # Compute an absolute path for the workspace root since paths containing "../"
  # don't always work with docker.
  local workspace_root
  workspace_root="$(readlink -f -- "$(dirname -- "${BASH_SOURCE[0]}")/..")"
  DOCKER_RUN_FLAGS=(
    '--rm'
    '--tty'
    '--env=CARGO_HOME=/root/.cargo'
    '--env=TERM'
    "--volume=${workspace_root}/.cargo-cache:/root/.cargo"
    "--volume=${workspace_root}:/workspace"
    '--workdir=/workspace'
  )

  build_docker_image() {
    local context_dir
    context_dir="$(mktemp -d)"
    trap 'rm -rf -- "${context_dir}"' EXIT
    docker build --cache-from "${DOCKER_IMAGE_NAME}" --tag "${DOCKER_IMAGE_NAME}" -f - "${context_dir}" <<EOF
FROM rust@sha256:4013eb0e2e5c7157d5f0f11d83594d8bad62238a86957f3d57e447a6a6bdf563
RUN rustup default nightly-2023-11-15
RUN rustup target add x86_64-unknown-none
RUN curl -LSso protoc.zip https://github.com/protocolbuffers/protobuf/releases/download/v25.2/protoc-25.2-linux-x86_64.zip && \
    echo "78ab9c3288919bdaa6cfcec6127a04813cf8a0ce406afa625e48e816abee2878 protoc.zip" | sha256sum -c && \
    unzip -q protoc.zip -d /usr/local/protobuf && \
    rm protoc.zip
ENV PROTOC /usr/local/protobuf/bin/protoc
EOF
  }
}
f
