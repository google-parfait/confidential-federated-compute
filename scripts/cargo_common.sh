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
    cp scripts/setup_build_env.sh "${context_dir}"
    trap 'rm -rf -- "${context_dir}"' EXIT
    docker build --cache-from "${DOCKER_IMAGE_NAME}" --tag "${DOCKER_IMAGE_NAME}" -f - "${context_dir}" <<EOF
FROM rust@sha256:4013eb0e2e5c7157d5f0f11d83594d8bad62238a86957f3d57e447a6a6bdf563
COPY setup_build_env.sh /tmp/setup_build_env.sh
RUN /tmp/setup_build_env.sh
EOF
  }
}
f
