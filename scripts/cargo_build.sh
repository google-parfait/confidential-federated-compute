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
set -x

cd $(dirname -- "$0")/..

source scripts/cargo_common.sh

# List of packages that will be built in release mode, along with the name of
# the resulting artifacts in BINARY_OUTPUTS_DIR.
declare -Ar RELEASE_PACKAGES=(
  [ledger_enclave_app]=ledger/binary
  [replicated_ledger_enclave_app]=replicated_ledger/binary
  [square_enclave_app]=square_example/binary
  [sum_enclave_app]=sum_example/binary
)

build_docker_image

if [ "$1" == "continuous" ] || [ "$1" == "presubmit" ]; then
  # Note that we in pre/postsubmits we build everything with "--locked" to
  # ensure that the lockfile is up-to-date, and hence that builds will
  # reproducibly re-use the same dep versions each time.
  readonly CARGO_LOCKED="--locked"
fi

# The first `cargo build` and `cargo test` invocations build the workspace's default set of
# packages, which are all `no_std` and can target the "x86_64-unknown-none" target.
#
# We then build the tools/explain_fcp_attestation_record package using a separate `cargo`
# invocation, since it is not `no_std` compatible nor can it target the "x86_64-unknown-none"
# target.
#
# See the comments in the root Cargo.toml file for more info.
docker run "${DOCKER_RUN_FLAGS[@]}" "${DOCKER_IMAGE_NAME}" \
    sh -c "
    set -x && \
    cargo build ${CARGO_LOCKED} && cargo test ${CARGO_LOCKED} && \
    (cd tools/explain_fcp_attestation_record && \
        cargo build ${CARGO_LOCKED} && cargo test ${CARGO_LOCKED} )"

if [ "$1" == "release" ]; then
  # Build packages one at a time so that cargo doesn't merge features:
  # https://doc.rust-lang.org/nightly/cargo/reference/resolver.html#features.
  packages="${!RELEASE_PACKAGES[@]}"
  # Note: release builds unconditionally run with "--locked" as well, since
  # those should never update the lock file either.
  docker run "${DOCKER_RUN_FLAGS[@]}" "${DOCKER_IMAGE_NAME}" bash -c \
      "set -x && \
      echo -n \"$packages\" \
      | xargs -d ' ' -I {} cargo build --locked --release -p {}"

  # BINARY_OUTPUTS_DIR may be unset if this script is run manually; it'll
  # always be set during CI builds.
  if [[ -n "${BINARY_OUTPUTS_DIR}" ]]; then
    for pkg in "${!RELEASE_PACKAGES[@]}"; do
      src="target/x86_64-unknown-none/release/${pkg}"
      dst="${BINARY_OUTPUTS_DIR}/${RELEASE_PACKAGES[$pkg]}"
      mkdir --parents "$(dirname "${dst}")"
      cp "${src}" "${dst}"
    done
  fi
fi
