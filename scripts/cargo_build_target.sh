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
# Builds one or more cargo packages in release mode.
set -ex

# List of available cargo packages, along with the place where the artifact
# will appear. Positional arguments are required to be in the key set.
declare -Ar RELEASE_PACKAGES=(
  [ledger_enclave_app]=ledger/binary
  [square_enclave_app]=square_example/binary
  [sum_enclave_app]=sum_example/binary
)

# Maybe this has been done before, so don't abort on failure.
git submodule update --init || true

cd $(dirname -- "$0")/..
source scripts/cargo_common.sh

declare -a packages

while [[ $# -gt 0 ]]; do
  case $1 in
    -o|--output_dir)
      output_dir="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      packages+=("$1")
      shift # past argument
      ;;
  esac
done

[ -z "${output_dir}" ] && exit 1
[ ${#packages[@]} -eq 0 ] && exit 1

build_docker_image

# Build packages one at a time so that cargo doesn't merge features:
# https://doc.rust-lang.org/nightly/cargo/reference/resolver.html#features.
# Note: release builds are unconditionally `--locked`, since a lock file
# update should not happen.
docker run "${DOCKER_RUN_FLAGS[@]}" "${DOCKER_IMAGE_NAME}" bash -c \
    "set -x && \
    echo -n \"$packages\" \
    | xargs -d ' ' -I {} cargo build --locked --release -p {}"

for pkg in "${packages[@]}"; do
  src="target/x86_64-unknown-none/release/${pkg}"
  dst="${output_dir}/${RELEASE_PACKAGES[$pkg]}"
  mkdir --parents "$(dirname "${dst}")"
  cp --preserve=timestamps "${src}" "${dst}"
done
