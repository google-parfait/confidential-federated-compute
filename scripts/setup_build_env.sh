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
# Script that installs all the non rust binaries needed to build the project.
# Meant to be used with the official rust container as a starting point.
set -o xtrace
set -o errexit
set -o nounset
set -o pipefail

# Download a artifact and verify its hash against an expected value.
download_and_verify_hash() {
  # Parameter validation
  if [[ $# -ne 3 ]]; then
    echo "Usage: download_and_verify <URL> <OUTPUT_PATH> <EXPECTED_SHA256>" >&2
    return 1
  fi

  # Assign arguments to local variables
  local url="$1"
  local output_path="$2"
  local expected_sha256="$3"

  # Download the file
  curl -f -L -o "${output_path}" "${url}" || {
    echo "ERROR: Download failed ($?)" >&2
    return 1
  }

  echo "Downloaded ${url} to ${output_path}."

  # Calculate the SHA256 hash of the downloaded file
  local actual_sha256=$(sha256sum "${output_path}" | awk '{print $1}')

  # Hash verification
  if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
    rm "${output_path}"
    echo "ERROR: SHA256 mismatch. Expected ${expected_sha256}, got ${actual_sha256}." >&2
    return 1
  fi

  echo "Download and verification successful."
}

# Rustup is prerequisite.
if ! command -v rustup >/dev/null 2>&1; then
  echo "ERROR: rustup is must be already installed. Please install it to proceed."
  exit 1
fi

# Configure rust.
rustup default nightly-2023-11-15
rustup target add x86_64-unknown-none

# Install bazelisk/bazel.
if ! command -v bazelisk >/dev/null 2>&1; then
  download_and_verify_hash \
    "https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64" \
    "/tmp/bazelisk" \
    "d28b588ac0916abd6bf02defb5433f6eddf7cba35ffa808eabb65a44aab226f7"
  mv "/tmp/bazelisk" "/usr/bin"
  chmod +x "/usr/bin/bazelisk"
fi

# Install protoc.
if ! command -v protoc >/dev/null 2>&1; then
  download_and_verify_hash \
    "https://github.com/protocolbuffers/protobuf/releases/download/v25.2/protoc-25.2-linux-x86_64.zip" \
    "/tmp/protoc.zip" \
    "78ab9c3288919bdaa6cfcec6127a04813cf8a0ce406afa625e48e816abee2878"
  # Install instructions from https://google.github.io/proto-lens/installing-protoc.html.
  unzip -o "/tmp/protoc.zip" -d /usr/local bin/protoc
  unzip -o "/tmp/protoc.zip" -d /usr/local 'include/*'
  rm "/tmp/protoc.zip"
fi

if [[ -v GITHUB_ACTION ]]; then
  # Solves the following error when running on GitHub Actions:
  #
  # fatal: detected dubious ownership in repository at '/workspace'
  #   To add an exception for this directory, call:
  #   git config --global --add safe.directory /workspace
  git config --global --add safe.directory /workspace
  echo "Added /workspace to git config's safe.directory."

  # GitHub Actions must clone submodules explicitly.
  git submodule update --init
  echo "Updated submodules."
fi
