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
# A script that runs the portions of continuous integration that use Bazel,
# including building and testing all targets in the workspace. Assumes all
# necessary dependencies for building are present, so it is recommended to run
# this script from within the development Docker container. If `release` is
# passed, also builds a Pipeline Transform Server Docker container for
# executing TFF computations and a Pipeline Transform Server Docker container
# for executing SQL queries. Also creates tarballs of OCI Runtime Bundles
# from the Docker containers and exports them to BINARY_OUTPUTS_DIR.
set -e

cd $(dirname "$0")/..
readonly WORKSPACE_DIR="$PWD"
readonly SCRIPTS_DIR="${WORKSPACE_DIR}/scripts"

bazelisk test //...

if [ "$1" == "release" ]; then
  bazelisk build -c opt \
      //containers/sql_server:sql_server_oci_filesystem_bundle.tar \
      //containers/test_concat:test_concat_server_oci_filesystem_bundle.tar \
      //tff_worker/server:pipeline_transform_server_zip
  readonly BAZEL_BIN="$(bazelisk info -c opt bazel-bin)"

  # The Python-based tff_worker server has a more complex Dockerfile that hasn't
  # been ported to rules_oci. Until then, build it with Docker.
  # Copy inputs to a temporary directory to minimize the build context sent to
  # the Docker daemon.
  readonly DOCKER_CONTEXT_DIR="$(mktemp -d)"
  trap 'rm -rf -- "${DOCKER_CONTEXT_DIR}"' EXIT
  cp "${BAZEL_BIN}/tff_worker/server/pipeline_transform_server.zip" \
      "${DOCKER_CONTEXT_DIR}/"
  docker build -f "${WORKSPACE_DIR}/tff_worker/server/Dockerfile" \
      -t tff_pipeline_transfer_server "${DOCKER_CONTEXT_DIR}"
  readonly TARGET_DIR="${WORKSPACE_DIR}/target/bazel"
  mkdir --parents "${TARGET_DIR}"
  "${SCRIPTS_DIR}/export_container_bundle.sh" -c tff_pipeline_transfer_server \
      -o "${TARGET_DIR}/tff_pipeline_transform_server_oci_filesystem_bundle.tar"

  # BINARY_OUTPUTS_DIR may be unset if this script is run manually; it'll
  # always be set during CI builds.
  if [[ -n "${BINARY_OUTPUTS_DIR}" ]]; then
    mkdir --parents "${BINARY_OUTPUTS_DIR}"
    cp -v \
      "${BAZEL_BIN}/containers/sql_server/sql_server_oci_filesystem_bundle.tar" \
      "${BAZEL_BIN}/containers/test_concat/test_concat_server_oci_filesystem_bundle.tar" \
      "${TARGET_DIR}/tff_pipeline_transform_server_oci_filesystem_bundle.tar" \
      "${BINARY_OUTPUTS_DIR}/"
  fi
fi
