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

bazelisk test -- ... -containers/sql_server:pipeline_transform_server_benchmarks

if [ "$1" == "release" ]; then
  ${BAZEL_USER_MODIFIER} bazelisk build --build_python_zip tff_worker/server:pipeline_transform_server
  ${BAZEL_USER_MODIFIER} bazelisk build containers/sql_server:main

  # Run a command to obtain the bazel-bin output directory for the current
  # configuration. This is guaranteed to be correct even in cases where the
  # bazel-bin symlink cannot be created for some reason- which may be the case
  # when running the build as a non-root user.
  readonly BAZEL_BIN_DIR=$(${BAZEL_USER_MODIFIER} bazelisk info bazel-bin)

  # Copy the zipfile that should be run to a different directory to minimize the
  # build context that must be sent to the Docker daemon.
  readonly TARGET_DIR="./target/bazel"
  rm --recursive --force "${TARGET_DIR}"
  mkdir --parents "${TARGET_DIR}"

  cp "${BAZEL_BIN_DIR}/tff_worker/server/pipeline_transform_server.zip" "${TARGET_DIR}/pipeline_transform_server.zip"
  cp "${BAZEL_BIN_DIR}/containers/sql_server/main" "${TARGET_DIR}/main"

  readonly TFF_CONTAINER_IMAGE='tff_pipeline_transform_server'
  readonly SQL_CONTAINER_IMAGE='sql_pipeline_transform_server'
  echo "[INFO] Building the docker images ${TFF_CONTAINER_IMAGE} and ${SQL_CONTAINER_IMAGE} with workspace ${WORKSPACE_DIR}"
  (
    cd "${TARGET_DIR}"
    docker build -f "${WORKSPACE_DIR}/tff_worker/server/Dockerfile" -t "${TFF_CONTAINER_IMAGE}" .
    docker build -f "${WORKSPACE_DIR}/containers/sql_server/Dockerfile" -t "${SQL_CONTAINER_IMAGE}" .
  )

  readonly TFF_TAR_FILE_NAME="${TARGET_DIR}/tff_pipeline_transform_server_oci_filesystem_bundle.tar"
  "${SCRIPTS_DIR}/export_container_bundle.sh" \
      -c "${TFF_CONTAINER_IMAGE}" \
      -o "${TFF_TAR_FILE_NAME}"

  readonly SQL_TAR_FILE_NAME="${TARGET_DIR}/sql_pipeline_transform_server_oci_filesystem_bundle.tar"
  "${SCRIPTS_DIR}/export_container_bundle.sh" \
      -c "${SQL_CONTAINER_IMAGE}" \
      -o "${SQL_TAR_FILE_NAME}"

  # BINARY_OUTPUTS_DIR may be unset if this script is run manually; it'll
  # always be set during CI builds.
  if [[ -n "${BINARY_OUTPUTS_DIR}" ]]; then
    mkdir --parents "${BINARY_OUTPUTS_DIR}"
    cp -v \
        "${TFF_TAR_FILE_NAME}" \
        "${BINARY_OUTPUTS_DIR}/"
    cp -v \
        "${SQL_TAR_FILE_NAME}" \
        "${BINARY_OUTPUTS_DIR}/"
  fi
fi
