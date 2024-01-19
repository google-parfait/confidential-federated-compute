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
      //containers/test_concat:test_concat_server_oci_filesystem_bundle.tar

  # BINARY_OUTPUTS_DIR may be unset if this script is run manually; it'll
  # always be set during CI builds.
  if [[ -n "${BINARY_OUTPUTS_DIR}" ]]; then
    mkdir --parents "${BINARY_OUTPUTS_DIR}"
    readonly BAZEL_BIN="$(bazelisk info -c opt bazel-bin)"
    cp -v \
      "${BAZEL_BIN}/containers/sql_server/sql_server_oci_filesystem_bundle.tar" \
      "${BAZEL_BIN}/containers/test_concat/test_concat_server_oci_filesystem_bundle.tar" \
      "${BINARY_OUTPUTS_DIR}/"
  fi

  # The Python-based tff_worker server has a more complex Dockerfile that hasn't
  # been ported to rules_oci. Until then, build it with Docker.
  ${BAZEL_USER_MODIFIER} bazelisk build --build_python_zip tff_worker/server:pipeline_transform_server

  # Run a command to obtain the bazel-bin output directory for the current
  # configuration. This is guaranteed to be correct even in cases where the
  # bazel-bin symlink cannot be created for some reason- which may be the case
  # when running the build as a non-root user.
  readonly BAZEL_BIN_DIR=$(${BAZEL_USER_MODIFIER} bazelisk info bazel-bin)

  # Copy the zipfile that should be run to a different directory to minimize the
  # build context that must be sent to the Docker daemon.
  readonly TARGET_DIR="./target/bazel"
  rm --recursive --force "${TARGET_DIR}"

  # Build each released container in sequence.
  readonly TFF_TUPLE="tff_pipeline_transform_server,tff_worker/server,pipeline_transform_server.zip"
  for i in ${TFF_TUPLE}
  do
    # Break the tuple into its constituent parts.
    IFS=, read -r CONTAINER_IMAGE BINARY_DIR BINARY <<< "$i"
    # Create a different directory for each container which contains any
    # artifacts necessary for Dockerfile to execute and produce a container, to
    # minimize the build context that needs to be sent to the Docker daemon.
    CONTAINER_DOCKER_CONTEXT_DIR="${TARGET_DIR}/${CONTAINER_IMAGE}"
    echo "[INFO] Building the docker image ${CONTAINER_IMAGE} with workspace ${WORKSPACE_DIR}"
    mkdir --parents "${CONTAINER_DOCKER_CONTEXT_DIR}"
    echo "[INFO] Copying ${BAZEL_BIN_DIR}/${BINARY_DIR}/${BINARY} to ${CONTAINER_DOCKER_CONTEXT_DIR}/${BINARY}."
    cp "${BAZEL_BIN_DIR}/${BINARY_DIR}/${BINARY}" "${CONTAINER_DOCKER_CONTEXT_DIR}/${BINARY}"
    (
      cd "${CONTAINER_DOCKER_CONTEXT_DIR}"
      docker build -f "${WORKSPACE_DIR}/${BINARY_DIR}/Dockerfile" -t "${CONTAINER_IMAGE}" .
    )

    TAR_FILE_NAME="${TARGET_DIR}/${CONTAINER_IMAGE}_oci_filesystem_bundle.tar"
    "${SCRIPTS_DIR}/export_container_bundle.sh" \
        -c "${CONTAINER_IMAGE}" \
        -o "${TAR_FILE_NAME}"
    # BINARY_OUTPUTS_DIR may be unset if this script is run manually; it'll
    # always be set during CI builds.
    if [[ -n "${BINARY_OUTPUTS_DIR}" ]]; then
      mkdir --parents "${BINARY_OUTPUTS_DIR}"
      cp -v "${TAR_FILE_NAME}" \
          "${BINARY_OUTPUTS_DIR}/"
    fi
  done
fi
