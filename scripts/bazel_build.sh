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
# including building and testing all targets in the workspace.
#
# If `release` is passed, also builds binaries and OCI Runtime Bundles in
# release mode and exports them to BINARY_OUTPUTS_DIR.
#
# If `continuous` is passed, runs all tests with address sanitizer.
#
# If `sanitizers` is passed, runs all tests with each of the configured
# sanitizers.
set -e

readonly WORKSPACE_DIR="$(dirname -- "$0")/.."
# If bazelisk isn't in user's path, the BAZELISK environment variable may be set
# instead. This may also be used to pass startup options like --nosystem_rc to
# bazel; this usage requires us to not quote ${BAZELISK} when used later.
readonly BAZELISK="${BAZELISK:-bazelisk}"

# List of targets that will be built in release mode, along with the name of the
# resulting artifacts in BINARY_OUTPUTS_DIR.
declare -Ar RELEASE_TARGETS=(
  [//containers/sql_server:oci_runtime_bundle.tar]=sql_server/container.tar
  [//containers/test_concat:oci_runtime_bundle.tar]=test_concat/container.tar
  [//containers/agg_core:oci_runtime_bundle.tar]=agg_core/container.tar
)

if [ "$1" == "continuous" ]; then
  ${BAZELISK} test //... --config=asan --build_tag_filters=-asan --test_tag_filters=-asan
elif [ "$1" == "sanitizers" ]; then
  ${BAZELISK} test //... --config=asan --build_tag_filters=-asan --test_tag_filters=-asan
  ${BAZELISK} test //... --config=tsan --build_tag_filters=-tsan --test_tag_filters=-tsan
  ${BAZELISK} test //... --config=ubsan --build_tag_filters=-noubsan --test_tag_filters=-noubsan
elif [ "$1" == "release" ]; then
  ${BAZELISK} test //...
  ${BAZELISK} build -c opt "${!RELEASE_TARGETS[@]}" \
      //tff_worker/server:pipeline_transform_server_zip
  readonly BAZEL_BIN="$(${BAZELISK} info -c opt bazel-bin)"

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
  "${WORKSPACE_DIR}/scripts/export_container_bundle.sh" -c tff_pipeline_transfer_server \
      -o "${TARGET_DIR}/tff_worker_container.tar"

  # BINARY_OUTPUTS_DIR may be unset if this script is run manually; it'll
  # always be set during CI builds.
  if [[ -n "${BINARY_OUTPUTS_DIR}" ]]; then
    for target in "${!RELEASE_TARGETS[@]}"; do
      src="${BAZEL_BIN}${target/:/\//}"
      dst="${BINARY_OUTPUTS_DIR}/${RELEASE_TARGETS[$target]}"
      mkdir --parents "$(dirname "$dst")"
      cp -f "$src" "$dst"
    done

    mkdir --parents "${BINARY_OUTPUTS_DIR}/tff_worker"
    cp "${TARGET_DIR}/tff_worker_container.tar" "${BINARY_OUTPUTS_DIR}/tff_worker/container.tar"
  fi
else
  ${BAZELISK} test //...
fi
