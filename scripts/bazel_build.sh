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
set -x

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
  [//containers/confidential_transform_test_concat:oci_runtime_bundle.tar]=confidential_transform_test_concat/container.tar
  [//containers/agg_core:oci_runtime_bundle.tar]=agg_core/container.tar
)

if [ "$1" == "continuous" ]; then
  ${BAZELISK} test //... --config=asan --build_tag_filters=-asan --test_tag_filters=-asan
elif [ "$1" == "sanitizers" ]; then
  ${BAZELISK} test //... --config=asan --build_tag_filters=-asan --test_tag_filters=-asan
  ${BAZELISK} test //... --config=tsan --build_tag_filters=-tsan --test_tag_filters=-tsan
  ${BAZELISK} test //... --config=ubsan --build_tag_filters=-noubsan --test_tag_filters=-noubsan
elif [ "$1" == "release" ]; then
  # Tests fail after 2h in GitHub actions, seemingly since the worker runs
  # out of space. Hence skip tests.
  if [ -z "${GITHUB_ACTION}" ]; then
    ${BAZELISK} test //...
  fi
  ${BAZELISK} build -c opt "${!RELEASE_TARGETS[@]}"

  # BINARY_OUTPUTS_DIR may be unset if this script is run manually; it'll
  # always be set during CI builds.
  if [[ -n "${BINARY_OUTPUTS_DIR}" ]]; then
    readonly BAZEL_BIN="$(${BAZELISK} info -c opt bazel-bin)"
    for target in "${!RELEASE_TARGETS[@]}"; do
      src="${BAZEL_BIN}${target/:/\//}"
      dst="${BINARY_OUTPUTS_DIR}/${RELEASE_TARGETS[$target]}"
      if [[ "${GITHUB_ACTION:-}" == "provenance" ]]; then
        # The provenance GitHub Action expects a flat set of output binaries.
        flattened_target=${RELEASE_TARGETS[$target]//\//_}
        dst="${BINARY_OUTPUTS_DIR}/${flattened_target}"
        echo "using flat binary output for ${dst}"
      fi
      mkdir --parents "$(dirname "$dst")"
      cp -f "$src" "$dst"
    done
  fi
else
  ${BAZELISK} test //...
fi
