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
# A script that runs the build process for the entire repository.
# This script can be run locally or in CI.
#
# If `release` is passed, this indicates that the script should build the
# artifacts for release. In other words, it will build binaries in the form in
# which they can be executed in an enclave and export them to
# BINARY_OUTPUTS_DIR.
#
# If `continuous` is passed, runs all tests with address sanitizer.
#
# If `sanitizers` is passed, runs all tests with each of the configured
# sanitizers.
set -e
set -x

# If bazelisk isn't in user's path, the BAZELISK environment variable may be set
# instead. This may also be used to pass startup options like --nosystem_rc to
# bazel; this usage requires us to not quote ${BAZELISK} when used later.
readonly BAZELISK="${BAZELISK:-bazelisk}"

if [ "$1" == "continuous" ]; then
  ${BAZELISK} test //... --config=asan --build_tag_filters=-asan --test_tag_filters=-asan
elif [ "$1" == "sanitizers" ]; then
  ${BAZELISK} test //... --config=asan --build_tag_filters=-asan --test_tag_filters=-asan
  ${BAZELISK} test //... --config=tsan --build_tag_filters=-tsan --test_tag_filters=-tsan
  ${BAZELISK} test //... --config=ubsan --build_tag_filters=-noubsan --test_tag_filters=-noubsan
elif [ "$1" == "release" ]; then
  ${BAZELISK} test //...

  # BINARY_OUTPUTS_DIR may be unset if this script is run manually; it'll
  # always be set during CI builds.
  if [[ -n "${BINARY_OUTPUTS_DIR}" ]]; then
    ${BAZELISK} run -c opt //:install_all_binaries -- --destdir "${BINARY_OUTPUTS_DIR}"
  else
    # If unset, verify the binaries can be built with -c opt.
    ${BAZELISK} build -c opt //:install_all_binaries
  fi
else
  ${BAZELISK} test //...
fi
