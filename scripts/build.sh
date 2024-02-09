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
# If `continuous` is passed, runs all bazel tests with address sanitizer.
#
# If `sanitizers` is passed, runs all bazel tests with each of the configured
# sanitizers.
set -e

cd $(dirname -- "$0")/..

./scripts/cargo_build.sh "$@"
./scripts/bazel_build.sh "$@"
