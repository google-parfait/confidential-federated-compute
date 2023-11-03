#!/bin/bash
#
# Copyright 2023 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# A script that runs the bazel build process in a Docker container with the
# necessary dependencies.
set -e

cd $(dirname "$0")/..
readonly WORKSPACE_DIR="$PWD"
readonly SCRIPTS_RELATIVE_DIR="scripts"

export XDG_RUNTIME_DIR=/var/run

${WORKSPACE_DIR}/${SCRIPTS_RELATIVE_DIR}/docker_run.sh /bin/bash ${SCRIPTS_RELATIVE_DIR}/bazel_build.sh "$@"
