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
# This script runs the provided command in the development Docker container.

cd $(dirname "$0")/..
readonly WORKSPACE_DIR="$PWD"
readonly SCRIPTS_DIR="${WORKSPACE_DIR}/scripts"

readonly DOCKER_IMAGE_ID=$("${SCRIPTS_DIR}/docker_build.sh")

docker_run_flags=(
  '--rm'
  '--tty'
  '--env=TERM=xterm-256color'
  "--volume=$PWD:/workspace"
  # Enable Docker-in-Docker by giving the container access to the host docker
  # daemon.
  "--volume=$XDG_RUNTIME_DIR/docker.sock:/var/run/docker.sock"
  '--volume=/dev/log:/dev/log'
  '--workdir=/workspace'
  '--network=host'
  '--security-opt=seccomp=unconfined'
  '--env=BINARY_OUTPUTS_DIR'
  "--volume=$BINARY_OUTPUTS_DIR:$BINARY_OUTPUTS_DIR"
)

if [ -t 0 ]; then
  docker_run_flags+=('--interactive')
fi

# Run the provided command.
docker run "${docker_run_flags[@]}" "$DOCKER_IMAGE_ID" "$@"
