#!/bin/bash

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
  '--env=KOKORO_ARTIFACTS_DIR'
  "--volume=$KOKORO_ARTIFACTS_DIR:$KOKORO_ARTIFACTS_DIR"
)

if [ -t 0 ]; then
  docker_run_flags+=('--interactive')
fi

# Run the provided command.
docker run "${docker_run_flags[@]}" "$DOCKER_IMAGE_ID" "$@"
