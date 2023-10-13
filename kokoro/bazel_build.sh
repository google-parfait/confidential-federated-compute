#!/bin/bash

# A script that runs the bazel build process in a Docker container with the
# necessary dependencies.
set -e

cd $(dirname "$0")/..
readonly WORKSPACE_DIR="$PWD"
readonly SCRIPTS_RELATIVE_DIR="scripts"

export XDG_RUNTIME_DIR=/var/run

${WORKSPACE_DIR}/${SCRIPTS_RELATIVE_DIR}/docker_run.sh /bin/bash ${SCRIPTS_RELATIVE_DIR}/bazel_build.sh "$@"
