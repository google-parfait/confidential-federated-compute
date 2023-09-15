#!/bin/bash

set -e

cd $(dirname "$0")/..

if ! command -v bazelisk &> /dev/null
then
  echo "bazelisk could not be found: downloading from github"
  rm --recursive --force ./bazelisk-linux-amd64
  wget https://github.com/bazelbuild/bazelisk/releases/download/v1.17.0/bazelisk-linux-amd64 && \
  chmod 755 bazelisk-linux-amd64
  readonly BAZELISK="./bazelisk-linux-amd64"
else
  readonly BAZELISK="bazelisk"
fi

readonly TEST_COMMAND="${BAZELISK} test ..."

# Run Bazelisk as a user rather than root in order to use hermetic python
# version. If this script is run locally, it should already be running as a user
# instead of root.
if (( $EUID == 0 )); then
  readonly NON_ROOT_USER='testing_user'
  if getent passwd ${NON_ROOT_USER} > /dev/null 2>&1; then
    echo "User ${NON_ROOT_USER} already exists"
  else
    echo "Creating user ${NON_ROOT_USER} to run bazel with hermetic Python"
    useradd ${NON_ROOT_USER}
  fi
  readonly BAZELISK_CACHE_DIR="/home/${NON_ROOT_USER}"
  if [ ! -d "${BAZELISK_CACHE_DIR}" ]; then
    echo "Creating directory ${BAZELISK_CACHE_DIR}"
    mkdir ${BAZELISK_CACHE_DIR}
  fi
  chown -R ${NON_ROOT_USER} ${BAZELISK_CACHE_DIR}
  chmod -R a+wrx ${BAZELISK_CACHE_DIR}
  sudo -u ${NON_ROOT_USER} ${TEST_COMMAND}
else
  ${TEST_COMMAND}
fi
