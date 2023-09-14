#!/bin/bash

# A script that runs the portions of continuous integration that use Bazel,
# including building and testing all targets in the workspace. If `release` is
# passed, also builds the Pipeline Transform Server Docker container, and
# creates a tarball of an OCI Runtime Bundle from the Docker container to
# export.
set -e

cd $(dirname "$0")/..
readonly WORKSPACE_DIR="$PWD"
readonly SCRIPTS_DIR="${WORKSPACE_DIR}/scripts"

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

# Run Bazelisk as a user rather than root in order to use hermetic python
# version. If this script is run locally, it should already be running as a user
# instead of root.
if (( $EUID == 0 )); then
  readonly NON_ROOT_USER='testing_user'
  if getent passwd "${NON_ROOT_USER}" > /dev/null 2>&1; then
    echo "User ${NON_ROOT_USER} already exists"
  else
    echo "Creating user ${NON_ROOT_USER} to run bazel with hermetic Python"
    useradd "${NON_ROOT_USER}"
  fi
  readonly BAZELISK_CACHE_DIR="/home/${NON_ROOT_USER}"
  if [ ! -d "${BAZELISK_CACHE_DIR}" ]; then
    echo "Creating directory ${BAZELISK_CACHE_DIR}"
    mkdir "${BAZELISK_CACHE_DIR}"
  fi
  chown -R "${NON_ROOT_USER}" "${BAZELISK_CACHE_DIR}"
  chmod -R a+wrx "${BAZELISK_CACHE_DIR}"
  readonly BAZEL_USER_MODIFIER="sudo -u ${NON_ROOT_USER}"
else
  readonly BAZEL_USER_MODIFIER=""
fi

${BAZEL_USER_MODIFIER} ${BAZELISK} test ...

if [ "$1" == "release" ]; then
  ${BAZEL_USER_MODIFIER} ${BAZELISK} build --build_python_zip tff_worker/server:pipeline_transform_server
  # Run a command to obtain the bazel-bin output directory for the current
  # configuration. This is guaranteed to be correct even in cases where the
  # bazel-bin symlink cannot be created for some reason- which may be the case
  # when running the build as a non-root user.
  readonly BAZEL_BIN_DIR=$(${BAZEL_USER_MODIFIER} ${BAZELISK} info bazel-bin)

  # Copy the zipfile that should be run to a different directory to minimize the
  # build context that must be sent to the Docker daemon.
  readonly TARGET_DIR="./target/bazel"
  rm --recursive --force "${TARGET_DIR}"
  mkdir --parents "${TARGET_DIR}"
  cp "${BAZEL_BIN_DIR}/tff_worker/server/pipeline_transform_server.zip" "${TARGET_DIR}/pipeline_transform_server.zip"


  readonly CONTAINER_IMAGE='pipeline_transform_server'
  echo "[INFO] Building the docker image ${CONTAINER_IMAGE} with workspace ${WORKSPACE_DIR}"
  (
    cd "${TARGET_DIR}"
    docker build -f "${WORKSPACE_DIR}/tff_worker/server/Dockerfile" -t "${CONTAINER_IMAGE}" .
  )

  readonly TAR_FILE_NAME="${TARGET_DIR}/pipeline_transform_server_oci_filesystem_bundle.tar"
  "${SCRIPTS_DIR}/export_container_bundle.sh" \
      -c "${CONTAINER_IMAGE}" \
      -o "${TAR_FILE_NAME}"

  # KOKORO_ARTIFACTS_DIR may be unset if this script is run manually; it'll
  # always be set during CI builds.
  if [[ ! -z "${KOKORO_ARTIFACTS_DIR}" ]]; then
    mkdir --parents "${KOKORO_ARTIFACTS_DIR}/binaries"
    cp -v \
        "${TAR_FILE_NAME}" \
        "${KOKORO_ARTIFACTS_DIR}/binaries/"
  fi
fi
