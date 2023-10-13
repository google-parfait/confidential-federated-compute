#!/bin/bash

# Script to export a container image and convert it into an OCI Runtime Bundle.
# (https://github.com/opencontainers/runtime-spec/blob/4e3b9264a330d094b0386c3703c5f379119711e8/bundle.md)
#
# This script is adapted from https://github.com/project-oak/oak/blob/main/scripts/export_container_bundle.
# Once Oak has a more reusable way of generating the OCI Runtime Bundle, (see
# https://github.com/project-oak/oak/issues/4283) we should use that instead.
set -e

readonly SCRIPTS_DIR="$(dirname "$0")"
readonly ABSOLUTE_SCRIPT_PATH="$( cd -- ${SCRIPTS_DIR} >/dev/null 2>&1 ; pwd -P )"

print_usage_and_exit() {
  echo "Usage:"
  echo "  $0 [options] -c <container-image>

  Exports the filesystem in the specified container image as an OCI Runtime Bundle.

Options:
    -h                    Display this help and exit.
    -o <output-file>      The target output file for the bundle. If this is not specified
                          the bundle will be created as <container-image>.tar"
  exit 0
}

while getopts "hc:o:" opt; do
  case $opt in
    h)
      print_usage_and_exit
      ;;
    o)
      readonly OUTPUT_FILE="${OPTARG}"
      ;;
    c)
      readonly CONTAINER_IMAGE="${OPTARG}"
      ;;
    *)
      echo "Invalid argument: ${OPTARG}"
      exit 1;;
  esac
done

if [ -z "${CONTAINER_IMAGE}" ]; then
  echo "Missing required option: -c <container-image>"
  exit 1
fi

if [ -z "${OUTPUT_FILE}" ]; then
  OUTPUT_FILE="${CONTAINER_IMAGE}.tar"
fi

readonly WORK_DIR="$(mktemp --directory)"
readonly ROOTFS_DIR="${WORK_DIR}/rootfs"
mkdir "${ROOTFS_DIR}"

echo "[INFO] Exporting the container image"
docker export "$(docker create "${CONTAINER_IMAGE}")" \
  | tar --directory="${ROOTFS_DIR}" --extract --file=-

# Extract the container CMD if it is present.
CONTAINER_CMD=$(docker \
             inspect \
             --format='{{json .Config.Cmd}}' \
             "${CONTAINER_IMAGE}")

if [ -z "${CONTAINER_CMD}" ]; then
  echo "[INFO] 'CMD' not found in container image. Searching for 'ENTRY_POINT'"
  CONTAINER_CMD=$(docker \
               inspect \
               --format='{{json .Config.Entrypoint}}' \
               "${CONTAINER_IMAGE}")
fi

if [ -z "${CONTAINER_CMD}" ]; then
  echo "[ERROR] Neither 'CMD' or 'ENTRYPOINT' found in container image."
  exit 1
fi

echo "[INFO] Container entry point cmd: \"${CONTAINER_CMD}\""

echo "[INFO] Creating config.json"
(
    cd "${WORK_DIR}"
    runc spec
    # Replace the entrypoint. Use a python script so as not to require a dependency on jq.
    python3 ${ABSOLUTE_SCRIPT_PATH}/replace_config_command.py ${CONTAINER_CMD} < config.json > new.json
    mv --force new.json config.json
    echo "[INFO] Contents of config.json after replacing process.args with CMD: $(cat config.json)"
)

echo "[INFO] Creating runtime bundle"

# By default /etc/hosts is empty. To enable name resolution, add a mapping for
# localhost.
echo '127.0.0.1 localhost' >> "${ROOTFS_DIR}/etc/hosts"

echo "[INFO] Contents of /etc/hosts: $(cat ${ROOTFS_DIR}/etc/hosts)"

tar --create --file="${OUTPUT_FILE}" --directory="${WORK_DIR}" ./rootfs ./config.json
