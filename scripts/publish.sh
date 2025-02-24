#!/bin/bash
#
# Copyright 2025 Google LLC.
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
# Writes a GitHub attestation to discoverability storage, including updating
# the search index. Needs `gcloud` and appropriate permissions on the buckets.
#
set -o errexit
set -o nounset
set -o pipefail

readonly FBUCKET=oak-files  # File bucket on GCS
readonly IBUCKET=oak-index  # Index bucket on GCS

# ID of index to access GitHub attestations by commit hash and package name.
readonly PROV_FOR_COMMIT=7

usage_and_exit() {
  >&2 echo "Usage: $0 <provenance_path> sha1:<commit_hash> <package_name>"
  exit 1
}

copy_file() {
  # Don't clobber so create permission on the destination is sufficient.
  gcloud storage cp --no-clobber "$1" "$2"
}

# Uploads a file and returns its SHA2_256 hash.
upload_file() {
  file="$1"
  hash="sha2-256:$(sha256sum "${file}" | cut -d " " -f 1)"
  copy_file "${file}" "gs://${FBUCKET}/${hash}"
  echo "${hash}"
}

publish() {
  provenance_path="$1"
  commit_hash="$2"
  package_name="$3"

  phash_path=$(mktemp)
  upload_file "${provenance_path}" > "${phash_path}"
  copy_file \
    "${phash_path}" \
    "gs://${IBUCKET}/${PROV_FOR_COMMIT}/${commit_hash}/${package_name}"
}

set -o xtrace
if [[ $# != 3 ]]; then
  usage_and_exit
fi
publish "$1" "$2" "$3"
