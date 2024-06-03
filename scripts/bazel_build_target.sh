#!/bin/bash
#
# Copyright 2024 Google LLC.
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
# Builds and copies one or more release targets. At least one target as well
# as --output_dir are required.
set -ex

# If bazelisk isn't in user's path, the BAZELISK environment variable may be set
# instead. This may also be used to pass startup options like --nosystem_rc to
# bazel; this usage requires us to not quote ${BAZELISK} when used later.
readonly BAZELISK="${BAZELISK:-bazelisk}"

# List of available bazel targets, along with the place where the artifact
# will appear. Positional arguments are required to be in the key set.
declare -Ar ARTIFACTS=(
  [//containers/sql_server:oci_runtime_bundle.tar]=sql_server/container.tar
  [//containers/test_concat:oci_runtime_bundle.tar]=test_concat/container.tar
  [//containers/agg_core:oci_runtime_bundle.tar]=agg_core/container.tar
)

declare -a targets

while [[ $# -gt 0 ]]; do
  case $1 in
    -o|--output_dir)
      output_dir="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      targets+=("$1")
      shift # past argument
      ;;
  esac
done

[ -z "${output_dir}" ] && exit 1
[ ${#targets[@]} -eq 0 ] && exit 1

${BAZELISK} build -c opt "${targets[@]}"

bin_dir="$(${BAZELISK} info -c opt bazel-bin)"
for target in "${targets[@]}"; do
  src="${bin_dir}${target/:/\//}"
  dst="${output_dir}/${ARTIFACTS[$target]}"
  mkdir --parents "$(dirname "${dst}")"
  cp --preserve=timestamps --force "${src}" "${dst}"
done
