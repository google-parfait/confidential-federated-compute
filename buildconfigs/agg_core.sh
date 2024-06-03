#!/bin/bash
#
# Build configuration for agg_core.
#
export PACKAGE_NAME=agg_core

export BUILD_COMMAND=(
  scripts/bazel_build_target.sh
  --output_dir
  binaries
  //containers/agg_core:oci_runtime_bundle.tar
)

export SUBJECT_PATHS=(
  binaries/agg_core/container.tar
)
