#!/bin/bash
#
# Build configuration for fed_sql.
#
export PACKAGE_NAME=fed_sql

export BUILD_COMMAND=(
  scripts/build_target.sh
  --output_dir
  binaries
  //containers/fed_sql:oci_runtime_bundle.tar
)

export SUBJECT_PATHS=(
  binaries/fed_sql/container.tar
)
