#!/bin/bash
#
# Build configuration for ledger_enclave_app.
#
export PACKAGE_NAME=ledger

export BUILD_COMMAND=(
  scripts/cargo_build_target.sh
  --output_dir
  binaries
  ledger_enclave_app
)

export SUBJECT_PATHS=(
  binaries/ledger/binary
)
