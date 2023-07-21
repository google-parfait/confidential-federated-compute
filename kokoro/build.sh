#!/bin/bash

set -e

cd $(dirname "$0")/..

cargo build && cargo test

if [ "$1" == "release" ]; then
  cargo build --release \
          -p ledger_enclave_app \
          -p square_enclave_app \
          -p sum_enclave_app

  mkdir -p "${KOKORO_ARTIFACTS_DIR}/binaries"
  cp -v \
          target/x86_64-unknown-none/release/ledger_enclave_app \
          target/x86_64-unknown-none/release/square_enclave_app \
          target/x86_64-unknown-none/release/sum_enclave_app \
          "${KOKORO_ARTIFACTS_DIR}/binaries/"
fi
