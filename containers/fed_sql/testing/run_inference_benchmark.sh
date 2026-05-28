#!/bin/bash
# Copyright 2025 Google LLC
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

set -e

MODEL_DIR="/tmp/gemma_benchmark_models"
VARIATION="3.0-4b-it-sfp"
TOKENIZER_FILE="tokenizer.spm"
WEIGHTS_FILE="4.0-it-it-sfp.sbs"

# Print usage instructions
usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --variation <name>    Gemma model variation (default: 3.0-4b-it-sfp)"
  echo "  --model_dir <path>    Directory to save downloaded models (default: /tmp/gemma_benchmark_models)"
  echo "  --help                Show this help message"
  exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --variation)
      VARIATION="$2"
      shift 2
      ;;
    --model_dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# Validate model variation file names
if [[ "$VARIATION" == "2.0-2b-it-sfp" ]]; then
  WEIGHTS_FILE="2.0-2b-it-sfp.sbs"
elif [[ "$VARIATION" == "3.0-4b-it-sfp" ]]; then
  WEIGHTS_FILE="4.0-it-it-sfp.sbs"
fi

mkdir -p "$MODEL_DIR"

# Check if Kaggle CLI is installed
if ! command -v kaggle &> /dev/null; then
  echo "Error: 'kaggle' CLI tool is not installed."
  echo "Please install it using: pip install kaggle"
  echo "And ensure you have your Kaggle credentials stored in ~/.kaggle/kaggle.json."
  exit 1
fi

# Download tokenizer file if missing
if [ ! -f "$MODEL_DIR/$TOKENIZER_FILE" ]; then
  echo "Downloading $TOKENIZER_FILE from Kaggle..."
  kaggle models instances files download google/gemma-3/gemmaCpp/"$VARIATION" \
      --file_name="$TOKENIZER_FILE" --path="$MODEL_DIR"
fi

# Download model weights file if missing
if [ ! -f "$MODEL_DIR/$WEIGHTS_FILE" ]; then
  echo "Downloading $WEIGHTS_FILE from Kaggle..."
  kaggle models instances files download google/gemma-3/gemmaCpp/"$VARIATION" \
      --file_name="$WEIGHTS_FILE" --path="$MODEL_DIR"
fi

echo "Gemma model and tokenizer are successfully staged in $MODEL_DIR."
echo "Building and running the inference benchmark..."

# Build and execute the Bazel benchmark, passing in the download directory paths
bazelisk run -c opt //containers/fed_sql:inference_model_bm -- \
    --tokenizer_path="$MODEL_DIR/$TOKENIZER_FILE" \
    --model_path="$MODEL_DIR/$WEIGHTS_FILE"
