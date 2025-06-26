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

# Dockerfile for the build environment in order to match the target
# system image and reduce compilation issues. Hermetic build would
# be preferred, but currently does not work.
#
# Build steps:
#   cd confidential-federated-compute/containers/minimum_rules_cuda 
#   docker build -t mvp . -f Dockerfile
#   docker run -it -v $(pwd):/workspace -w /workspace mvp:latest
#   bazelisk build ...
ARG cuda_version=12.9.0-cudnn-devel-ubuntu22.04 
ARG GO_VERSION=1.23.0
FROM nvidia/cuda:${cuda_version}

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts
ARG DEBIAN_FRONTEND=noninteractive

# Install weget to download go, also install libtinfo5 which is required by clang.
RUN apt-get --yes update \
    && apt-get install --yes --no-install-recommends wget libtinfo5

# Install specified Go version
RUN wget -q https://go.dev/dl/go1.23.0.linux-amd64.tar.gz \
    && tar -C /usr/local -xzf go1.23.0.linux-amd64.tar.gz \
    && rm go1.23.0.linux-amd64.tar.gz

# Set Go environment variables globally for subsequent RUN, CMD, ENTRYPOINT
ENV PATH="/usr/local/go/bin:${PATH}"
ENV GOPATH="/go"
ENV PATH="${GOPATH}/bin:${PATH}"
RUN go install github.com/bazelbuild/bazelisk@latest

# Instruct the cuda toolchain to use clang instead of nvcc.
ENV CUDA_COMPILER_USE_CC_TOOLCHAIN=true
# Tells the cuda toolchain where to find the clang compiler.
ENV CUDA_CLANG_LABEL="@llvm_toolchain//:cc-toolchain-x86_64-linux"
