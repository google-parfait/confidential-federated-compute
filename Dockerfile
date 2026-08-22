# Copyright 2026 Google LLC.
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

# # A Dockerfile for debugging hermeticity / RBE-compatibility issues.
#
# This file provides a convenient way to create an environment that fairly
# closely mimics the RBE images our builds run on (which have very few
# additional packages installed and therefore often fail when builds aren't
# fully hermetic).
#
# This is only meant as a tool to aid development/debugging of build issues, we
# don't want to rely on Docker containers for build hermeticity/reproducibility,
# and our builds should therefore be made to work both outside and inside this
# type of container.
#
# How to use:
#
#   # Build the container once. (Rebuild whenever this updated.)
#   podman build -t cfc-remote-env .
#
#   # Run the image. This will spin up a terminal in which you can invoke
#   # bazelisk or # scripts/build.sh. You can edit files in your main workspace
#   # and re-run builds easily and interactively.
#   #
#   # The .cache/cfcremote mount ensures that repeated invocations inside the
#   # container share the same bazel cache, which speeds repeated builds up
#   # considerably.
#   podman run --userns=keep-id:uid=1000,gid=1000 \
#     -v ~/.cache/cfcremote:/home/developer/.cache \
#     -v "$PWD:/workspace" -t -i cfc-remote-env

# Build stage to download Bazelisk
ARG BAZELISK_VERSION=v1.29.0

FROM docker.io/curlimages/curl:latest AS downloader
ARG BAZELISK_VERSION
RUN curl -fsSL -o /tmp/bazelisk "https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-amd64" && \
    chmod 0755 /tmp/bazelisk

# This hash should be kept in sync with the hash in the //:remote_platform
# target in our BUILD files.
FROM gcr.io/cloud-marketplace/google/debian12@sha256:be15f84c44c4ee68b4f187128f0278df1b0f424c04fb5f08b098967d5b896388

# Update CA certificates and install git (needed for `git_repository` bazel
# deps).
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates git && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Non-root user configuration
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd -g ${USER_GID} ${USERNAME} && \
    useradd -u ${USER_UID} -g ${USER_GID} -m -s /bin/bash ${USERNAME}

# Install Bazelisk and create bazel symlink
COPY --from=downloader /tmp/bazelisk /usr/local/bin/bazelisk
RUN ln -s /usr/local/bin/bazelisk /usr/local/bin/bazel

USER ${USERNAME}
WORKDIR /workspace

CMD ["/bin/bash"]
