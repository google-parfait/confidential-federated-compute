FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu-22.04

# Install Bazel
ARG BAZELISK_VERSION=v1.25.0
ARG BAZELISK_DOWNLOAD_SHA=dev-mode
RUN curl -fSsL -o /usr/local/bin/bazelisk https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-amd64 \
    && ([ "${BAZELISK_DOWNLOAD_SHA}" = "dev-mode" ] || (sha256sum /usr/local/bin/bazelisk && echo "${BAZELISK_DOWNLOAD_SHA} */usr/local/bin/bazelisk" | sha256sum --check - )) \
    && chmod 0755 /usr/local/bin/bazelisk \
    && sudo ln -s /usr/local/bin/bazelisk /usr/local/bin/bazel \
    && curl -fsSL https://raw.githubusercontent.com/bazelbuild/bazel/$(bazel --version | awk '{print $2}')/scripts/bazel-complete-header.bash >> /home/vscode/.bashrc \
    && curl -fsSL https://raw.githubusercontent.com/bazelbuild/bazel/$(bazel --version | awk '{print $2}')/scripts/bazel-complete-template.bash >> /home/vscode/.bashrc \
    && bazel help completion >> /home/vscode/.bashrc

# Install Buildifier
ARG BUILDIFIER_VERSION=v8.5.1
ARG BUILDIFIER_DOWNLOAD_SHA=dev-mode
RUN curl -fSsL -o /usr/local/bin/buildifier https://github.com/bazelbuild/buildtools/releases/download/${BUILDIFIER_VERSION}/buildifier-linux-amd64 \
    && ([ "${BUILDIFIER_DOWNLOAD_SHA}" = "dev-mode" ] || (sha256sum /usr/local/bin/buildifier && echo "${BUILDIFIER_DOWNLOAD_SHA} */usr/local/bin/buildifier" | sha256sum --check - )) \
    && chmod 0755 /usr/local/bin/buildifier

# Install Keep-Sorted
ARG KEEP_SORTED_VERSION=v0.9.0
RUN curl -fSsL -o /usr/local/bin/keep-sorted https://github.com/google/keep-sorted/releases/download/${KEEP_SORTED_VERSION}/keep-sorted_linux \
    && chmod 0755 /usr/local/bin/keep-sorted


# Install Deps
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y python3 python3-pip python-is-python3 clang-format python3-venv cloc \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/library-scripts
