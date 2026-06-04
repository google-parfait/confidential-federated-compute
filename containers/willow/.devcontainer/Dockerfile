FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu-22.04

# Install Bazel
ARG BAZEL_VERSION=8.2.1
ARG BAZEL_COMPLETE_HEADER_SHA=1b9386d3c7cb348bd9981df62cf569cfc11ca891fdb6a1f1ab10257480e2d651
ARG BAZEL_COMPLETE_TEMPLATE_SHA=e37e50fabd701ec80ddbe2ed304e44656858cb6df9b0004dc7b805b2dd4959f9
ARG BAZELISK_VERSION=v1.25.0
ARG BAZELISK_DOWNLOAD_SHA=dev-mode
RUN curl -fSsL -o /usr/local/bin/bazelisk https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-amd64 \
    && ([ "${BAZELISK_DOWNLOAD_SHA}" = "dev-mode" ] || (sha256sum /usr/local/bin/bazelisk && echo "${BAZELISK_DOWNLOAD_SHA} */usr/local/bin/bazelisk" | sha256sum --check - )) \
    && chmod 0755 /usr/local/bin/bazelisk \
    && sudo ln -s /usr/local/bin/bazelisk /usr/local/bin/bazel \
    && curl -fsSL -o /tmp/bazel-complete-header.bash https://raw.githubusercontent.com/bazelbuild/bazel/${BAZEL_VERSION}/scripts/bazel-complete-header.bash \
    && echo "${BAZEL_COMPLETE_HEADER_SHA} */tmp/bazel-complete-header.bash" | sha256sum --check - \
    && cat /tmp/bazel-complete-header.bash >> /home/vscode/.bashrc \
    && curl -fsSL -o /tmp/bazel-complete-template.bash https://raw.githubusercontent.com/bazelbuild/bazel/${BAZEL_VERSION}/scripts/bazel-complete-template.bash \
    && echo "${BAZEL_COMPLETE_TEMPLATE_SHA} */tmp/bazel-complete-template.bash" | sha256sum --check - \
    && cat /tmp/bazel-complete-template.bash >> /home/vscode/.bashrc \
    && export USE_BAZEL_VERSION=${BAZEL_VERSION} \
    && bazel help completion >> /home/vscode/.bashrc \
    && rm /tmp/bazel-complete-header.bash /tmp/bazel-complete-template.bash

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
