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

# Install Deps
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y python3 python3-pip python-is-python3 clang-format python3-venv cloc \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/library-scripts
