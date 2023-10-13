# Builds a docker container with dependencies necessary for development of
# parts of the confidential-federated-compute project that use Bazel.
FROM python:3.10.0

# Install docker CLI.
RUN curl --fail --silent --show-error --location https://download.docker.com/linux/debian/gpg | apt-key add -
RUN echo "deb [arch=amd64] https://download.docker.com/linux/debian bullseye stable"  > /etc/apt/sources.list.d/backports.list

# Get necessary dependecies.
# We're rate-limiting HTTP requests to 500 kB/s as otherwise we may get timeout errors
# when downloading from snapshot.debian.org.
RUN apt-get --yes update \
  && apt-get install --no-install-recommends --yes --option Acquire::http::Dl-Limit=500 \
  clang \
  docker-ce-cli \
  git \
  python3-dev \
  pip \
  python3-pip \
  python-clang \
  runc \
  wget \
  zip

# Bazelisk is the recommended way to download Bazel.
RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.17.0/bazelisk-linux-amd64 && \
    chmod 755 bazelisk-linux-amd64 && \
    mv bazelisk-linux-amd64 /usr/bin/bazelisk
