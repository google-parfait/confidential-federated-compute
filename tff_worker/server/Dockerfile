# Copyright 2023 The Confidential Federated Compute Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This Dockerfile builds a Docker image that runs a gRPC Python server
# implementing the Pipeline Transform service.

# Use a multi-stage build to reduce the final image size.
# We run Bazel to build the python server in the first stage, and then
# copy the build artifacts to the final stage.
# Bazel also has rules for creating Docker images, but rules_docker is
# in minimal maintenance mode, and the rules_oci python example
# includes some complicated workarounds for existing issues with bazel
# python support. Thus, prefer to avoid dependencies on either of these
# tools and instead use this Dockerfile to assemble the image.
FROM python:3.10.0
# Ensure all requirements needed by bazel are present.
RUN apt-get update && apt-get install -y apt-transport-https curl software-properties-common git gcc gnupg2 g++ default-jre python3-dev zip wget

# Use bazelisk to ensure this docker build process uses the same version of
# bazel as builds outside the container.
RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.17.0/bazelisk-linux-amd64 && \
    chmod 755 bazelisk-linux-amd64 && \
    mv bazelisk-linux-amd64 /usr/bin/bazelisk

# Copy the source code needed for building the Pipeline Transform Server.
COPY .bazelrc .bazelrc
COPY .bazelignore .bazelignore
COPY .bazelversion .bazelversion
COPY WORKSPACE WORKSPACE
COPY third_party/federated_compute/BUILD third_party/federated_compute/*.patch third_party/federated_compute/
COPY tff_worker/server/BUILD tff_worker/server/pipeline_transform_server.py tff_worker/server/

# Build the Python gRPC Pipeline Transform Server.
RUN bazelisk --output_base=/tmp/build_output build --build_python_zip tff_worker/server:pipeline_transform_server

# Final stage
# Use a base image that contains dependencies for running Python.
FROM python:3.10.0
# Copy the necessary build artifacts into the final stage.
COPY --from=0 /bazel-bin/tff_worker/server/pipeline_transform_server.zip /bazel-bin/tff_worker/server/pipeline_transform_server.zip
# Document that the server will be running on port 50051.
# The port still needs to be published and mapped when running the docker container.
EXPOSE 50051
# Configure the container to run the server on startup.
ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
CMD [ "python3 /bazel-bin/tff_worker/server/pipeline_transform_server.zip" ]
