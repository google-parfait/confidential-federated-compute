# Copyright 2023 Google LLC.
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

# This Dockerfile builds a Docker image that runs a gRPC Python server
# implementing the Pipeline Transform service.
#
# Rather than building this docker container directly, use
# `scripts/bazel_build.sh release` to build and test the correct Bazel artifacts
# and also build the container.
#
# Use a base image that contains dependencies for running Python.
FROM python:3.10.0
# Copy the necessary build artifacts into the final stage.
COPY ./pipeline_transform_server.zip pipeline_transform_server.zip
# Document that the server can run on port 8080
# The port still needs to be published and mapped when running the docker container.
EXPOSE 8080
# Configure the container to run the server on startup.
CMD [ "python3", "pipeline_transform_server.zip" ]
