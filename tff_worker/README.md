# Tff Worker

This folder contains initial prototyping for building and running Docker images
that use TFF for use in Trusted Brella.

## Prerequisites

See the main repo [README](../README.md#prerequisites) for the prerequisites required for building the
container.

## Developing the TFF Worker Pipeline Transform Server

The TFF Worker Pipeline Transform server is a gRPC server running on Oak
Containers and implementing the PipelineTransform API so that the untrusted
application can instruct it to perform transformations on data using TFF.

Note: The following commands should all be executed from the root of the
confidential-federated-compute repository.

### Building for fast iteration during development

To build and test the code while you are actively making changes, you can start
up a shell in a Docker container that has all the necessary build dependencies
by running the following command:

```
./scripts/docker_run.sh bash
```

This may take a while the first time since it has to build the Docker container
with the necessary build dependencies but should be faster on future iterations.

Within this shell you can build and test the code using Bazelisk commands. For
example, to run all Bazel tests in the repository you can try:

```
bazelisk test ...
```

We use Bazelisk rather than directly using Bazel, as use of Bazelisk is the
recommended way to install Bazel according to the
[installation instructions](https://bazel.build/install).

The Bazel build artifacts will be cached within the `.bazel_cache` folder so
that they last beyond the lifetime of the Docker container which improves build
times.

### Building for use with Oak Containers

Once you are ready to test your local version of the Python Pipeline
Transform server as part of the larger system, you will need to build the Docker
image that will run the server and package it as an OCI runtime bundle. To
ensure the resulting OCI runtime bundle is copied to a desired directory, ensure
the `KOKORO_ARTIFACTS_DIR` environment variable is set. The following command
should print a directory:

```
echo "${KOKORO_ARTIFACTS_DIR}"
```

Now run the release build to create the Docker container, package it as an OCI
runtime bundle, and copy it to `KOKORO_ARTIFACTS_DIR`.

```
./scripts/docker_run.sh ./scripts/bazel_build.sh release
```

### Debugging common issues

If you are running low on disk space you should prune dangling images and
containers:

```
docker image prune
```

```
docker container prune
```

## Regenerating requirements.txt for the TFF Pipeline Transform Server

To ensure safe downloads in the case that an attacker compromises the PyPI
account of a library we depend on, we require hashes for all packages installed
by Pip. We use requirements.txt to specify dependencies needed by the docker
image along with their hashes.

To regenerate requirements.txt from requirements.in, run the following command:

```
bazelisk run tff_worker:requirements.update
```

Note that it is imperative that the resulting requirements.txt is checked in; if
generating the requirements were part of the docker build process of the
tff_worker, we wouldn't get the security benefits of using hashes.
