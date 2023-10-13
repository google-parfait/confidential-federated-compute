# Tff Worker

This folder contains initial prototyping for building and running Docker images
that use TFF for use in Trusted Brella.

## Prerequisites

### Rootless Docker

Installing Docker is a prerequisite for the commands in this README.

Scripts in this repository that use Docker expect that the Docker daemon is
running as the local user instead of root.

In order to run Docker without root privileges, follow the guide at
https://docs.docker.com/engine/security/rootless/ .

Below is a quick summary of the relevant steps:

1.  If you have an existing version of Docker running as root, uninstall that
    first:

    ```bash
    sudo systemctl disable --now docker.service docker.socket
    sudo apt remove docker-ce docker-engine docker-runc docker-containerd
    ```

1.  Install `uidmap`:

    ```bash
    sudo apt install uidmap
    ```

1.  Add a range of subids for the current user:

    ```bash
    sudo usermod --add-subuids 500000-565535 --add-subgids 500000-565535 $USER
    ```

1.  Download the install script for rootless Docker, and run it as the current
    user:

    ```bash
    curl -fSSL https://get.docker.com/rootless > $HOME/rootless
    sh $HOME/rootless
    ```

1.  Add the generated environment variables to your shell:

    ```bash
    export PATH=$HOME/bin:$PATH
    export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
    ```

    **As an alternative** to setting the `DOCKER_HOST` environment variable, it
    is possible to instead run the following command to set the Docker context:

    ```bash
    docker context use rootless
    ```

    In either case, running the following command should show the current
    status:

    ```console
    $ docker context ls
    NAME        DESCRIPTION                               DOCKER ENDPOINT                       KUBERNETES ENDPOINT   ORCHESTRATOR
    default *   Current DOCKER_HOST based configuration   unix:///run/user/152101/docker.sock                         swarm
    rootless    Rootless mode                             unix:///run/user/152101/docker.sock
    Warning: DOCKER_HOST environment variable overrides the active context. To use a context, either set the global --context flag, or unset DOCKER_HOST environment variable.
    ```

    This should show either that the default context is selected and is using
    the user-local docker endpoint from the `DOCKER_HOST` variable, or that the
    `rootless` context is selected.

### Bazelisk

We use Bazelisk to ensure a consistent Bazel version and setup between local
development and continuous integration builds. Install bazelisk by following the
[instructions in the repo](https://github.com/bazelbuild/bazelisk#installation).

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
