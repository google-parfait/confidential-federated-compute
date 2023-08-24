# Tff Worker

This folder contains initial prototyping for building and running Docker images
that use TFF for use in Trusted Brella.

## Prerequisites

Installing Docker is a prerequisite for the commands in this README. Rootless
Docker is recommended for compatibility with Oak. See
[Oak's Rootless Docker](https://github.com/project-oak/oak/blob/main/docs/development.md#rootless-docker)
installation instructions.

## Regenerate requirements.txt for the TFF Pipeline Transform Server

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

## TFF Worker Pipeline Transform Server

The TFF Worker Pipeline Transform server will be a gRPC server running on Oak
Containers and implementing the pipeline_transform API so that the untrusted
application can instruct it to perform transformations on data using TFF.

For now, we can run a Python gRPC server as a docker container with a port
exposed to the host. For testing purposes, we run a C++ client on the host.
Using a C++ client makes it easier to build within a docker container but
produce a binary that can be run on the host, so that people who want to develop
using this repo don't need to have a particular version of bazel installed on
their host.

The following commands should all be executed from the root of the
confidential-federated-compute repository.

We use bazel from within a Docker container to build the C++ client via the
following steps:

1.  Install [bazelisk](https://github.com/bazelbuild/bazelisk#installation).

2.  Build the Pipeline Transform client:

    ```
    bazelisk build //tff_worker/client:pipeline_transform_cpp_client
    ```

    The C++ client should now be runnable from the host.

    ```
    bazel-bin/tff_worker/client/pipeline_transform_cpp_client
    ```

    Since there is no server running, the C++ client is expected to produce
    output like the following:

    ```
    Starting RPC Client for Pipeline Transform Server.
    RPC failed: 14: failed to connect to all addresses; last error: UNKNOWN: ipv4:127.0.0.1:50051: Failed to connect to remote host: Connection refused
    ```

3.  Build the Docker image that will run the Python server. The '.' argument
    specifies the context that is available to access files that are used when
    building the Docker image. Since the build process uses Bazel, the context
    needs to include the workspace root, so it is important that this command is
    run from the root of the confidential-federated-compute repo. Building the
    image may take a while the first time it runs, but on subsequent runs parts
    of the image building process will be cached.

    ```
    docker build -f tff_worker/server/Dockerfile -t pipeline_transform_server .
    ```

    Once the image has successfully built, you can run the server in the docker
    container, publishing and mapping the gRPC server port so it can be accessed
    from localhost:

    ```
    docker run -i -p 127.0.0.1:50051:50051 pipeline_transform_server:latest &
    ```

4.  Now the server should be running as a background job, so you can try running
    the C++ client again:

    ```
    bazel-bin/tff_worker/client/pipeline_transform_cpp_client
    ```

    This time, it should produce the following output:

    ```
    Starting RPC Client for Pipeline Transform Server.
    RPC failed: 12: Transform not implemented!
    ```

5.  To bring the docker process back to the foreground in order to quit the
    server, use the `fg` command.


## Debugging common issues

If you are running low on disk space you should prune dangling images and
containers:

```
docker image prune
```

```
docker container prune
```
