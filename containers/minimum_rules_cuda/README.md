# Minimum container that supports cuda

This container is a reference container for exploration or testing. 
It doesn't build in a fully hermetic way today, and all the bazel
commands require to run inside a docker container specified by the Dockerfile.

To build the container, follow the steps below:
1. `cd confidential-federated-compute/containers/minimum_rules_cuda`
2. `docker build -t mvp . -f Dockerfile`
3. `docker run -it -v $(pwd):/workspace -w /workspace mvp:latest`
4. `bazelisk build ...`

*Temporarily*, to extract the build artifacts out of the docker environment,
run `cp -RL bazel-bin/YOUR_TARGET .`.This command will copy the artifact to your local
minimum_rules_cuda directory. In the future, we'll bootstrap the steps with docker buildx
 or get rid of the docker environment.
