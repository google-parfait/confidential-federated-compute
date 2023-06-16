# Tff Worker

This folder contains initial prototyping for building and running Docker images
that use TFF for use in Trusted Brella.

## Instructions

Installing Docker is a prerequisite for the commands in this README. Rootless
Docker is recommended for compatibility with Oak. See
[Oak's Rootless Docker](https://github.com/project-oak/oak/blob/main/docs/development.md#rootless-docker)
installation instructions.

Run from within this directory, the following command builds a Docker image
containing TFF:

```
docker build . -t tff
```

You can then execute the hello_world.py script in the Docker container:

```
docker run tff:latest python ./app/hello_world.py
```

If you are running low on disk space you should prune dangling images and
containers:

```
docker image prune
```

```
docker container prune
```

# Regenerate requirements.txt for the tff_worker

To ensure safe downloads in the case that an attacker compromises the PyPI
account of a library we depend on, we require hashes for all packages installed
by Pip. We use requirements.txt to specify dependencies needed by the docker
image along with their hashes.

To regenerate requirements.txt, run the gen_requirements.sh shell script. Note
that it is imperative that the resulting requirements.txt is checked in; if
generating the requirements were part of the docker build process of the
tff_worker, we wouldn't get the security benefits of using hashes.
