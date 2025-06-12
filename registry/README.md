# CFC Registry

This directory defines an **experimental**
[Bazel registry](https://bazel.build/external/registry) to allow projects using
bzlmod to more easily build `ConfidentialTransform` services using the libraries
in this repository.

To use it, use the `--registry` flag as follows: `bazelisk
--registry=https://bcr.bazel.build
--registry=https://raw.githubusercontent.com/google-parfait/confidential-federated-compute/main/registry/
...`

## Updating the Registry

Once a module version has been added to the registry, it must not be modified.
Instead, add a new version. Tips:

1.  Integrity hashes can be generated using the following command: `echo
    sha256-$(sha256sum -b $FILE | cut -d' ' -f1 | xxd -p -r | base64)`

2.  When modifying a not-yet-released module version (e.g. as part of the
    release process), it may be necessary to remove the lock file and restart
    bazel in order to get bazel to pick up changes: `bazelisk shutdown && rm -f
    MODULE.bazel.lock`
