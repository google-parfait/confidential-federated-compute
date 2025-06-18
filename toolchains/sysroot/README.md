# CFC Sysroot

This directory contains a small bazel workspace for building the reproducible
clang sysroot used elsewhere in this repository. (A sysroot contains system
libraries, such as libc, that are not part of C++ toolchain by default.)

This workspace uses
[`rules_distroless`](https://github.com/GoogleContainerTools/rules_distroless)
to build the sysroot tarball reproducibly from Debian packages. These packages
are downloaded from the [Debian snapshot archive](https://snapshot.debian.org/)
to ensure that builds can be reproduced even after the packages have been
removed from the current Debian snapshot.

## Updating the Sysroot

The sysroot is updated manually on an as-needed basis. To update the sysroot,
perform the following steps:

1.  Update `sysroot.yaml` to include new packages or a newer snapshot. Run
    `bazelisk run @sysroot//:lock` to update `sysroot.lock.json`. After
    confirming the sysroot builds successfully (`bazelisk build //:sysroot`),
    commit this change.
2.  Create a new tag (and optionally release) starting with `sysroot-` (e.g.
    `sysroot-YYYYMMDD`). This will trigger the "Release sysroot" workflow to
    build the sysroot and attach it to the release. This workflow will also
    upload build provenance to Sigstore.
3.  To help detect non-determinism, verify that the checksum for the
    locally-built sysroot matches what was built by the workflow.
4.  Update the version of the sysroot used in CFC to point to the new release.
    Prefer `sha256` over `integrity` since the former can be used to easily look
    up the build provenance in Sigstore.
