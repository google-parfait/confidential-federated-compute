# CFC Registry

This directory defined an **experimental**
[Bazel registry](https://bazel.build/external/registry) to allow projects using
bzlmod to more easily build `ConfidentialTransform` services using the libraries
in this repository.

Since this repository now uses bzlmod, this registry is deprecated and new
versions will not be added. Instead, depend on confidential-federated-compute
directly using `archive_override`.
