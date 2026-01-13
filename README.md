# Confidential Federated Compute

The Confidential Federated Compute project enables Federated Learning and
Analytics using
[Confidential Computing](https://en.wikipedia.org/wiki/Confidential_computing).
This repository holds publicly verifiable components that run within
[Trusted Execution Environments (TEEs)](https://en.wikipedia.org/wiki/Trusted_execution_environment)
and interact with user data. In order to run the components in this repository
on TEEs, this repository depends on the
[Project Oak](https://github.com/project-oak/oak) platform. In the Project Oak
terminology, this repository contains Trusted Applications that run on the Oak
Infrastructure.

## Design and Code Structure

The goal of this project is to give end users meaningful control over how their
data can be used when the data is uploaded to a central server. Trusted
Execution Environments allow proving what is running on a central server, and
our contribution is to design a distributed system that allows us to make claims
about a system of TEEs, some of which may not begin execution until after the
data has been uploaded, to create a "chain of trust" from the user that uploads
data to the TEEs that subsequently process the data.

Data that is uploaded is encrypted and bound to a policy stating how the data
can be used. This policy is a directed graph of allowed transformations on the
data. The data will only be allowed to be decrypted by components that prove
that the code they are running is allowed by the policy, by using remote
attestation with a hardware root-of-trust to determine what binary is running.

### Key Management Service

The component responsible for enforcing the policy on the data is called the
Confidential Federated Compute Key Management Service (CFC KMS). When data is
uploaded, it is initially only decryptable by the KMS, which replicates and
stores the corresponding decryption keys in-memory. This means that if the KMS
restarts, access to the uploaded data is lost forever. Other, short lived
components running in Trusted Execution Environments can provide attestations to
the KMS and request access to these keys, which are only shared if the
attestation matches the policy under which the data was uploaded. The KMS also
maintains rollback-protected state for each pipeline for tracking privacy
budgets.

The code for the KMS is located in the [`containers/kms`](containers/kms)
directory; see this directory for additional documentation. The KMS runs in a
TEE using
[Oak Containers](https://github.com/project-oak/oak/tree/main/oak_containers).

#### Ledger

The ledger was the KMS's predecessor and played a similar role. The main
difference was that the ledger tracked privacy budgets itself on a per-upload
basis instead of delegating budget tracking to each pipeline. Unfortunately,
this required ledger operations and state to scale with the amount of uploaded
data, making the ledger a bottleneck for large-scale data processing.

The ledger code has been removed, but it can still be found in the project
history
([binary](https://github.com/google-parfait/confidential-federated-compute/tree/5fe8b04f68628fc605d400adccab8169011267ca/replicated_ledger_enclave_app),
[README](https://github.com/google-parfait/confidential-federated-compute/blob/5fe8b04f68628fc605d400adccab8169011267ca/ledger_enclave_app/README.md)).

### Transformations

This repository also contains code for components that run transformations over
data within TEEs, if those transformations are allowed by the KMS-enforced
policy. Transforms are implemented using
[Oak Containers](https://github.com/project-oak/oak/tree/main/oak_containers).

*   [**`containers/confidential_transform_test_concat`**] Example transform that
    concatenates its inputs.
*   [**`containers/fed_sql`**] Transform that aggregates aggregate using both
    SQLite and Aggregation Cores.

See each transform's README for more details.

## Inspecting attestation verification records and endorsement transparency log entries

See [docs/README.md](docs/README.md) for instructions for mapping attestation
verification records logged by
[Federated Compute](https://github.com/google-parfait/federated-compute)
clients, as well as transparency log entries for KMS and data access policy
endorsements to the reproducibly buildable binaries in this repository.

## Building

The following section provides instructions for building artifacts that can be
run within Trusted Execution Environments from the source code in this repo.

The eventual goal is to achieve binary transparency; that is, we would like to
verifiably link a binary with the source code that produced the binary. This
way, when a Trusted Execution Environment remotely attests that it is running a
particular binary, anyone can verify that the binary was produced from a
particular version of the source code, and thus be convinced that the Trusted
Execution Environment is in fact running a particular application.

There are different strategies to achieve binary transparency. One strategy
consists of the use of a trusted builder along with provenance and endorsement
statements that are signed and published in a transparency log, as described in
detail in Oak's
[Transparent Release documentation](https://github.com/project-oak/transparent-release#release-transparency).
Another strategy is making the build process fully reproducible, so that given a
particular version of the source code, bitwise identical artifacts will be
produced by the build process regardless of where or when the build process
runs. This allows an external auditor to run the build process themselves in
order to verify that the source code at a particular version produces the
specified binary.

We provide the following instructions for building the code. As we make progress
toward binary transparency, we will refine these instructions to provide details
on how one can verify that a particular version of the source code produces a
particular binary.

### Prerequisites

#### Bazelisk

Bazelisk is the bazel-recommended way to obtain a specific bazel version. See
https://github.com/bazelbuild/bazelisk#installation for installation
instructions.

The build scripts in this repository require either bazelisk to be in your
`PATH` or the `BAZELISK` environment variable to be set to the location of the
bazelisk binary.

### Building Artifacts

Clone this repository to your developer machine. The following commands should
all be run from within the repository root.

To specify a directory where the build artifacts will be output, run the
following command. Consider adding the line below to your ~/.bashrc or ~/.zshrc
so you don't have to run this step every time you enter a new shell.

```
export BINARY_OUTPUTS_DIR=/tmp/confidential-federated-compute/binaries
```

To build all artifacts which can be run in enclaves, run the following command:

```
./scripts/build.sh release
```

After the above command completes, use the following command to list the output
binaries:

```
ls "${BINARY_OUTPUTS_DIR}"
```

You should see a subdirectory for each server, including the following:

```
confidential_transform_test_concat  fed_sql  kms
```

These correspond to the components described in
[Design and Code Structure](#design-and-code-structure).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## License

Apache 2.0; see [`LICENSE`](LICENSE) for details.

## Disclaimer

This is not an officially supported Google product.
