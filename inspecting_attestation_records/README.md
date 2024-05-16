# Inspecting Federated Compute attestation verification records

As described in the
[Confidential Federated Computations paper](https://arxiv.org/abs/2404.10764),
client devices running the
[Federated Compute](https://github.com/google-parfait/federated-compute) client
library and which participate in a `ConfidentialAggregations` protocol session
will verify attestation evidence for the ledger application hosted in this
repository. They will also verify the data access policy that the ledger will
enforce, which will specify one or more allowed data transformation applications
which are built from this repository.

After a successful verification, these devices will log an
[attestation verification record](https://github.com/google-parfait/federated-compute/blob/main/fcp/protos/confidentialcompute/verification_record.proto)
which can then be inspected. Please see
[fcp/client/attestation/README.md](https://github.com/google-parfait/federated-compute/blob/main/fcp/client/attestation/README.md)
in the Federated Compute repository for instructions on how to gather
attestation verification records from a device that is using the Federated
Compute client library.

## Inspecting attestation verification records

Attestation verification records consist of two major parts:

1.  *the attestation evidence that identifies the
    [ledger application](/ledger_enclave_app)* which generated the encryption
    key the client used to encrypt its data,

2.  *the [data access policy](/ledger_enclave_app#access-policies)* that
    prescribes the conditions under which the ledger binary will allow that
    encrypted data to be decrypted. A data access policy is effectively a graph
    describing allowed data transformations, where each transformation has to be
    performed by a TEE-hosted application. The set of allowable TEE-hosted
    applications for each transformation step are prescribed by a
    `ReferenceValues` proto. The ledger verifies the identity of the TEE-hosted
    data transformation using its attestation evidence. Only if this identity
    matches the `ReferenceValues`, does the ledger grant that data
    transformation access to the data.

Therefore, when inspecting attestation verification records there are two steps
to perform:

1.  Validating the ledger attestation verification, and determining the
    provenance of the binaries running in each layer (firmware, kernel,
    application) of the ledger TEE.

2.  Determining the provenances of the TEE-hosted data transformations to which
    the ledger will grant access to the encrypted data, as prescribed by the
    `ReferenceValues` data access policies.

To inspect the ledger attestation evidence and data access policy in an
`AttestationVerificationRecord`, the
[explain\_fcp\_attestation\_record](/tools/explain_fcp_attestation_record/)
tool in this repository can be used. This tool prints a human-readable summary
of the ledger attestation evidence and the data access policy. The attestation
evidence summary includes links to [SLSA
provenance](https://slsa.dev/spec/v0.1/provenance) stored on sigstore.dev for
each of the binaries at each layer of the TEE-hosted ledger application. The
data access policy summary does not currently include links to sigstore.dev for
the binary digests specified in each of the `ReferenceValues` protos in the
policy, but does include instructions for crafting such sigstore.dev links
manually.

The following is an example of the type of output the tool produces.

```sh
$ cd tools/explain_fcp_attestation_record
$ cargo run -- --record extracted_records/record_l2_to_l19_digest12345678.pb
Inspecting record at extracted_records/record_l2_to_l19_digest12345678.pb.

========================================
===== LEDGER ATTESTATION EVIDENCE ======
========================================

Oak Restricted Kernel Stack in a AMD SEV-SNP TEE

_____ Root Layer _____

... <snip> ...

_____ Application Layer _____

Binary [Digest]:
SHA2-256:892137def97d26c6b054093a5757919189878732ce4ab111212729007b30c0b4
Binary [Provenances]:
https://search.sigstore.dev/?hash=892137def97d26c6b054093a5757919189878732ce4ab111212729007b30c0b4

Note: this layer describes the "ledger" application binary, which is generally
a build of the `ledger_enclave_app` in the
https://github.com/google-parfait/confidential-federated-compute repository.

========================================
========== DATA ACCESS POLICY ==========
========================================

The data access policy allows 3 data transformations and defines 2 shared access
budgets.

>>>>> Transform #0 <<<<<
Source blob ID: 0
Destination blob ID: 1

Access budgets: the transform's access to its source blob is gated by *all* of
the following access rules:
- limited access budget (at most 2 times): the transform may only access its
  source blob this many times.
- limited shared access budget #1 (at most 2 times): this and other transforms
  sharing this same budget may only access their source blobs this many times
  combined.

Reference values for the Oak Restricted Kernel stack
oak_restricted_kernel:
  root_layer:
    amd_sev:
      min_tcb_version:
        boot_loader: 1
        tee: 2
        snp: 3
        microcode: 4
      stage0:

... <snip> ...
```

## Mapping binaries from an attestation verification record to their provenance

The binaries for the stage0 firmware, kernel, and container system image layers
are produced from the Oak repository. The container binaries, container
configurations, and restricted kernel application layer binaries are produced
from the code in this Confidential Federated Compute repository. We believe
that all binaries are reproducibly buildable.

All of the binaries attested in the ledger attestation evidence or allowed by a
`DataAccessPolicy` should have SLSA provenance entries on sigstore.dev. For any
allowed binary, you should be able to find the corresponding provenance using
the `https://search.sigstore.dev/?hash={THE_SHA256_HASH}` URL format, where
`{THE_SHA256_HASH}` is the SHA2-256 hash of the binary in the evidence/access
policy. These entries should show the binaries' provenance, including a link to
the Git commit on GitHub that the binaries were built from, as well as the
command that was used to build the binary, and which should allow you to
rebuild the same binary in a reproducible manner.

For example, below is an excerpt of the SLSA provenance record for the ledger
application binary listed in the example explanation output above
(https://search.sigstore.dev/?hash=892137def97d26c6b054093a5757919189878732ce4ab111212729007b30c0b4):

```
predicate:
  buildDefinition:
    buildType: https://slsa.dev/container-based-build/v0.1?draft
    externalParameters:
      source:
        uri: >-
          git+https://github.com/google-parfait/confidential-federated-compute@refs/heads/main
        digest:
          sha1: 20a4f3fc1f49943d03b76b264d3dc0ce90f83ade
      builderImage:
        uri: >-
          rust@sha256:4013eb0e2e5c7157d5f0f11d83594d8bad62238a86957f3d57e447a6a6bdf563
        digest:
          sha256: 4013eb0e2e5c7157d5f0f11d83594d8bad62238a86957f3d57e447a6a6bdf563
      configPath: buildconfigs/ledger_enclave_app.toml
      buildConfig:
        ArtifactPath: target/x86_64-unknown-none/release/ledger_enclave_app
        Command:
          - sh
          - '-c'
          - >-
            GITHUB_ACTION="provenance" scripts/setup_build_env.sh && cargo build
            --release --package ledger_enclave_app

... <snip> ...
```

It describes that the ledger application binary was produced at commit
20a4f3fc1f49943d03b76b264d3dc0ce90f83ade in the
https://github.com/google-parfait/confidential-federated-compute repository,
and it shows that the `GITHUB_ACTION="provenance" scripts/setup_build_env.sh &&
cargo build --release --package ledger_enclave_app` command was used to build
the binary.
