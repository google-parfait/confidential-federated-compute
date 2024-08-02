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
$ bazelisk run //tools/explain_fcp_attestation_record:main -- --record=$PWD/extracted_records/record_l2_to_l19_digest12345678.pb
Inspecting record at extracted_records/record_l2_to_l19_digest12345678.pb.

========================================
===== LEDGER ATTESTATION EVIDENCE ======
========================================

Oak Restricted Kernel Stack in a AMD SEV-SNP TEE

_____ Root Layer _____

... <snip> ...

_____ Application Layer _____

binary:
  sha2_256: 5d10d8013345814e07141c6a4c9297d37653239132749574a2a71483c413e9fe
config: {}


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

Application matcher for this transform:
- Tag: app2
- ...<snip>...
- Applications performing this transform must provide attestation evidence that
  can be verified with the following reference values:

Reference values for the Oak Restricted Kernel stack
_____ Root Layer _____

The attestation must be rooted in an AMD SEV-SNP TEE.

The reference values describing this layer are printed below.

amd_sev:
... <snip>...

_____ Kernel Layer _____

The reference values describing this layer are printed below.

acpi:
... <snip>...

_____ Application Layer _____

binary:
  skip: {}
configuration:
  skip: {}
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
workflow that was used to build the binary, and which should allow you to
rebuild the same binary in a reproducible manner.

For example, below is an excerpt of the SLSA provenance record for the ledger
application binary listed in the example explanation output above
(https://search.sigstore.dev/?hash=5d10d8013345814e07141c6a4c9297d37653239132749574a2a71483c413e9fe):

```
GitHub Workflow SHA: 0f8072c8e9dda36170f0fa466305e9664716fb56
GitHub Workflow Name: Build and attest all
GitHub Workflow Repository: google-parfait/confidential-federated-compute
GitHub Workflow Ref: refs/heads/main
OIDC Issuer (v2): https://token.actions.githubusercontent.com
Build Signer URI: https://github.com/google-parfait/confidential-federated-compute/.github/workflows/build.yaml@refs/heads/main
Build Signer Digest: 0f8072c8e9dda36170f0fa466305e9664716fb56
Runner Environment: github-hosted
Source Repository URI: https://github.com/google-parfait/confidential-federated-compute
Source Repository Digest: 0f8072c8e9dda36170f0fa466305e9664716fb56
Source Repository Ref: refs/heads/main
Source Repository Identifier: '775138920'
Source Repository Owner URI: https://github.com/google-parfait
Source Repository Owner Identifier: '164364956'
Build Config URI: https://github.com/google-parfait/confidential-federated-compute/.github/workflows/build.yaml@refs/heads/main
Build Config Digest: 0f8072c8e9dda36170f0fa466305e9664716fb56
Build Trigger: push
Run Invocation URI: https://github.com/google-parfait/confidential-federated-compute/actions/runs/10088700871/attempts/1

... <snip> ...
```

It describes that the ledger application binary was produced at commit
0f8072c8e9dda36170f0fa466305e9664716fb56 in the
https://github.com/google-parfait/confidential-federated-compute repository
using the "Build and attest all" workflow.
https://github.com/google-parfait/confidential-federated-compute/actions/runs/10088700871/attempts/1
has more information about the action that produced the binary.
