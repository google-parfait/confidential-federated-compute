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

NOTE: We plan to provide a tool in the near future that will facilitate these
types of inspections. For now, one must manually inspect the attestation
evidence (e.g. by using the
[Oak Attestation Verification library](https://github.com/project-oak/oak/tree/main/oak_attestation_verification))
and data access policy protos to identify the binary digests specified in the
evidence and access policy.

## Mapping binaries from an attestation verification record to their provenance

The binaries for the stage0 firmware, kernel, and container system image layers
are produced from the Oak repository. The container binaries, container
configurations, and restricted kernel application layer binaries are produced
from the code in this Confidential Federated Compute repository.

Most (but not yet all) of the binaries attested in the ledger attestation
evidence or allowed by a `DataAccessPolicy` have detailed entries on
sigstore.dev. Until we provide a more user-friendly tool, you can search for
records for a given SHA-256 binary hash using the
`https://search.sigstore.dev/?hash={THE_SHA256_HASH}` URL format. These entries
should show the binaries' provenance, including a link to the Git commit on
GitHub that the binaries were built from, and instructions for rebuilding the
same binaries in a reproducible manner.

NOTE: We will provide more details in the near future on how binaries at each
layer of the system (e.g. stage0, kernel, application binary) can be mapped to
their provenance, and how to inspect and/or reproducibly build their source
code.
