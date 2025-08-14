# Documentation for inspecting attestation evidence, data access policies, and binaries

As described in the *Client Side Architecture* section of the
[Confidential Federated Computations paper](https://arxiv.org/abs/2404.10764),
client devices running the
[Federated Compute](https://github.com/google-parfait/federated-compute) client
library and which participate in a `ConfidentialAggregations` protocol session
will verify attestation evidence for the CFC
[Key Management Service](/containers/kms) (KMS) or ledger applications hosted in
this repository. They will also verify the data access policy that the KMS or
ledger will enforce, which will specify one or more allowed data processing
applications which are built from this repository.

The paper describes two approaches for inspecting the KMS or ledger binaries and
the data processing applications authorized by the data access policies served
to devices, one based on instrumenting and inspecting a given client device, and
another based on monitoring "endorsements" published to a transparency log.
These approaches are complementary.

In the first approach, after setting up the instrumentation and after a
successful verification of KMS/ledger and access policy, these devices will log
an
[attestation verification record](https://github.com/google-parfait/federated-compute/blob/main/fcp/protos/confidentialcompute/verification_record.proto)
which can then be inspected. See
[inspecting_attestation_records.md](inspecting_attestation_records.md) for more
details on how these verification records can be mapped to specific binaries in
this repository. This approach's limitation is the infeasibility of
instrumenting all devices that could possibly contribute data.

The second approach involves monitoring the
[Rekor](https://docs.sigstore.dev/logging/overview/) transparency log for
KMS/ledger and data access policy endorsements, which are cryptographic
signatures of the KMS or ledger binary digests and data access policy digests.
Client devices use these endorsements and corresponding transparency log
inclusion proofs to decide whether or not to accept a given KMS, ledger, or
access policy. See [inspecting_endorsements.md](inspecting_endorsements.md) for
more details on this approach, including instructions for how to map such
endorsements to the specific binaries in this repository.

Common to both approaches a two-step verification process:

1.  Validating the KMS or ledger attestation verification, and determining the
    provenance of the binaries running in each layer (firmware, kernel,
    application) of the KMS or ledger TEE.

2.  Determining the provenances of the TEE-hosted data transformations to which
    the KMS or ledger will grant access to the encrypted data, as prescribed by
    the `ReferenceValues` in the data access policies served to clients devices.

A data access policy is effectively a graph describing allowed data
transformations, where each transformation has to be performed by a TEE-hosted
application. The set of allowable TEE-hosted applications for each
transformation step are prescribed by a `ReferenceValues` protobuf embedded in
the data access policy. The KMS or ledger verifies the identity of the
TEE-hosted data transformation using its attestation evidence. Only if this
identity matches the `ReferenceValues` is the data transform granted access to
the data.
