# Inspecting Federated Compute attestation verification records

Be sure to review the introductory [README.md](README.md) before reviewing this
document.

As described in the *Enabling External Verifiability of the Data Processing*
section of the
[Confidential Federated Computations paper](https://arxiv.org/abs/2404.10764),
after a successful verification of KMS attestation evidence and the data access
policy by a client device, the device will log an
[attestation verification record](https://github.com/google-parfait/federated-compute/blob/main/fcp/protos/confidentialcompute/verification_record.proto)
which can then be inspected. Please see
[fcp/client/attestation/README.md](https://github.com/google-parfait/federated-compute/blob/main/fcp/client/attestation/README.md)
in the Federated Compute repository for instructions on how to gather
attestation verification records from a device that is using the Federated
Compute client library.

The following sections describe how the information in these attestation
verification records can be inspected.

Note: the main limitation of the approach described in this document is the
infeasibility of instrumenting all devices that could possibly contribute data.
There is a more scalable approach to inspecting the KMS attestation evidence and
data access policies that any client device may accepts, which leverages a
public transparency log and which addresses this limitation. See
[inspecting_endorsements.md](inspecting_endorsements.md) for more details on
this approach.

## Inspecting attestation verification records

Attestation verification records consist of two major parts:

1.  *the attestation evidence that identifies the [KMS](/kms) application* that
    generated the encryption key the client used to encrypt its data,

2.  *the [data access policy](/kms#access-policies)* that prescribes the
    conditions under which the KMS or ledger binary will allow that encrypted
    data to be decrypted.

To inspect the KMS attestation evidence and data access policy in an
`AttestationVerificationRecord`, the
[explain\_fcp\_attestation\_record](/tools/explain_fcp_attestation_record/) tool
in this repository can be used. This tool prints a human-readable summary of the
attestation evidence and the data access policy. The attestation evidence
summary includes links to
[SLSA provenance](https://slsa.dev/spec/v0.1/provenance) stored on sigstore.dev
for each of the binaries at each layer of the TEE-hosted application. The data
access policy summary does not currently include links to sigstore.dev for the
binary digests specified in each of the `ReferenceValues` protos in the policy,
but does include instructions for crafting such sigstore.dev links manually.

The following is an example of the type of output the tool produces.

```sh
$ bazelisk run //tools/explain_fcp_attestation_record:main -- --record=$PWD/extracted_records/record_l2_to_l19_digest12345678.pb
Inspecting AttestationVerificationRecord at extracted_records/record_l2_to_l19_digest12345678.pb.

========================================
======= KMS ATTESTATION EVIDENCE =======
========================================

Oak Containers Stack in a AMD SEV-SNP TEE

_____ Root Layer _____

... <snip> ...

_____ Container Layer _____

binary:
  sha2_256: d14a1852f08b528f9975245f300d44e2500aaff9c8a9c5572e0114b7203e3e47
config: {}


Note: this layer describes the "KMS" application binary, which is generally
a build of the `kms` in the
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
are produced from the Oak repository. The container binaries and container
configurations are produced from the code in this Confidential Federated Compute
repository. We believe that all binaries are reproducibly buildable.

All of the binaries attested in the KMS attestation evidence or allowed by a
`DataAccessPolicy` should have SLSA provenance entries on sigstore.dev. For any
allowed binary, you should be able to find the corresponding provenance using
the `https://search.sigstore.dev/?hash={THE_SHA256_HASH}` URL format, where
`{THE_SHA256_HASH}` is the SHA2-256 hash of the binary in the evidence/access
policy. These entries should show the binaries' provenance, including a link to
the Git commit on GitHub that the binaries were built from, as well as the
workflow that was used to build the binary, and which should allow you to
rebuild the same binary in a reproducible manner.

For example, below is an excerpt of the SLSA provenance record for the KMS
application binary listed in the example explanation output above
(https://search.sigstore.dev/?hash=d14a1852f08b528f9975245f300d44e2500aaff9c8a9c5572e0114b7203e3e47):

```
GitHub Workflow SHA: fe439180bb24e65ad46e869eddee88f9c9b34875
GitHub Workflow Name: Build and attest
GitHub Workflow Repository: google-parfait/confidential-federated-compute
GitHub Workflow Ref: refs/heads/main
OIDC Issuer (v2): https://token.actions.githubusercontent.com
Build Signer URI: https://github.com/google-parfait/confidential-federated-compute/.github/workflows/reusable_build.yaml@refs/heads/main
Build Signer Digest: fe439180bb24e65ad46e869eddee88f9c9b34875
Runner Environment: github-hosted
Source Repository URI: https://github.com/google-parfait/confidential-federated-compute
Source Repository Digest: fe439180bb24e65ad46e869eddee88f9c9b34875
Source Repository Ref: refs/heads/main
Source Repository Identifier: '775138920'
Source Repository Owner URI: https://github.com/google-parfait
Source Repository Owner Identifier: '164364956'
Build Config URI: https://github.com/google-parfait/confidential-federated-compute/.github/workflows/build.yaml@refs/heads/main
Build Config Digest: fe439180bb24e65ad46e869eddee88f9c9b34875
Build Trigger: push
Run Invocation URI: https://github.com/google-parfait/confidential-federated-compute/actions/runs/26261649715/attempts/1

... <snip> ...
```

It describes that the KMS application binary was produced at commit
fe439180bb24e65ad46e869eddee88f9c9b34875 in the
https://github.com/google-parfait/confidential-federated-compute repository
using the "Build and attest" workflow.
https://github.com/google-parfait/confidential-federated-compute/actions/runs/26261649715/attempts/1
has more information about the action that produced the binary.
