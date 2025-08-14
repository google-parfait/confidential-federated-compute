# Access Policy Endorsement Reference Values (ERVs)

This directory holds `fcp.confidentialcompute.AccessPolicyEndorsementOptions`
messages, which in turn contain `oak.attestation.v1.EndorsementReferenceValue`
(ERV) messages, used by various client applications that use confidential
federated computations to verify that the access policies, which describe how
uploaded data is allowed to be processed, are valid and published to the Rekor
transparency log.

ERVs consist of a per-app public signing key with which access policy
endorsements are verified, and Rekor's public signing key, with which the
inclusion of the access policy endorsement in the Rekor log (an inclusion proof)
is verified.

Note that since these reference values are compiled into the client binaries,
older clients may be using older versions of these reference values.

Client applications also validate the attestation evidence for the KMS or ledger
service which enforces the access policies. Unless stated otherwise they use the
reference values specified in the
[/reference_values/ledger/reference_values.txtpb](/reference_values/ledger/reference_values.txtpb)
file for this purpose.

## Gboard

[Gboard](https://play.google.com/store/apps/details?id=com.google.android.inputmethod.latin)
uses the values in [`gboard.txtpb`](gboard.txtpb) to validate access policy
endorsements.

Processing of Gboard data with TEE-hosted confidential federated computations
will use differential privacy to produce anonymized results. We expect Gboard's
use of confidential federated computations to evolve over time, and any changes
to these details will be reflected here.
