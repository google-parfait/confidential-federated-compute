# KMS reference values

This directory holds the `oak.attestation.v1.ReferenceValues` for the Key
Management Service. These reference values are used by the KMS for peer-to-peer
attestation and (mostly) match reference values used by client applications for
client-to-server attestation. Older clients may use older reference values in
certain circumstances, and clients may not verify the kernel command line if
they were built without a regex library.

KMS `ReferenceValues` specify the TEE's required hardware root of trust (e.g.
AMD SEV-SNP), minimum CPU TCB versions, as well as a number of public signing
keys with which the endorsements of per-layer TEE binaries is verified. They
also contain Rekor's public signing key, with which the inclusion of binary
endorsements in the Rekor log (an inclusion proof) is verified.

Client applications also validate that the access policies, which describe how
uploaded data are allowed to be processed, are endorsed and publicly
inspectable. They use the keys specified in the
[`/access_policy_reference_values`](/access_policy_reference_values) directory
for this purpose.

## Details on included endorsement keys

A KMS application consists of a number of binaries running in various layers
of the KMS TEE (firmware, kernel, application). At the time of writing, each
layer's binary is covered by a separate endorsement and transparency log entry,
and each endorsement is created using a different endorsement key.

At the time of writing, the KMS application is written using the Oak Containers
stack, and the [`reference_values.txtpb`](reference_values_txtpb) file specifies
five endorsement keys in use, as well as the Rekor key used to validate
inclusion proofs with. Each of the keys represent DER-encoded values, which can
be inspected as per the following example (using the Rekor key):

```console
$ # Inspect the key's details (and print a PEM-formatted version).
$ KEY="\x30\x59\x30\x13\x06\x07\x2a\x86\x48\xce\x3d\x02\x01\x06\x08\x2a\x86\x48\xce\x3d\x03\x01\x07\x03\x42\x00\x04\xd8\x6d\x98\xfb\x6b\x5a\x6d\xd4\xd5\xe4\x17\x06\x88\x12\x31\xd1\xaf\x5f\x00\x5c\x2b\x90\x16\xe6\x2d\x21\xad\x92\xce\x0b\xde\xa5\xfa\xc9\x86\x34\xce\xe7\xc1\x9e\x10\xbc\x52\xbf\xe2\xcb\x9e\x46\x85\x63\xff\xf4\x0f\xdb\x63\x62\xe1\x0b\x7d\x0c\xf7\xe4\x58\xb7"
$ echo -n $KEY | openssl pkey -pubin -text
-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE2G2Y+2tabdTV5BcGiBIx0a9fAFwr
kBbmLSGtks4L3qX6yYY0zufBnhC8Ur/iy55GhWP/9A/bY2LhC30M9+RYtw==
-----END PUBLIC KEY-----
Public-Key: (256 bit)
pub:
    04:d8:6d:98:fb:6b:5a:6d:d4:d5:e4:17:06:88:12:
    31:d1:af:5f:00:5c:2b:90:16:e6:2d:21:ad:92:ce:
    0b:de:a5:fa:c9:86:34:ce:e7:c1:9e:10:bc:52:bf:
    e2:cb:9e:46:85:63:ff:f4:0f:db:63:62:e1:0b:7d:
    0c:f7:e4:58:b7
ASN1 OID: prime256v1
NIST CURVE: P-256

$ # Calculate the key's fingerprint
$ echo -n $KEY | openssl dgst -sha256
SHA2-256(stdin)= c0d23d6ad406973f9559f3ba2d1ca01f84147d8ffc5b8445c224f98b9591801d
```
