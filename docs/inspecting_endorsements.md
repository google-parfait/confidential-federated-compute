# Inspecting Confidential Federated Computation endorsements from the transparency log

Be sure to review the introductory [README.md](README.md) before reviewing this
document.

As described in the *Enhanced External Verifiability* section of the
[Confidential Federated Computations white
paper](https://arxiv.org/abs/2404.10764), client devices can verify the ledger
attestation evidence and data access policies by means of endorsements:
cryptographic signatures over the ledger binary attestation evidence and the
access policy, which in turn are published to a transparency log
([Rekor](https://docs.sigstore.dev/logging/overview/)).

Client devices validate that the ledger attestation evidence has been endorsed
by the appropriate binary endorsement keys, and they validate that there is a
valid corresponding transparency log inclusion proof, before accepting a ledger
public key. Client devices also validate that the access policy has been
endorsed by an appropriate endorsement key, and that this endorsement is also
covered by a transparency log inclusion proof.

By verifying these endorsements, the client can confirm that all data access
policies as well as the server-side ledger and data processing binaries are
correctly implemented and externally inspectable via their transparency log
entries. The following sections describe how those transparency log entries can
be discovered and mapped to the actual source code for the corresponding ledger
application and data processing steps described by the access policy.

The first part of this document describes how the complete set of ledger
applications that client devices may accept can be determined by monitoring the
Rekor transparency log. The second part of this document describes how the
complete set of active data access policies which clients may accept can be
determined in a similar fashion.

Note: there is an alternate approach to inspecting the ledger attestation
evidence and data access policies that a given client device accepts, which
involves instrumenting the device. See
[inspecting_attestation_records.md](inspecting_attestation_records.md)
for more details on this approach.

## Inspecting ledger binary transparency log entries

The endorsement keys client devices use to validate the ledger application can be
found in the [/reference_values/ledger](/reference_values/ledger) directory.

To find transparency log entries for these endorsement keys you can use the
[rekor-monitor](https://github.com/sigstore/rekor-monitor) tool. For example,
the following configuration lists the endorsement key fingerprints of each of
the ledger TEE binary layers, and will find a recent endorsement log entry for
the application layer TEE binary.

```console
$ cat << EOF
startIndex: 175178225
endIndex: 175178226
monitoredValues:
  fingerprints:
    - 98fc8ad40908f6c079c5d1677b85f261acdf08262c73f448f04bd4e9a090c8bb # stage0
    - 6052f352eac71f16947815eb34010f49ea2f1284a46b61777fb8c2accfa26d29 # kernel
    - 5f884b699bb66fe0b0ab07e2ee9ed9c221109ffdb2d13f470ed964952271d867 # init_ram_fs / orchestrator
    - ea0d1f8ffed9512019a2ec968790263d88ea3324c6a6a782114e5ea1be4fd38f # application
EOF > config.yml

$ go run github.com/sigstore/rekor-monitor/cmd/rekor_monitor@6248cd70ec4f0c18e4d23901041caea126da36bc \
    --config-file config.yml
...
Found ea0d1f8ffed9512019a2ec968790263d88ea3324c6a6a782114e5ea1be4fd38f 175178226 108e9186e8c5677a5d4bc53a7212664ffa2e4d86e5365059962ddad993148a08d7ecf4e667b7d30a
```

Note that the above configuration limits the tool's search to a log index range,
but you can also omit the start and end indices and run the tool periodically
(say, every 10 minutes), in which case the tool will scan all new log entries
added since the last run, and print out any that match the configured endorsement
keys. Rekor also exposes a [REST API](https://www.sigstore.dev/swagger/) and a
[GCP Pub/Sub event stream](https://docs.sigstore.dev/logging/event_stream/)
which you can use to monitor for new log entries instead.

For each of the entries matching the keys we're interested in, the full Rekor
log entry can then be accessed via the URL pattern
`https://search.sigstore.dev/?uuid={UUID}`, where `{UUID}` is the 3rd
alphanumeric column value. For example,
https://search.sigstore.dev/?uuid=108e9186e8c5677a5d4bc53a7212664ffa2e4d86e5365059962ddad993148a08d7ecf4e667b7d30a
shows the Rekor log entry payload for the application layer, copied below:

```yaml
apiVersion: 0.0.1
kind: rekord
spec:
  data:
    hash:
      algorithm: sha256
      value: cf687788b351802ef390c1bad0e0b7ff29b88055424305c0d714b96007f315b7
  signature:
    content: >-
      MEUCIBu+kSOo9HwxGnhBudwxsGp1b0nNsrJDjD64BeFtW+WMAiEAqVKvRcfFxgerNI8U+mZaSIibg93gqKhdeBTB3QEnDRI=
    format: x509
    publicKey:
      content: |
        -----BEGIN PUBLIC KEY-----
        MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEZ8zsWgnck5wLdySC3AHZVGfGWaxh
        kJYqmqbq1pqtsJwSbe5lJndS0zxoFPqmMpvuiMLuYgbHp5T9ex9xOYyG9A==
        -----END PUBLIC KEY-----
```

We can also look up the raw Rekor log entry using the [`getLogEntryByUUID` REST
API](https://www.sigstore.dev/swagger/#/entries/getLogEntryByUUID), which will
show us not just the log entry payload, but also its log integration time, log
index, and log ID, which is the SHA2-256 hash (fingerprint) of the DER-encoded
public key for the log at the time the entry was included in the log:

```console
$ curl -sL "https://rekor.sigstore.dev/api/v1/log/entries/108e9186e8c5677a5d4bc53a7212664ffa2e4d86e5365059962ddad993148a08d7ecf4e667b7d30a" \
    | jq
{
  "108e9186e8c5677a5d4bc53a7212664ffa2e4d86e5365059962ddad993148a08d7ecf4e667b7d30a": {
    "body": "eyJhcGlWZXJz...",
    "integratedTime": 1740685839,
    "logID": "c0d23d6ad406973f9559f3ba2d1ca01f84147d8ffc5b8445c224f98b9591801d",
    "logIndex": 175178226,
    ...
  }
}
```

Note that this log ID
`c0d23d6ad406973f9559f3ba2d1ca01f84147d8ffc5b8445c224f98b9591801d` matches the
fingerprint of the Rekor key we've included in our reference values. You can
fetch (and validate the fingerprint of) this key yourself using Rekor's REST
API:

```console
$ curl -sL 'https://rekor.sigstore.dev/api/v1/log/publicKey' \
  | openssl pkey -pubin -inform pem -outform der \
  | openssl dgst -sha256
SHA2-256(stdin)= c0d23d6ad406973f9559f3ba2d1ca01f84147d8ffc5b8445c224f98b9591801d
```

The `jq` tool also allows us to parse the transparency log entry's body
programmatically. For example, we can conveniently find the endorsed SHA2-256
digest as follows:

```console
$ curl -sL "https://rekor.sigstore.dev/api/v1/log/entries/108e9186e8c5677a5d4bc53a7212664ffa2e4d86e5365059962ddad993148a08d7ecf4e667b7d30a" \
    | jq -r '.[] | .body' | base64 -d | jq '.spec.data.hash.value'

"cf687788b351802ef390c1bad0e0b7ff29b88055424305c0d714b96007f315b7"
```

### From transparency log entry to Oak endorsement

The SHA2-256 hash in the transparency log above
`cf687788b351802ef390c1bad0e0b7ff29b88055424305c0d714b96007f315b7` is the hash of
an Oak endorsement (see
[schema](https://github.com/project-oak/oak/blob/main/docs/tr/endorsement_v1.md)),
which in turn specifies a subject digest which is a hash of the actual payload
being endorsed, along with an endorsement validity time range which the client
will respect.

These Oak endorsements can be looked up using the following URI pattern:
https://federatedcompute-pa.googleapis.com/data/transparency/sha2-256:{HASH_ON_REKOR}
where `{HASH_ON_REKOR}` is the hash listed in the Rekor log entry. For the
example above we can find the Oak endorsement statement at
https://federatedcompute-pa.googleapis.com/data/transparency/sha2-256:cf687788b351802ef390c1bad0e0b7ff29b88055424305c0d714b96007f315b7,
copied below:

```json
{
  "_type": "https://in-toto.io/Statement/v1",
  "predicateType": "https://project-oak.github.io/oak/tr/endorsement/v1",
  "subject": [
    {
      "name": "replicated_ledger",
      "digest": {
        "sha256": "0714299084b669bf57ac8831068ba0b719f25f0a8d4aab83d8de3c09cd513c02"
      }
    }
  ],
  "predicate": {
    "issuedOn": "2025-02-27T19:50:35.017000Z",
    "validity": {
      "notBefore": "2025-02-27T19:50:35.017000Z",
      "notAfter": "2025-05-28T19:50:35.017000Z"
    }
  }
}
```

You can check for yourself that this endorsement matches the hash in the
transparency log entry, by calculating its hash:

```console
$ curl -sL "https://federatedcompute-pa.googleapis.com/data/transparency/sha2-256:cf687788b351802ef390c1bad0e0b7ff29b88055424305c0d714b96007f315b7" | openssl dgst -sha256
SHA2-256(stdin)= cf687788b351802ef390c1bad0e0b7ff29b88055424305c0d714b96007f315b7
```

The meaning of the endorsement subject digest differs per TEE layer being
endorsed, as detailed in the following sections. However, in all cases the
endorsements eventually lead to one or more hashes of reproducibly built
binaries for which provenance — SLSA log entries describing how you can rebuild
the same binary with the same hash — can be found on Rekor. These provenance
entries can be found using the URL pattern:
https://search.sigstore.dev/?hash={BINARY_HASH} where `{BINARY_HASH}` is the
SHA2-256 hash of the endorsed binary payload. We'll illustrate this in more
detail in the following sections.

### Ledger stage0 firmware layer

At the time of writing,
[`108e9186e8c5677a3c58f26a32fb956dea56de36a7714046dcc5dffbb4efddb16fb6081849062525`](https://search.sigstore.dev/?uuid=108e9186e8c5677a3c58f26a32fb956dea56de36a7714046dcc5dffbb4efddb16fb6081849062525)
is the Rekor UUID of a recent stage0 firmware endorsement, which covers an Oak
endorsement with hash
`46ef697e2c79abc9b59822e4b809a8567eee5124b65bb48033cd99874f501662`, which can be
fetched from
https://federatedcompute-pa.googleapis.com/data/transparency/sha2-256:46ef697e2c79abc9b59822e4b809a8567eee5124b65bb48033cd99874f501662:

```json
{
  "_type": "https://in-toto.io/Statement/v1",
  "predicateType": "https://project-oak.github.io/oak/tr/endorsement/v1",
  "subject": [
    {
      "name": "stage0_bin",
      "digest": {
        "sha256": "f96f934c66a6ea35c08015a74d43db748856fa0c1d1684beadb83f9b578c7bbd"
      }
    }
  ],
  "predicate": {
    "issuedOn": "2025-02-21T17:12:11.273000Z",
    "validity": {
      "notBefore": "2025-02-21T17:12:11.273000Z",
      "notAfter": "2025-05-22T17:12:11.273000Z"
    },
    "claims": [
      {
        "type": "https://github.com/project-oak/oak/blob/main/docs/tr/claim/10271.md"
      },
      {
        "type": "https://github.com/project-oak/oak/blob/main/docs/tr/claim/66738.md"
      }
    ]
  }
}
```

The subject digest of stage0 layer endorsements is the SHA2-256 hash of another
intermediate payload, which is an `oak.attestation.v1.FirmwareAttachment`-typed
protobuf (see
https://github.com/project-oak/oak/blob/main/docs/tr/claim/10271.md for more
info).

The payload isn't human-readable since it is a binary protobuf, but we can get a
more human-readable view using the `protoc` tool (we've added some annotations
for illustration purposes below). It will print out one or more hashes, per Oak
Restricted Kernel vCPU configuration:

```text
$ curl -sL https://federatedcompute-pa.googleapis.com/data/transparency/sha2-256:f96f934c66a6ea35c08015a74d43db748856fa0c1d1684beadb83f9b578c7bbd \
    | protoc --decode_raw
1 {
  1: 1   # <-- configuration for 1 vCPU
  2 {    # v-- SHA2-384 hash of stage0 binary for this configuration
    32: "b9cbfccb673c78503e1deca179623519c9c78809793d568df8ada85a75e3d80716a4fa9af55695f19763cb098378359f"
  }
}
1 {
  1: 4
  2 {
    32: "ef5da6d0ba25a5c155bea7dbc217e6d605e8e2161143fa9f6d8e7e7916807c3b64a4de8f5e4b9722df80d3f34a43901d"
  }
}
1 {
  1: 16
  2 {
    32: "ad74b552c29ee357ecda144ea9174743ccd9976a726bddee4b03f6fed123e3ace6b387452549403dad8a475ce8a53943"
  }
}
1 {
  1: 64
  2 {
    32: "5c2c94de1453c26ff85f016d28c6b9516f1a03fc4e4a01a922de9e998b9d674e3101f0b783b29bb83840336e03e1672f"
  }
}
```

These are the hashes of the individual kernel binaries, for which provenance is
available on Rekor (they correspond to field 32 of the
[HexDigest](https://github.com/project-oak/oak/blob/25fdba08d2fc1c25c6d4b0a185ed7f3d86d770ba/proto/digest.proto#L51)
protobuf). However, these stage0 firmware binaries will be measured by the AMD
CPU and included in the CPU's SEV-SNP attestation report, and the AMD uses the
SHA2-384 hash function to perform this measurement. Because Rekor only allows
storing provenance by SHA2-256 hash, we must therefore look up their provenance
by decoding the hex-formatted SHA2-384 hash and then calculating the SHA2-256
hash. We can do this for the first firmware configuration's hash as follows:

```console
$ echo -n "b9cbfccb673c78503e1deca179623519c9c78809793d568df8ada85a75e3d80716a4fa9af55695f19763cb098378359f" \
    | xxd -r -p \
    | openssl dgst -sha256
SHA2-256(stdin)= 0b52f9eedf77d524d09ef401a54693e39426f32167420e8274abe889bfaa6acf
```

It is this SHA2-256 hash for which provenance is actually available on Rekor. At
https://search.sigstore.dev/?hash=0b52f9eedf77d524d09ef401a54693e39426f32167420e8274abe889bfaa6acf
we can find the following excerpt:

```text
Source Repository URI: https://github.com/project-oak/oak
Source Repository Digest: 5122264bccf471dacb15f092f4e8f5f32df97bb1
...
Run Invocation URI: https://github.com/project-oak/oak/actions/runs/13455834315/attempts/1
```

This tells us that this stage0 binary was built from commit
`5122264bccf471dacb15f092f4e8f5f32df97bb1` in
https://github.com/project-oak/oak. And in the
`build_attest_all(buildconfigs/stage0_bin.sh)` section of the "Run Invocation"
logs we can see:

```text
Attestation created for sha2_384_measurement_of_initial_memory_with_stage0_and_01_vcpu@sha256:0b52f9eedf77d524d09ef401a54693e39426f32167420e8274abe889bfaa6acf
```

indicating that this was the firmware configuration targeting a single vCPU.


### Ledger kernel layer

At the time of writing,
[`108e9186e8c5677a6a3c05eab59b258c20bc813a6f2c6118bbb34cdba272afecfac7578cb727ba67`](https://search.sigstore.dev/?uuid=108e9186e8c5677a6a3c05eab59b258c20bc813a6f2c6118bbb34cdba272afecfac7578cb727ba67)
is the Rekor UUID of a recent kernel layer endorsement, which covers an Oak
endorsement with hash
covers an Oak endorsement with hash
`169d7dc7dfaaff4769af20aef4642787d1aeb86ba8d5a7dd9d52b6fcd0c6427d`
which can be fetched from
https://federatedcompute-pa.googleapis.com/data/transparency/sha2-256:169d7dc7dfaaff4769af20aef4642787d1aeb86ba8d5a7dd9d52b6fcd0c6427d:

```json
{
  "_type": "https://in-toto.io/Statement/v1",
  "predicateType": "https://project-oak.github.io/oak/tr/endorsement/v1",
  "subject": [
    {
      "name": "oak_restricted_kernel_simple_io_init_rd_wrapper_bin",
      "digest": {
        "sha256": "e07ad7496484e4ec22ed1bb2fa5b4cdbc58703a64307d0e38f1c0d1facf540bd"
      }
    }
  ],
  "predicate": {
    "issuedOn": "2025-02-21T17:12:10.883000Z",
    "validity": {
      "notBefore": "2025-02-21T17:12:10.883000Z",
      "notAfter": "2025-05-22T17:12:10.883000Z"
    },
    "claims": [
      {
        "type": "https://github.com/project-oak/oak/blob/main/docs/tr/claim/36746.md"
      },
      {
        "type": "https://github.com/project-oak/oak/blob/main/docs/tr/claim/98982.md"
      }
    ]
  }
}
```

The subject digest of Oak Restricted Kernel layer endorsements is the SHA2-256
hash of another intermediate payload, which is an
`oak.attestation.v1.KernelAttachment`-typed protobuf (see
https://github.com/project-oak/oak/blob/main/docs/tr/claim/98982.md for more
info).

We can fetch this protobuf using the same '/data/transparency' URL pattern at
URL
https://federatedcompute-pa.googleapis.com/data/transparency/sha2-256:e07ad7496484e4ec22ed1bb2fa5b4cdbc58703a64307d0e38f1c0d1facf540bd,
and it generally will contain one or more hashes, one per Oak Restricted Kernel
vCPU configuration. In this case it looks as follows:

```console
$ curl -sL "https://federatedcompute-pa.googleapis.com/data/transparency/sha2-256:e07ad7496484e4ec22ed1bb2fa5b4cdbc58703a64307d0e38f1c0d1facf540bd" \
    | protoc --decode_raw
1 { # <--- kernel image
  16: "0200006163004daef634350537af029a208e0bf9ffebcb494639bcc8882b4f6a8b5de7a8d3b1"
  17: "ead4c09277f74ae48cf1c04bb3811266bd099910"
  18: "a25a7e2ab3bae81fdef8b31974596167ef31af59128ba7b6e05b5ee473222b02" # <--- SHA2-256 hash
  19: "c0c5c492b72931b64a45975f09bcaff14ef8e2a5e04f3ed6b0156ce07ea29fdac173ca09c0b2be8f7da1afcb907a39e61a096cc014efcf96aa4194245e721771"
  20: "0abcdb8bdea145b841050a773510f5c0bc114851c81cf16122e0f50f199f46ae70a9c8c9fa4802a098d1df59d5c8647349cc877867b31e55d65420926a58efab"
  21: "055586ef72f153d1662778d263ceea8481e46af762bb9f7144f1e135ffa460c1990511228e69d5ffc2ceb3a90d309725"
  22: "d4c2cc977c765b5ba40f4ac15c9c3c947c72eb0588da3ab23245ef426c31bb17"
  23: "00e0b7e3afca1594b30fb3f656dc1ce2d688991cb958c6e2fe3f5d62"
  32: "496303095a9e203090732d7ef5959860e2288c150d0d00597a1278bffb9dc40726d540d239d892ed0186a674b8d5eabf"
}
2 { # <--- kernel setup data
  16: "01001000f787ecb996c0158085698f8a71f75cd023bebb4549c315c4bb27388da4020171"
  17: "d0787af619375feb58490bd08ee85d25266466d3"
  18: "4cd020820da663063f4185ca14a7e803cd7c9ca1483c64e836db840604b6fac1" # <--- SHA2-256 hash
  19: "a6a0e968a93fa544e8cba746455cb4d6b6e005ac1ea3d62bdd531e2fb38d1a9e6fd3d82240ebef54aca6b196ff7b52b0ed95a885b82bb3e7acd5920ba0a0d194"
  20: "e68d9fde05550f9404ec03b21c469ce5f28e2afb471718f87e8a39262ac4abeebaf052296cca15c3a530bb6c1367d9bfc0be847f6a3d3278f199591fda3ac5c3"
  21: "68a52675263b95c44f2cda0ff46cac7b6b4900dbba648332ba0bff32b42800dc7cb59457a8d232e73016d2cf10812cb9"
  22: "8afd043b8a0b124988965a8774a60b72675aedca94fc9f6c210f75bb56808c9c"
  23 {
    12: "1c92b1a571213d971aee789a7c04509e45648d52467f0c217e3207"
  }
  32: "dd79803b4e303d6f5355d15f009ea6d7d75e2db2f4813e2a95fab171aa89a916ec68149c0560e8b664be34ff22e864ec"
}
```

In this wall of text,
`a25a7e2ab3bae81fdef8b31974596167ef31af59128ba7b6e05b5ee473222b02` represents
the SHA2-256 hash of the actual kernel image, and
`4cd020820da663063f4185ca14a7e803cd7c9ca1483c64e836db840604b6fac1` is the
SHA2-256 hash of the kernel's setup data (they correspond to field 18 of the
[HexDigest](https://github.com/project-oak/oak/blob/25fdba08d2fc1c25c6d4b0a185ed7f3d86d770ba/proto/digest.proto#L51)
protobuf).

It is these SHA2-256 hashes for which provenance is actually available on Rekor.
At
https://search.sigstore.dev/?hash=a25a7e2ab3bae81fdef8b31974596167ef31af59128ba7b6e05b5ee473222b02
we can find the following excerpt:

```text
Source Repository URI: https://github.com/project-oak/oak
Source Repository Digest: c1826e3801bb098b56d4d3e6df79989b0b354b9b
...
Run Invocation URI: https://github.com/project-oak/oak/actions/runs/12916191013/attempts/1
```

This tells us that this kernel binary was built from commit
`c1826e3801bb098b56d4d3e6df79989b0b354b9b` in
https://github.com/project-oak/oak. And in the
`build_attest_all(buildconfigs/oak_restricted_kernel_simple_io_init_rd_wrapper_bin.sh)`
section of the "Run Invocation" logs we can see the corresponding build outputs
being attested:

```text
Attestation created for oak_restricted_kernel_wrapper_simple_io_channel_measurement_image@sha256:a25a7e2ab3bae81fdef8b31974596167ef31af59128ba7b6e05b5ee473222b02
Attestation created for oak_restricted_kernel_wrapper_simple_io_channel_measurement_setup_data@sha256:4cd020820da663063f4185ca14a7e803cd7c9ca1483c64e836db840604b6fac1
```

### Ledger init RAM FS layer

At the time of writing,
[`108e9186e8c5677aac48856a4b42c1a0ca3fe504e08521daded9c7069206920cf0492891a98e0afd`](https://search.sigstore.dev/?uuid=108e9186e8c5677aac48856a4b42c1a0ca3fe504e08521daded9c7069206920cf0492891a98e0afd)
is the Rekor UUID of a recent kernel layer endorsement, which covers an Oak
endorsement with hash
covers an Oak endorsement with hash
`31345ae627374a555f54752dbb6ee717b11d7aeb9b24fdddda2b941771938b18`
which can be fetched from
https://federatedcompute-pa.googleapis.com/data/transparency/sha2-256:31345ae627374a555f54752dbb6ee717b11d7aeb9b24fdddda2b941771938b18:

```json
{
  "_type": "https://in-toto.io/Statement/v1",
  "predicateType": "https://project-oak.github.io/oak/tr/endorsement/v1",
  "subject": [
    {
      "name": "oak_orchestrator",
      "digest": {
        "sha256": "fe2584c47dbcd4b35dd47138bf6a8a17b68001fdd5848d5a27964a60cc8b4407"
      }
    }
  ],
  "predicate": {
    "issuedOn": "2025-02-21T17:12:10.867000Z",
    "validity": {
      "notBefore": "2025-02-21T17:12:10.867000Z",
      "notAfter": "2025-05-22T17:12:10.867000Z"
    },
    "claims": [
      {
        "type": "https://github.com/project-oak/oak/blob/main/docs/tr/claim/87425.md"
      }
    ]
  }
}
```

The subject digest of init ram FS layer endorsements is simply the SHA2-256 hash
of reproducibly buildable Oak Orchestrator binary being endorsed. Hence, its
provenance can be found directly on Rekor. In this case at
https://search.sigstore.dev/?hash=fe2584c47dbcd4b35dd47138bf6a8a17b68001fdd5848d5a27964a60cc8b4407,
with the following excerpt:

```text
Source Repository URI: https://github.com/project-oak/oak
Source Repository Digest: 230ceea48f1d4f1fdc77aa40087a79c4977e5ad9
...
Run Invocation URI: https://github.com/project-oak/oak/actions/runs/13402171427/attempts/1
```

This tells us that this kernel binary was built from commit
`230ceea48f1d4f1fdc77aa40087a79c4977e5ad9` in
https://github.com/project-oak/oak. And in the `build_attest_all
(buildconfigs/oak_orchestrator.sh)` section of the "Run Invocation" logs we can
see the corresponding build outputs being attested:

```text
Attestation created for oak_orchestrator@sha256:fe2584c47dbcd4b35dd47138bf6a8a17b68001fdd5848d5a27964a60cc8b4407
```

### Ledger application layer

The subject digest of application layer endorsements is simply the SHA2-256 hash of
reproducibly buildable ledger application binary being endorsed.

While all previous binaries we looked at so far are produced from code in the
Oak GitHub project, the ledger application binary is maintained in this
repository. It is this binary that implements the business logic of protecting
data decryption keys in accordance with the data access policies associated with
uploaded encrypted data.

As we showed in the [earlier
section](#from-transparency-log-entry-to-oak-endorsement), the application layer
endorsement covers subject digest
0714299084b669bf57ac8831068ba0b719f25f0a8d4aab83d8de3c09cd513c02, for which we
can hence find provenance at
https://search.sigstore.dev/?hash=0714299084b669bf57ac8831068ba0b719f25f0a8d4aab83d8de3c09cd513c02,
with the following excerpt:

```
Source Repository URI: https://github.com/google-parfait/confidential-federated-compute
Source Repository Digest: 54636c42aab5c7a72a5b1533af7d49e48b4b5d30
...
Run Invocation URI: https://github.com/google-parfait/confidential-federated-compute/actions/runs/13576761178/attempts/1
```

It shows that the `replicated_ledger` application binary was produced at commit
`54636c42aab5c7a72a5b1533af7d49e48b4b5d30` in the
https://github.com/google-parfait/confidential-federated-compute repository. The
"Run Invocation" logs have more information about the action that produced the
binary, which you can use to reproduce the build steps and arrive at the same
result.

## Inspecting data access policy transparency log entries

The endorsement keys client devices use for validating data access policies
depends on the app running on the device, and per-app endorsement keys can be
found in the [/reference_values/access_policy](/reference_values/access_policy)
directory. For each data access policy endorsement key there is an associated
set of privacy properties that the corresponding app will ensure are upheld by
every endorsed data access policy.

Just as with the ledger binary endorsements, you can use the `rekor-monitor`
tool to find transparency log entries for this endorsement key. The following
example configuration will surface a recent data access policy endorsement for
the
[Gboard](https://play.google.com/store/apps/details?id=com.google.android.inputmethod.latin)
app:

```console
$ cat > config.yml << 'EOF'
startIndex: 175189128
endIndex: 175189129
monitoredValues:
  fingerprints:
    - 3ebd2cd4ec2a56655c9022e734d7469c0a8612f7f676b001d090897f36bae560 # Gboard
EOF

$ go run github.com/sigstore/rekor-monitor/cmd/rekor_monitor@6248cd70ec4f0c18e4d23901041caea126da36bc \
    --config-file config.yml
...
Found 3ebd2cd4ec2a56655c9022e734d7469c0a8612f7f676b001d090897f36bae560 175189129 108e9186e8c5677af9f63f6c97d4bcaee3210af2da8ae214ebb56b76e2456c2bccdee68bc44c5e48
```

The data access policy endorsement transparency log entry with UUID
[`108e9186e8c5677af9f63f6c97d4bcaee3210af2da8ae214ebb56b76e2456c2bccdee68bc44c5e48`](108e9186e8c5677af9f63f6c97d4bcaee3210af2da8ae214ebb56b76e2456c2bccdee68bc44c5e48)
covers an endorsement with SHA2-256 hash
`d72534e0cae135b3ac11395d37956dbb89b328df4274a28b35cca22500845033`, which we can
download from
https://federatedcompute-pa.googleapis.com/data/transparency/sha2-256:d72534e0cae135b3ac11395d37956dbb89b328df4274a28b35cca22500845033,
and it in turn endorses an access policy binary, which we can download from
https://federatedcompute-pa.googleapis.com/data/transparency/sha2-256:a324fe9606c953cfba56a666ad642fcef61764d0b0bd1878d3833842aa5dfbda.

However, since the policy is another binary protobuf it is more useful to print
a summary of the access policy using the `tools/explain_fcp_attestation_record`
tool:

```console
$ curl -sL "https://federatedcompute-pa.googleapis.com/data/transparency/sha2-256:a324fe9606c953cfba56a666ad642fcef61764d0b0bd1878d3833842aa5dfbda" \
  | bazelisk run //tools/explain_fcp_attestation_record:main -- --access-policy -
... <snip> ...
The data access policy allows 2 data transformations and defines 0 shared access budgets
... <snip> ...
>>>>> Transform #0 <<<<<
Source blob ID: 0

Access budgets: the transform's access to its source blob is gated by *all* of the following access rules:
- limited access budget (at most 1 times): the transform may only access its source blob this many times.

Application matcher for this transform:
- Tag:
- Binary configuration restrictions:
  StructMatcher { fields: [FieldMatcher { path: "serialize_dest", matcher: Some(ValueMatcher { kind: Some(NumberValue(NumberMatcher { kind: Some(Eq(1.0)) })) }) }, FieldMatcher { path: "intrinsic_uri", matcher: So
me(ValueMatcher { kind: Some(StringValue(StringMatcher { kind: Some(Eq("fedsql_dp_group_by")) })) }) }, FieldMatcher { path: "epsilon", matcher: Some(ValueMatcher { kind: Some(NumberValue(NumberMatcher { kind: Som
e(Le(1.0987)) })) }) }, FieldMatcher { path: "delta", matcher: Some(ValueMatcher { kind: Some(NumberValue(NumberMatcher { kind: Some(Le(1e-8)) })) }) }] }
- Applications performing this transform must provide attestation evidence that can be verified with the following reference values:

Reference values for the Oak Containers stack
... <snip> ...
_____ Container Layer _____

Attestations identifying artifacts accepted by the reference values for this layer are described below.
Accepted Binary Artifacts:
- https://search.sigstore.dev/?hash=898c9654f7b7ad51967c292b6731a4c08b65e5d7196ada28359c6fe49a7f3f96
... <snip> ...

>>>>> Transform #1 <<<<<
Source blob ID: 1

Access budgets: the transform's access to its source blob is gated by *all* of the following access rules:
- limited access budget (at most 1 times): the transform may only access its source blob this many times.

Application matcher for this transform:
- Tag:
- Binary configuration restrictions:
  StructMatcher { fields: [FieldMatcher { path: "serialize_dest", matcher: Some(ValueMatcher { kind: Some(NumberValue(NumberMatcher { kind: Some(Eq(1.0)) })) }) }, FieldMatcher { path: "intrinsic_uri", matcher: So
me(ValueMatcher { kind: Some(StringValue(StringMatcher { kind: Some(Eq("fedsql_dp_group_by")) })) }) }, FieldMatcher { path: "epsilon", matcher: Some(ValueMatcher { kind: Some(NumberValue(NumberMatcher { kind: Som
e(Le(1.0987)) })) }) }, FieldMatcher { path: "delta", matcher: Some(ValueMatcher { kind: Some(NumberValue(NumberMatcher { kind: Some(Le(1e-8)) })) }) }] }
- Applications performing this transform must provide attestation evidence that can be verified with the following reference values:

Reference values for the Oak Containers stack
... <snip> ...
_____ Container Layer _____

Attestations identifying artifacts accepted by the reference values for this layer are described below.
Accepted Binary Artifacts:
- https://search.sigstore.dev/?hash=898c9654f7b7ad51967c292b6731a4c08b65e5d7196ada28359c6fe49a7f3f96
... <snip> ...
```

From this output, we can see that the access policy allows a two-stage
processing pipeline, for both stages using a TEE-hosted data processing
application for which the Oak Containers application layer binary provenance is
available at
https://search.sigstore.dev/?hash=898c9654f7b7ad51967c292b6731a4c08b65e5d7196ada28359c6fe49a7f3f96.
Also included in the tools' output, but not displayed in excerpt above, are
details on the type of required TEE (e.g. AMD SEV-SNP) and links to the binary
provenance for every other layer of the data processing application in the
policy.

By inspecting the `Binary configuration restrictions` section of the tools'
output we can also conclude that the data processing application is being told
to perform a `fedsql_dp_group_by` aggregation (which uses differential privacy,
or DP), and that it enforces maximum DP epsilon and delta values of 1.0987 and
1e-8, respectively.

When looking at the data processing binary's provenance on Rekor, we can see the following:

```text
Source Repository URI: https://github.com/google-parfait/confidential-federated-compute
Source Repository Digest: ee6e28479eace01ef8f05c156a7020da0676ac1c
Run Invocation URI: https://github.com/google-parfait/confidential-federated-compute/actions/runs/12912957042/attempts/1
```

Telling us that the binary was built at commit
ee6e28479eace01ef8f05c156a7020da0676ac1c in the
[confidential-federated-compute](https://github.com/google-parfait/confidential-federated-compute)
repository. The logs available at the "Run Invocation" link have more details.
The source code for this particular application binary is available in the
[/containers/fed_sql](/containers/fed_sql) directory.

## Additional notes

### Restrictions on endorsement log integration times
To make it easier to discover all relevant active endorsements, we have added
functionality to limit the endorsements that will be accepted by the ledger
during peer-to-peer attestation, as well as by client devices, to those
published to the transparency log only after a certain absolute date and within
a certain number of days from the current time at the time of verification.

This is reflected in the ledger and data access policies' reference values use
of the `signed_timestamp` field:

```
rekor {
  verify {
    keys { ... }
    signed_timestamp {
      not_before_absolute {
        seconds: 1739577600  # 2025-02-15 00:00:00 UTC
      }
      not_before_relative {
        seconds: -7776000  # 90 days
      }
    }
  }
}
```

This indicates that only endorsements integrated into Rekor (as indicated by the
`integratedTime` field shown in previous sections) *after February 15, 2025* and
*within 90 days* of the current time will be accepted. We are currently rolling
out these restrictions across the production client & server stack, and may make
further improvements in this area in the future.

### Availability of resources via `federatedcompute-pa.googleapis.com` URIs

The `https://federatedcompute-pa.googleapis.com/data/transparency` endpoint
plays an important role in facilitating the discovery the provenance associated
with each endorsement. For this reason we endeavour to keep this endpoint highly
available, and to keep all endorsement resources available for at least 1 year
from the time of endorsement creation. Any changes to this policy will be
reflected here.

## Conclusion

Following the instructions above, you can determine all possible code paths
through which uploaded client data might be processed. By inspecting the
provenance (open-source code and build instructions) for all relevant TEE-hosted
binaries (ledger and data processing steps) and access policies, anyone can
follow along and validate or falsify the privacy properties of data processed
using confidential federated computations.
