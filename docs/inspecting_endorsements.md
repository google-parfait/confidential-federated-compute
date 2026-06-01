# Inspecting Confidential Federated Computation endorsements from the transparency log

Be sure to review the introductory [README.md](README.md) before reviewing this
document.

As described in the *Enhanced External Verifiability* section of the
[Confidential Federated Computations white paper](https://arxiv.org/abs/2404.10764),
client devices can verify the KMS attestation evidence and data access policies
by means of endorsements: cryptographic signatures over the binary attestation
evidence and the access policy, which in turn are published to a transparency
log ([Rekor](https://docs.sigstore.dev/logging/overview/)).

Client devices validate that the KMS attestation evidence has been endorsed by
the appropriate binary endorsement keys, and they validate that there is a valid
corresponding transparency log inclusion proof, before accepting a KMS public
key. Client devices also validate that the access policy has been endorsed by an
appropriate endorsement key, and that this endorsement is also covered by a
transparency log inclusion proof.

By verifying these endorsements, the client can confirm that all data access
policies as well as the server-side KMS and data processing binaries are
correctly implemented and externally inspectable via their transparency log
entries. The following sections describe how those transparency log entries can
be discovered and mapped to the actual source code for the corresponding KMS
application and data processing steps described by the access policy.

The first part of this document describes how the complete set of KMS
applications that client devices may accept can be determined by monitoring the
Rekor transparency log. The second part of this document describes how the
complete set of active data access policies which clients may accept can be
determined in a similar fashion.

Note: there is an alternate approach to inspecting the KMS attestation evidence
and data access policies that a given client device accepts, which involves
instrumenting the device. See
[inspecting_attestation_records.md](inspecting_attestation_records.md) for more
details on this approach.

## Inspecting binary transparency log entries

The endorsement keys client devices use to validate the KMS can be found in the
[`/kms/reference_values`](/kms/reference_values) directory. There are at least
two ways for finding Rekor transparency log entries for endorsement keys: via
the `rekor-monitor` tool, and via Rekor's public BigQuery dataset. These are
documented in the next sections.

### Finding transparency log entries using the `rekor-monitor` tool

To find transparency log entries for these endorsement keys you can use the
[rekor-monitor](https://github.com/sigstore/rekor-monitor) tool. For example,
the following configuration lists the endorsement key fingerprints of each of
the KMS TEE binary layers, and will find a recent endorsement log entry for
the container layer TEE binary.

```console
$ cat << EOF
startIndex: 1634041291
endIndex: 1634041292
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
Found ea0d1f8ffed9512019a2ec968790263d88ea3324c6a6a782114e5ea1be4fd38f 1634041292 108e9186e8c5677a4e8e68884126228c18dbe566ab43eb0be79b077401149bd340bc295d78e9cd86
```

Note that the above configuration limits the tool's search to a log index range,
but you can also omit the start and end indices and run the tool periodically
(say, every 10 minutes), in which case the tool will scan all new log entries
added since the last run, and print out any that match the configured endorsement
keys.

Because the `rekor-monitor` tool scans all Rekor log entries in the range it is
told to look at, and because it verifies the correctness of the sequence of log
entries, it will produce an authoritative list of log entries created by the
configured endorsement keys. But it also means the tool can be quite slow. It
is best suited for actively monitoring for newly created log entries from some
point in time onwards, rather than finding already-created entries.

Rekor also exposes a [REST API](https://www.sigstore.dev/swagger/) and a
[GCP Pub/Sub event stream](https://docs.sigstore.dev/logging/event_stream/)
which you can use to monitor for new log entries instead.

### Finding previously-created entries using Rekor's BigQuery dataset

Since [August 2025](https://blog.sigstore.dev/rekor-bigquery-dataset/) Rekor
also exposes a public BigQuery dataset called `rekor`.  This dataset can be
used to query for previously-created log entries much more conveniently.

The following query can be used to find log entries and hash of the endorsed
payloads, endorsed by one or more endorsement keys:

```sql
/*
Will return one row per endorsement, with an EntrySha256 column containing the
endorsement's SHA256 digest (which can be used to fetch the endorsement
payload), and a KeyDigest specifying which endorsement key endorsed it.
*/
SELECT Entries.*, (
  SELECT Key FROM `bigquery-public-data.rekor.IndexKeys` AS IndexKeys
  WHERE IndexKeys.EntryUUID = Entries.EntryUUID AND Key LIKE 'sha256:%'
  ) AS EntrySha256, KeyDigest
FROM `bigquery-public-data.rekor.Entries` AS Entries
INNER JOIN `bigquery-public-data.rekor.Identities` AS Keys USING (EntryUUID)
INNER JOIN UNNEST(Keys.Digests) AS KeyDigest
-- This queries for application endorsements made by the KMS endorsement key.
WHERE KeyDigest IN ("ea0d1f8ffed9512019a2ec968790263d88ea3324c6a6a782114e5ea1be4fd38f")
ORDER BY IntegratedTime DESC;
```

For example, this is the information that would be returned for the same log
entry as found with the `rekor-tool` above.

|index|EntryUUID|LogIndex|LogID|IntegratedTime|Kind|APIVersion|Size|EntrySha256|KeyDigest|
|---|---|---|---|---|---|---|---|---|---|
|...|108e9186e8c5677a4e8e68884126228c18dbe566ab43eb0be79b077401149bd340bc295d78e9cd86|1634041292|c0d23d6ad406973f9559f3ba2d1ca01f84147d8ffc5b8445c224f98b9591801d|2026-05-26T16:54:44+00:00|rekord|0\.0\.1|2990|sha256:62e818ac02a0427b9811d08468e286aca2f5c68ca44191a9ab1f86075afe5a87|ea0d1f8ffed9512019a2ec968790263d88ea3324c6a6a782114e5ea1be4fd38f|

### Inspecting transparency log entries in detail

For each of the entries matching the keys we're interested in, the full Rekor
log entry can then be accessed via the URL pattern
`https://search.sigstore.dev/?uuid={UUID}`, where `{UUID}` is the 3rd
alphanumeric column value. For example,
https://search.sigstore.dev/?uuid=108e9186e8c5677a4e8e68884126228c18dbe566ab43eb0be79b077401149bd340bc295d78e9cd86
shows the Rekor log entry payload for the container layer, copied below:

```yaml
apiVersion: 0.0.1
kind: rekord
spec:
  data:
    hash:
      algorithm: sha256
      value: 62e818ac02a0427b9811d08468e286aca2f5c68ca44191a9ab1f86075afe5a87
  signature:
    content: >-
      MEUCIQDjUMsYYSBFf5aHTWbCfp4ypXpdxgbix8efLMTOENcihwIgRS9wFZuyXt1iw3mKguUDt7kdVCCTyal2KFzhSEc+gmE=
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
$ curl -sL "https://rekor.sigstore.dev/api/v1/log/entries/108e9186e8c5677a4e8e68884126228c18dbe566ab43eb0be79b077401149bd340bc295d78e9cd86" \
    | jq
{
  "108e9186e8c5677a4e8e68884126228c18dbe566ab43eb0be79b077401149bd340bc295d78e9cd86": {
    "body": "eyJhcGlWZXJz...",
    "integratedTime": 1779814484,
    "logID": "c0d23d6ad406973f9559f3ba2d1ca01f84147d8ffc5b8445c224f98b9591801d",
    "logIndex": 1634041292,
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
$ curl -sL "https://rekor.sigstore.dev/api/v1/log/entries/108e9186e8c5677a4e8e68884126228c18dbe566ab43eb0be79b077401149bd340bc295d78e9cd86" \
    | jq -r '.[] | .body' | base64 -d | jq '.spec.data.hash.value'

"62e818ac02a0427b9811d08468e286aca2f5c68ca44191a9ab1f86075afe5a87"
```

### From transparency log entry hash to published artifact

The SHA2-256 hash in the transparency log entries can point to one of two types
of artifacts, either an
[Oak endorsement](https://github.com/project-oak/oak/blob/main/docs/tr/endorsement_v1.md),
or a
[`SignedPayload` signature structure](https://github.com/google-parfait/federated-compute/blob/main/fcp/protos/confidentialcompute/payload_transparency.proto).
In either case, the artifact specifies the digest of the actual payload being
endorsed, along with an endorsement validity time range which the client will
respect. The payload can be looked up using the following URI pattern:
https://federatedcompute-pa.googleapis.com/data/transparency/sha2-256:{HASH_ON_REKOR}
where `{HASH_ON_REKOR}` is the hash listed in the Rekor log entry.

#### Oak endorsements

The SHA2-256 hash in the transparency log above
`62e818ac02a0427b9811d08468e286aca2f5c68ca44191a9ab1f86075afe5a87` is the hash
of an Oak endorsement statement that can be found at
https://federatedcompute-pa.googleapis.com/data/transparency/sha2-256:62e818ac02a0427b9811d08468e286aca2f5c68ca44191a9ab1f86075afe5a87,
copied below:

```json
{
  "_type": "https://in-toto.io/Statement/v1",
  "predicateType": "https://project-oak.github.io/oak/tr/endorsement/v1",
  "subject": [
    {
      "name": "kms",
      "digest": {
        "sha256": "d14a1852f08b528f9975245f300d44e2500aaff9c8a9c5572e0114b7203e3e47"
      }
    }
  ],
  "predicate": {
    "issuedOn": "2026-05-26T16:54:11.355000Z",
    "validity": {
      "notBefore": "2026-05-26T16:54:11.355000Z",
      "notAfter": "2026-07-10T16:54:11.355000Z"
    }
  }
}
```

You can check for yourself that this endorsement matches the hash in the
transparency log entry, by calculating its hash:

```console
$ curl -sL "https://federatedcompute-pa.googleapis.com/data/transparency/sha2-256:62e818ac02a0427b9811d08468e286aca2f5c68ca44191a9ab1f86075afe5a87" | openssl dgst -sha256
SHA2-256(stdin)= 62e818ac02a0427b9811d08468e286aca2f5c68ca44191a9ab1f86075afe5a87
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

#### SignedPayload structures

As an alternative to an Oak endorsement, the SHA2-256 hash in a transparency log
may be the hash of a `SignedPayload` signature structure. Since `SignedPayload`
signature structures are binary, it is most useful to summarize them using the
`tools/explain_fcp_attestation_record` tool:

```console
$ curl -sL "https://federatedcompute-pa.googleapis.com/data/transparency/sha2-256:090100dac890e96926b9b25e2bb28b04bcfda934596118a9226f696dbebfb346" \
  | bazelisk run //tools/explain_fcp_attestation_record:main -- --signed-payload -
... <snip> ...
Inspecting SignedPayload signature structure provided via stdin.

Downloading attestation evidence from https://federatedcompute-pa.googleapis.com/data/transparency/sha2-256:5cdf572b364e7e95bf6b5ebeb7f86e20e034fcce8b8495122c13a4dbec1f006f.

Oak Containers Stack in a AMD SEV-SNP TEE

_____ Root Layer _____
... <snip> ...
```

### KMS stage0 firmware layer

At the time of writing,
[`108e9186e8c5677abf8065e40dc89b6742de2f811c1dc4a39f2a0d812482305d4ecde0552f34449a`](https://search.sigstore.dev/?uuid=108e9186e8c5677abf8065e40dc89b6742de2f811c1dc4a39f2a0d812482305d4ecde0552f34449a)
is the Rekor UUID of a recent stage0 firmware endorsement, which covers an Oak
endorsement with hash
`d977e83f7759cf975bc4bd2aa03b1a1e94729ca39240b624ad7d356f2eb98d61`, which can be
fetched from
https://federatedcompute-pa.googleapis.com/data/transparency/sha2-256:d977e83f7759cf975bc4bd2aa03b1a1e94729ca39240b624ad7d356f2eb98d61:

```json
{
  "_type": "https://in-toto.io/Statement/v1",
  "predicateType": "https://project-oak.github.io/oak/tr/endorsement/v1",
  "subject": [
    {
      "name": "stage0_bin",
      "digest": {
        "sha256": "55f3d6e965bf4035bbd507d11eed7fde7cd58571e02120e628299010feb31dec"
      }
    }
  ],
  "predicate": {
    "issuedOn": "2026-04-03T02:38:45.775000Z",
    "validity": {
      "notBefore": "2026-04-03T02:38:45.775000Z",
      "notAfter": "2026-07-02T02:38:45.775000Z"
    },
    "claims": [
      {
        "type": "https://github.com/project-oak/oak/blob/main/docs/tr/claim/10271.md"
      },
      {
        "type": "https://github.com/project-oak/oak/blob/main/docs/tr/claim/52637.md"
      },
      {
        "type": "https://github.com/project-oak/oak/blob/main/docs/tr/claim/66738.md"
      },
      {
        "type": "https://github.com/project-oak/oak/blob/main/docs/tr/claim/92939.md"
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
$ curl -sL https://federatedcompute-pa.googleapis.com/data/transparency/sha2-256:55f3d6e965bf4035bbd507d11eed7fde7cd58571e02120e628299010feb31dec \
    | protoc --decode_raw
1 {
  1: 1   # <-- configuration for 1 vCPU
  2 {    # v-- SHA2-384 hash of stage0 binary for this configuration
    32: "43762815304e31c0f6f459cb1c441642ce1f2a29e1e809a722b413097d149ce87101569b86dc913558e6b14e202c3654"
  }
}
1 {
  1: 4
  2 {
    32: "5d86bf09d9ff52e317ca296ee68440cba12cf9e2f73621e1d60c54053ea0a1195ee528828f07dc237d98f69ef78eda9c"
  }
}
1 {
  1: 16
  2 {
    32: "3d320336520aad9f87c9c82f365a221e99f40d728f5d3b69b0ed01012c25f2806ba6f39abd5d47cc3782f8526ce92ef1"
  }
}
1 {
  1: 64
  2 {
    32: "9c443976486e4d6579bf36d40e9b4f104ec0891041c87f1f7b4987f70f93dd27d0618f6b4fac0e787b8f01a03785f79b"
  }
}
2 {
  18: "a8eec309b3d0bb84ef2d632278877c964a820f9e422cb29c81696062e932bf12"
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
$ echo -n "43762815304e31c0f6f459cb1c441642ce1f2a29e1e809a722b413097d149ce87101569b86dc913558e6b14e202c3654" \
    | xxd -r -p \
    | openssl dgst -sha256
SHA2-256(stdin)= 419a4c366156f951aba454422a7e56a8e19d115f68fce70ecbfe8962b47ce39b
```

It is this SHA2-256 hash for which provenance is actually available on Rekor. At
https://search.sigstore.dev/?hash=419a4c366156f951aba454422a7e56a8e19d115f68fce70ecbfe8962b47ce39b
we can find the following excerpt:

```text
Source Repository URI: https://github.com/project-oak/oak
Source Repository Digest: d2aaa34bf4ce2d14cd4a5d38aece851864fdbad0
...
Run Invocation URI: https://github.com/project-oak/oak/actions/runs/23568371228/attempts/1
```

This tells us that this stage0 binary was built from commit
`d2aaa34bf4ce2d14cd4a5d38aece851864fdbad0` in
https://github.com/project-oak/oak. And in the
`build_attest_all(buildconfigs/stage0_bin.sh)` section of the "Run Invocation"
logs we can see:

```text
Attestation created for sha2_384_measurement_of_initial_memory_with_stage0_and_01_vcpu@sha256:419a4c366156f951aba454422a7e56a8e19d115f68fce70ecbfe8962b47ce39b
```

indicating that this was the firmware configuration targeting a single vCPU.


### KMS kernel layer

At the time of writing,
[`108e9186e8c5677a1429d8c2471fb0acdf9733a6da6260352a1c78a42c8b45c1795185c04e7dfa3f`](https://search.sigstore.dev/?uuid=108e9186e8c5677a1429d8c2471fb0acdf9733a6da6260352a1c78a42c8b45c1795185c04e7dfa3f)
is the Rekor UUID of a recent kernel layer endorsement, which covers an Oak
endorsement with hash
covers an Oak endorsement with hash
`f4082858f7a42c8e789f9f975e1f467491c8e0102c8f643882be9b043368cb6c`
which can be fetched from
https://federatedcompute-pa.googleapis.com/data/transparency/sha2-256:f4082858f7a42c8e789f9f975e1f467491c8e0102c8f643882be9b043368cb6c:

```json
{
  "_type": "https://in-toto.io/Statement/v1",
  "predicateType": "https://project-oak.github.io/oak/tr/endorsement/v1",
  "subject": [
    {
      "name": "oak_containers_kernel",
      "digest": {
        "sha256": "7bd05af7a304dca51e0db3a88ab2b2cfdde46c2aaf4701b9b5e77b2c9e236b1e"
      }
    }
  ],
  "predicate": {
    "issuedOn": "2026-04-03T02:38:44.288000Z",
    "validity": {
      "notBefore": "2026-04-03T02:38:44.288000Z",
      "notAfter": "2026-07-02T02:38:44.288000Z"
    },
    "claims": [
      {
        "type": "https://github.com/project-oak/oak/blob/main/docs/tr/claim/22790.md"
      },
      {
        "type": "https://github.com/project-oak/oak/blob/main/docs/tr/claim/52637.md"
      },
      {
        "type": "https://github.com/project-oak/oak/blob/main/docs/tr/claim/92939.md"
      },
      {
        "type": "https://github.com/project-oak/oak/blob/main/docs/tr/claim/98982.md"
      }
    ]
  }
}
```

The subject digest of Oak Containers Kernel layer endorsements is the SHA2-256
hash of another intermediate payload, which is an
`oak.attestation.v1.KernelAttachment`-typed protobuf (see
https://github.com/project-oak/oak/blob/main/docs/tr/claim/98982.md for more
info).

We can fetch this protobuf using the same '/data/transparency' URL pattern at
URL
https://federatedcompute-pa.googleapis.com/data/transparency/sha2-256:7bd05af7a304dca51e0db3a88ab2b2cfdde46c2aaf4701b9b5e77b2c9e236b1e,
and it generally will contain one or more hashes, one per Oak Containers
vCPU configuration. In this case it looks as follows:

```console
$ curl -sL "https://federatedcompute-pa.googleapis.com/data/transparency/sha2-256:7bd05af7a304dca51e0db3a88ab2b2cfdde46c2aaf4701b9b5e77b2c9e236b1e" \
    | protoc --decode_raw
1 { # <--- kernel image
  16: "02000059c00052673152810c5aef4b9253ec4e7e91e5323c5486b93f423d351b6ed1ecd9c8c5"
  17: "d58c3c6c8b15583665c757feb4892d4f54857546"
  18: "3993b37bcb0b5312f21bd9d41ca63a66ac632e8f19e9df6d044fd4db1e95ca1b" # <--- SHA2-256 hash
  19: "a838857265e98e6c24c0bb632ba5794698e91391a6b588affd7016826604294f85ea25e3e12772662acc28b627460af9115fb5265960a4efa3c90f5780e969d6"
  20: "569b7d7d556d18631a88961f5d5e18b3e6aea182014359f7d3bee84851fee537a1aef89b5a1977b726ddc100857ab3a346e0e916b2e3ebba215d44e87406b141"
  21: "39eacbaf9e9eb590bd69f440c434d1a941317d45a4ace9f707451376da4094814dda848d2e223408e69e64a4adaa0957"
  22: "a17b304a16fda8cd667563a25a9b00eb68a65a12f757034e6c6930d38c4bcce6"
  23: "ce86e1814a3b6b60b4652be8e1f95abcf52a547ae01f428b8660ed04"
  32: "0b1f666261129821e5f795854f6fe517eb3e7657702ebad63c30a3d5c1b51e37cb5c575e343b70acc75c9878106ccf07"
}
2 { # <--- kernel setup data
  16: "010040006b5e4a8cbcf237bb1b7883143859d0b80bea9f9baa044aab16d6d95d8c278c28"
  17: "d4cf317b2da68622682e0954b7f74b8d477b0c7a"
  18: "17c499c1515057824d0a1bc5ac9306c19d3cfbc928da2945d518a7addde4bd93" # <--- SHA2-256 hash
  19: "6d891bef776e9a2d966741fcc1e398f1b21bd4b44d1cc58d73a211e8a4349774bd050c3093096b60e1c7d52bcaebb923de063ea2f1c81e67ff1ba7db7a15ddc5"
  20: "73bcd53b80940d3747e39c60c9f4a83c5e54e773287db1683fc523567e13cd47ce20e26177d99e3bffe05ff934b33df379a6ba0b891132605052d4ec50484108"
  21: "4044e2ae34ca7d043bfc3b8e13b5cf3cfd32c1368de45f1c20fea47ce2f8579ad28752d353bf21232efa76163a981948"
  22: "24873e8540eb344b4a184e7daa6c2f33e34c3dbb9ba2ac57440b0bbccabf5897"
  23: "afd1828926069181cb314fdeb785d6e48311428e37610207a4203517"
  32: "946f62006dd6f56a4062c5742ae02148df2d0248940726f25182169ed72935b154b42e5749174dafcbbfee8fa9dd2e9d"
}
3 {
  18: "27be52c964e3c36ce438a74e58b11e3cb08fe38480243160d46819968d8f4446"
}
```

In this wall of text,
`3993b37bcb0b5312f21bd9d41ca63a66ac632e8f19e9df6d044fd4db1e95ca1b` represents
the SHA2-256 hash of the actual kernel image, and
`17c499c1515057824d0a1bc5ac9306c19d3cfbc928da2945d518a7addde4bd93` is the
SHA2-256 hash of the kernel's setup data (they correspond to field 18 of the
[HexDigest](https://github.com/project-oak/oak/blob/25fdba08d2fc1c25c6d4b0a185ed7f3d86d770ba/proto/digest.proto#L51)
protobuf).

It is these SHA2-256 hashes for which provenance is actually available on Rekor.
At
https://search.sigstore.dev/?hash=3993b37bcb0b5312f21bd9d41ca63a66ac632e8f19e9df6d044fd4db1e95ca1b
we can find the following excerpt:

```text
Source Repository URI: https://github.com/project-oak/oak
Source Repository Digest: 9da0a2f445c1d07ceaed33e1bd24be4bbf28f641
...
Run Invocation URI: https://github.com/project-oak/oak/actions/runs/25106261180/attempts/1
```

This tells us that this kernel binary was built from commit
`9da0a2f445c1d07ceaed33e1bd24be4bbf28f641` in
https://github.com/project-oak/oak. And in the
`build_attest_all(buildconfigs/oak_containers_kernel.sh)`
section of the "Run Invocation" logs we can see the corresponding build outputs
being attested:

```text
Attestation created for 3 subjects
...
Attestation uploaded to repository
https://github.com/project-oak/oak/attestations/25943859
```

The subjects are listed at
https://github.com/project-oak/oak/attestations/25943859:

|Subject|Subject digest|
|---|---|
|oak_containers_kernel_setup_data|sha256:17c499c1515057824d0a1bc5ac9306c19d3cfbc928da2945d518a7addde4bd93|
|oak_containers_kernel|sha256:27be52c964e3c36ce438a74e58b11e3cb08fe38480243160d46819968d8f4446|
|oak_containers_kernel_image|sha256:3993b37bcb0b5312f21bd9d41ca63a66ac632e8f19e9df6d044fd4db1e95ca1b|

### KMS init RAM FS layer

At the time of writing,
[`108e9186e8c5677a449460fdc77adf1dc2eead6b42cb895426053aeb5201ffa936318ea0285a5740`](https://search.sigstore.dev/?uuid=108e9186e8c5677a449460fdc77adf1dc2eead6b42cb895426053aeb5201ffa936318ea0285a5740)
is the Rekor UUID of a recent init ram FS layer endorsement, which covers an Oak
endorsement with hash
`a5717475aedbd0d8e9bb9ac66398343a64e0373dcae4e52efe3d59b22be3e5c8`
which can be fetched from
https://federatedcompute-pa.googleapis.com/data/transparency/sha2-256:a5717475aedbd0d8e9bb9ac66398343a64e0373dcae4e52efe3d59b22be3e5c8:

```json
{
  "_type": "https://in-toto.io/Statement/v1",
  "predicateType": "https://project-oak.github.io/oak/tr/endorsement/v1",
  "subject": [
    {
      "name": "oak_containers_stage1",
      "digest": {
        "sha256": "15b89e8182af987bd3d1dc590440b6d3943ce5bcc42233226b541502138df53d"
      }
    }
  ],
  "predicate": {
    "issuedOn": "2026-04-03T02:38:45.323000Z",
    "validity": {
      "notBefore": "2026-04-03T02:38:45.323000Z",
      "notAfter": "2026-07-02T02:38:45.323000Z"
    },
    "claims": [
      {
        "type": "https://github.com/project-oak/oak/blob/main/docs/tr/claim/52637.md"
      },
      {
        "type": "https://github.com/project-oak/oak/blob/main/docs/tr/claim/85483.md"
      },
      {
        "type": "https://github.com/project-oak/oak/blob/main/docs/tr/claim/92939.md"
      }
    ]
  }
}
```

The subject digest of an init ram FS layer endorsements is simply the SHA2-256
hash of the reproducibly buildable Oak Containers Stage 1 binary being endorsed.
Hence, its provenance can be found directly on Rekor. In this case at
https://search.sigstore.dev/?hash=15b89e8182af987bd3d1dc590440b6d3943ce5bcc42233226b541502138df53d,
with the following excerpt:

```text
Source Repository URI: https://github.com/project-oak/oak
Source Repository Digest: 782fb92b980f9c6d2016eb89d1ac87930e420466
...
Run Invocation URI: https://github.com/project-oak/oak/actions/runs/23791749077/attempts/1
```

This tells us that this kernel binary was built from commit
`782fb92b980f9c6d2016eb89d1ac87930e420466` in
https://github.com/project-oak/oak. And in the `build_attest_all
(buildconfigs/oak_containers_stage1.sh)` section of the "Run Invocation" logs we
can see the corresponding build outputs being attested:

```text
Attestation created for stage1.cpio@sha256:15b89e8182af987bd3d1dc590440b6d3943ce5bcc42233226b541502138df53d
```

### KMS system image layer

At the time of writing,
[`108e9186e8c5677ae4f09eb1efa8be9c05165a47dc5656d043d7a291711c8cff391c8f3eecaf27b4`](https://search.sigstore.dev/?uuid=108e9186e8c5677ae4f09eb1efa8be9c05165a47dc5656d043d7a291711c8cff391c8f3eecaf27b4)
is the Rekor UUID of a recent system image layer endorsement, which covers an Oak
endorsement with hash
`30a403e0232b7ed9d141d3b553bfa3ac3ab422e46451d103882ed87c45ee9e7e`
which can be fetched from
https://federatedcompute-pa.googleapis.com/data/transparency/sha2-256:30a403e0232b7ed9d141d3b553bfa3ac3ab422e46451d103882ed87c45ee9e7e:

```json
{
  "_type": "https://in-toto.io/Statement/v1",
  "predicateType": "https://project-oak.github.io/oak/tr/endorsement/v1",
  "subject": [
    {
      "name": "oak_containers_system_image",
      "digest": {
        "sha256": "36040ec0c4470e7f409097cd43c1904894124d58f47dba59a2b9ff167705d018"
      }
    }
  ],
  "predicate": {
    "issuedOn": "2026-04-03T02:38:45.337000Z",
    "validity": {
      "notBefore": "2026-04-03T02:38:45.337000Z",
      "notAfter": "2026-07-02T02:38:45.337000Z"
    },
    "claims": [
      {
        "type": "https://github.com/project-oak/oak/blob/main/docs/tr/claim/52637.md"
      },
      {
        "type": "https://github.com/project-oak/oak/blob/main/docs/tr/claim/58963.md"
      },
      {
        "type": "https://github.com/project-oak/oak/blob/main/docs/tr/claim/92939.md"
      }
    ]
  }
}
```

The subject digest of a system image layer endorsements is simply the SHA2-256
hash of the reproducibly buildable filesystem image being endorsed. Hence, its
provenance can be found directly on Rekor. In this case at
https://search.sigstore.dev/?hash=36040ec0c4470e7f409097cd43c1904894124d58f47dba59a2b9ff167705d018,
with the following excerpt:

```text
Source Repository URI: https://github.com/project-oak/oak
Source Repository Digest: 1cec7c5f0d0e78ee2f5d624438c7835cf34604e7
...
Run Invocation URI: https://github.com/project-oak/oak/actions/runs/23840724104/attempts/1
```

This tells us that this kernel binary was built from commit
`1cec7c5f0d0e78ee2f5d624438c7835cf34604e7` in
https://github.com/project-oak/oak. And in the `build_attest_all
(buildconfigs/oak_containers_system_image.sh)` section of the "Run Invocation" logs we can
see the corresponding build outputs being attested:

```text
Attestation created for oak_containers_system_image.tar.xz@sha256:36040ec0c4470e7f409097cd43c1904894124d58f47dba59a2b9ff167705d018
```

### KMS container layer

The subject digest of container layer endorsements is simply the SHA2-256 hash
of the reproducibly buildable KMS application binary being endorsed.

While all previous binaries we looked at so far are produced from code in the
Oak GitHub project, the KMS application binary is maintained in this
repository. It is this binary that implements the business logic of protecting
data decryption keys in accordance with the data access policies associated with
uploaded encrypted data.

As we showed in the
[earlier section](#from-transparency-log-entry-hash-to-published-artifact), the
container layer endorsement covers subject digest
d14a1852f08b528f9975245f300d44e2500aaff9c8a9c5572e0114b7203e3e47, for which we
can hence find provenance at
https://search.sigstore.dev/?hash=d14a1852f08b528f9975245f300d44e2500aaff9c8a9c5572e0114b7203e3e47,
with the following excerpt:

```
Source Repository URI: https://github.com/google-parfait/confidential-federated-compute
Source Repository Digest: fe439180bb24e65ad46e869eddee88f9c9b34875
...
Run Invocation URI: https://github.com/google-parfait/confidential-federated-compute/actions/runs/26261649715/attempts/1
```

It shows that the `kms` application binary was produced at commit
`fe439180bb24e65ad46e869eddee88f9c9b34875` in the
https://github.com/google-parfait/confidential-federated-compute repository. The
"Run Invocation" logs have more information about the action that produced the
binary, which you can use to reproduce the build steps and arrive at the same
result.

## Inspecting data access policy transparency log entries

The endorsement keys client devices use for validating data access policies
depends on the app running on the device, and per-app endorsement keys can be
found in the [/access_policy_reference_values](/access_policy_reference_values)
directory. For each data access policy endorsement key there is an associated
set of privacy properties that the corresponding app will ensure are upheld by
every endorsed data access policy.

Just as with the KMS binary endorsements, you can use the `rekor-monitor` tool
or the Rekor public BigQuery dataset to find transparency log entries for this
endorsement key. For example, using the SQL query provided earlier with
[Gboard](https://play.google.com/store/apps/details?id=com.google.android.inputmethod.latin)'s
endorsement key of
`3ebd2cd4ec2a56655c9022e734d7469c0a8612f7f676b001d090897f36bae560` would produce
the following row:

|index|EntryUUID|LogIndex|LogID|IntegratedTime|Kind|APIVersion|Size|EntrySha256|KeyDigest|
|---|---|---|---|---|---|---|---|---|---|
|...|108e9186e8c5677af9f63f6c97d4bcaee3210af2da8ae214ebb56b76e2456c2bccdee68bc44c5e48|175189129|c0d23d6ad406973f9559f3ba2d1ca01f84147d8ffc5b8445c224f98b9591801d|2025-02-27 20:24:58+00:00|rekord|0\.0\.1|2577|sha256:d72534e0cae135b3ac11395d37956dbb89b328df4274a28b35cca22500845033|3ebd2cd4ec2a56655c9022e734d7469c0a8612f7f676b001d090897f36bae560|

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
application for which the Oak Containers container layer binary provenance is
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
functionality to limit the endorsements that will be accepted by the KMS during
peer-to-peer attestation, as well as by client devices, to those published to
the transparency log only after a certain absolute date and within a certain
number of days from the current time at the time of verification.

This is reflected in the KMS and data access policies' reference values use of
the `signed_timestamp` field:

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
binaries (KMS and data processing steps) and access policies, anyone can follow
along and validate or falsify the privacy properties of data processed using
confidential federated computations.
