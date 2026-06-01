# Attestation evidence test data

This directory contains real attestation evidence extracted from one of our applications
running in a real TEE. The binaries at each layer in the attestation evidence should all have SLSA
provenance available on sigstore.dev.

To update this file, extract an new `oak.attestation.v1.Evidence` proto from an instance running in
a real TEE, and serialize it to the binary protobuf format.

## Test certificates

This directory also contains X.509 certificates used for testing https
connections. These certificates were generated using the following commands:

```console
$ openssl req -x509 -noenc -subj "/CN=Test Root CA" -days 36500 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -keyout test_root.key.pem -out test_root.pem
$ openssl req -x509 -noenc -subj /CN=localhost -addext authorityKeyIdentifier=keyid,issuer -addext basicConstraints=CA:FALSE -addext keyUsage=digitalSignature,keyEncipherment -addext extendedKeyUsage=serverAuth -addext "subjectAltName=DNS:*.googleapis.com,DNS:googleapis.com" -days 36500 -CA test_root.pem -CAkey test_root.key.pem -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -keyout test_cert.key.pem -out test_cert.pem
```
