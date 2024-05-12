# Attestation evidence test data

This directory contains real attestation evidence extracted from one of our ledger applications
running in a real TEE. The binaries at each layer in the attestation evidence should all have SLSA
provenance available on sigstore.dev.

To update this file, extract an new `oak.attestation.v1.Evidence` proto from an instance running in
a real TEE, and serialize it to the binary protobuf format.