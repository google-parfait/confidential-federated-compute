# Test Concat Servers


This folder contains a simple test container that can run in a Trusted Execution
Environment that implements the gRPC Pipeline Transform API. The implementation
uses the common cryptographic protocols required for confidential federated
compute containers to access and transform encrypted data. The transformation it
implements is trivial- it simply concatenates the decrypted strings together.
Currently, it supports being configured only once before any data is
transformed.

This container implementation is meant to be used for e2e tests of the
cryptographic protocols used throughout the system.
