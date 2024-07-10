# Test Concat Servers

This folder contains simple test containers that can run in a Trusted Execution
Environment. One implements the gRPC Pipeline Transform API and the other
implements the gRPC Confidential Transform API. Both implementations uses the
common cryptographic protocols required for confidential federated compute
containers to access and transform encrypted data. The transformation they both
implement is trivial- they simply concatenates the decrypted strings together.
Currently, they support being configured only once before any data is
transformed.

These container implementations are meant to be used for e2e tests of the
cryptographic protocols used throughout the system.
