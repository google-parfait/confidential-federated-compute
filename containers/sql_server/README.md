# SQL Server

This folder contains a server implementing the gRPC Pipeline Transform API that
takes a SQL query and input data, creates an in-memory SQLite database, and
executes the given query. The implementation uses the common cryptographic
protocols required for confidential federated compute containers to access and
transform encrypted data.

This container implementation is meant to be used for benchmarking run time
performance.
