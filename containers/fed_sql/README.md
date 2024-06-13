# FedSQL Server

This folder contains a FedSQL container that can run in a Trusted Execution
Environment that implements the gRPC Confidential Transform API. The
implementation uses the common cryptographic protocols required for confidential
federated compute containers to access and transform encrypted data. The
container can be used to aggregate hierarachically using both SQLite and
Aggregation Cores. Depending on the request configurations, the container can
either

1.  Accumulate inputs

Input: client contributions.

 The container executes a per-client SQLite query,
and accumulates the query results using an Aggregation Core. This is the leaf
step of hierarchical aggregation.

2.  Merge inputs

Input: serialized Aggregation Core state.

 The container merges the input Aggregation Core with its Aggregation Core.

3.  Report serialized results.

Output: serialized Aggregation Core state.

This will be encrypted if any of the inputs from step 1. and 2. are encrypted.
This is the mid-level step of hierarchical aggregation.

4.  Report aggregated results

Output: aggregated results.

This is the root step of hierarchical aggregation.

This container implementation is meant to be used for Private Analytics Flume
pipelines.
