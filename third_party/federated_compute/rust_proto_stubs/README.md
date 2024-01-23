Stub (empty) definitions of protobuf messages used by protos in the
federated-compute repository but defined in other cargo crates.

Since it's not easy for one crate to depend on sources (e.g., .proto files) in
another crate, this hack is needed to make cross-crate proto dependencies work.
During compilation, these stubs are replaced by the real cargo types using
`micro_rpc_build::ExternPath`.
