# Square Service

`no_std` compatible implementation of the business logic of a
[`PipelineTransform`](../../pipelines_transforms/proto/pipeline_transform.proto)
that squares inputs.

*   `Initialize` configuration: none.
*   Input: a single little-endian `u64`. If the input is encrypted, the
    `encrypted_symmetric_key_associated_data` must be set and must contain the
    public key to use to encrypt the output.
*   Output: a single little-endian `u64`.
