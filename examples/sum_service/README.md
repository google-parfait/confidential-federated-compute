# Sum Service

`no_std` compatible implementation of the business logic of a
[`PipelineTransform`](../../pipelines_transforms/proto/pipeline_transform.proto)
that sums inputs.

*   Input: zero or more little-endian `u64`s.
*   Output: a single little-endian `u64`.
