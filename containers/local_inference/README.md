# Local inference container

A TEE container that serves the batched inference `ConfidentialTransform` API
using an **in-process** CPU model (`gemma.cpp` or `llama.cpp`), rather than
proxying inference out to a host process the way
[`containers/private_inference`](../private_inference) and
[`containers/gcp`](../gcp) do.

## Structure

The container is the standard batched inference server
([`common/inference`](../common/inference)) wired up to a local model:

*   `BatchedInferenceFn` (shared) owns the whole pipeline: prompt templating,
    batching by `RuntimeConfig.max_batch_size`, output parsing and row
    expansion.
*   `LocalBatchedInferenceEngine` (here) is a thin adapter over
    [`InferenceModel`](../common/local_inference/inference_model.h). It takes
    the already-rendered prompts and returns one raw string per prompt. It must
    not template, parse or expand anything — that would double-apply the work
    the `Fn` already did.
*   `LocalBatchedInferenceEngineProvider` (here) loads the model weights once,
    from the blobs streamed during `StreamInitialize`, and shares that model
    read-only across every `BatchedInferenceFn`.

## Model modes

Only `MODEL_MODE_SHARED_MODEL` is supported. The weights are loaded once and
shared; `gemma.cpp` generation is serialized on the shared model's mutex.
`MODEL_MODE_MODEL_PER_SESSION` is rejected with `UNIMPLEMENTED`.

## Building

This directory is its own Bazel module (see `MODULE.bazel`), so it has to be
built from within it:

```sh
bazel test //...
```
