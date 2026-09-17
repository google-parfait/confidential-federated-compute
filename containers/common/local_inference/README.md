# Local (in-process CPU) inference library

This directory contains an in-process CPU inference engine implementation via
**`InferenceModel`**, a concrete model engine backed by in-process C++ inference
runtimes (`gemma.cpp` and `llama.cpp`). It manages in-memory model weights and
tokenizer loading, and executes CPU inference on batches of prompts.

`InferenceModel` is a pure generation engine: it takes already-rendered prompts
and returns one raw string per prompt. Prompt templating, output parsing and
row expansion are the responsibility of its callers —
[`fed_sql::InferenceRunner`](../../fed_sql/inference_runner.h) for the fed_sql
container and [`BatchedInferenceFn`](../inference/batched_inference_fn.cc) for
the local_inference container.

This code is currently used by both [`containers/fed_sql`](../../fed_sql) and
[`containers/local_inference`](../../local_inference).

> Note: `common/local_inference/` provides the *concrete CPU
> execution engine* (`InferenceModel`) using embedded C++ runtimes. For generic
> inference abstractions, batching pipeline, and gRPC transform server
> interfaces, see [`common/inference/`](../inference/).
