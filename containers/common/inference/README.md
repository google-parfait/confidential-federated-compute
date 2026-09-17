# Inference common library

This directory contains runtime-agnostic interfaces, pipeline functions, and
server infrastructure for batched inference within Confidential Federated
Compute containers:

- **`BatchedInferenceEngine` & `BatchedInferenceEngineProvider`**: Abstract
  interfaces defining how a model engine evaluates batches of prompts and how
  engine instances are created.
- **`BatchedInferenceFn`**: Pipeline function integrating prompt templating,
  batching, engine execution, and tabular output serialization.
- **`BatchedInferenceServer`**: gRPC server implementing the Confidential
  Transform API for batched inference workloads.
- **`inference_model_helper`**: Utility functions for parsing model
  configurations, extracting prompts from inputs, and formatting inference
  inputs and outputs.

> Note: `common/inference/` defines the *generic inference pipeline,
> interfaces, and transform server* (independent of any specific LLM runtime or
> execution backend). For concrete, in-process CPU inference implementations
> backed by embedded runtimes (such as Gemma and Llama.cpp), see
> [`common/local_inference/`](../local_inference/).
