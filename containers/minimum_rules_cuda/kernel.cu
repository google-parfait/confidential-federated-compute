// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Simple cuda kernel which verifies cuda works.
#include <iostream>

#include "kernel.h"
#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    cudaError_t err = (expr);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA Error Code  : %d\n     Error String: %s\n", err, \
              cudaGetErrorString(err));                                      \
      exit(err);                                                             \
    }                                                                        \
  } while (0)
__global__ void kernel() { printf("Cuda kernel called!\n"); }

void launch() {
  kernel<<<1, 1>>>();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}
