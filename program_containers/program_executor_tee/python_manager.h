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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PYTHON_MANAGER_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PYTHON_MANAGER_H_

#include <condition_variable>
#include <future>
#include <iostream>
#include <queue>
#include <thread>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"

// A PythonTask stores a function to run and the promise that should be set upon
// completion.
struct PythonTask {
  std::function<void()> function;
  std::promise<absl::Status> promise;
};

// The PythonManager guards access to a single execution thread that holds the
// GIL and executes python functions in a FIFO manner from a queue. There is
// one global PythonManager instance per process.
class PythonManager {
 public:
  // Obtain the single global PythonManager instance.
  static PythonManager& GetInstance() {
    static PythonManager instance;
    return instance;
  }

  // Start the execution thread.
  void Start();
  // Stop the execution thread.
  void Stop();

  // Add a function to execute to the queue.
  absl::Status ExecuteTask(std::function<void()> function);

 private:
  PythonManager() = default;
  ~PythonManager() = default;

  // The function that runs on the execution thread and executes functions from
  // the queue in a FIFO manner.
  void ThreadLoop();

  // Whether the execution thread is currently running.
  bool running_ = false;
  std::thread executor_thread_;

  // Mutex guarding the queue since functions can be added to the queue from
  // multiple threads.
  absl::Mutex queue_mutex_;
  // FIFO queue of functions that the execution thread should execute.
  std::queue<PythonTask> task_queue_ ABSL_GUARDED_BY(queue_mutex_);
  // Conditional variable used to make the execution thread sleep until the
  // queue is non-empty.
  absl::CondVar queue_cv_;
};

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_LEARNING_CONTAINERS_PROGRAM_EXECUTOR_TEE_PYTHON_MANAGER_H_