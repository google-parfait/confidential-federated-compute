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
#include "python_manager.h"

#include <pybind11/embed.h>

#include <chrono>

#include "absl/status/status.h"

void PythonManager::Start() {
  if (running_) return;
  running_ = true;
  executor_thread_ = std::thread(&PythonManager::ThreadLoop, this);
}

void PythonManager::Stop() {
  if (!running_) return;
  running_ = false;
  queue_cv_.Signal();  // Wake up the thread if it's waiting
  if (executor_thread_.joinable()) {
    executor_thread_.join();
  }
}

void PythonManager::ThreadLoop() {
  pybind11::scoped_interpreter interpreter;

  while (running_) {
    PythonTask task;
    {
      absl::MutexLock lock(&queue_mutex_);

      // Wait until a task is available or shutdown is requested.
      while (task_queue_.empty() && running_) {
        queue_cv_.Wait(&queue_mutex_);  // Wait until notified
      }

      if (!running_ && task_queue_.empty()) {
        return;
      }

      task = std::move(task_queue_.front());
      task_queue_.pop();
    }

    // Execute the task.
    try {
      task.function();
      task.promise.set_value(absl::OkStatus());
    } catch (const std::exception& e) {
      task.promise.set_value(
          absl::InternalError("PythonManager hit exception executing task: " +
                              std::string(e.what())));
    }
  }
}

absl::Status PythonManager::ExecuteTask(std::function<void()> function) {
  std::promise<absl::Status> promise;
  std::future<absl::Status> future = promise.get_future();

  // Add the task to the queue.
  PythonTask task = {.function = std::move(function),
                     .promise = std::move(promise)};
  {
    absl::MutexLock lock(&queue_mutex_);
    task_queue_.push(std::move(task));
  }
  queue_cv_.Signal();  // Wake up the thread

  // Wait for the task to execute before returning.
  return future.get();
}