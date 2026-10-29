// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "process_runner.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "reproc/reproc.h"

namespace confidential_federated_compute::gvisor {
namespace {

struct ReprocDeleter {
  void operator()(reproc_t* process) const { reproc_destroy(process); }
};

absl::Status ReprocError(absl::string_view what, int error) {
  return absl::InternalError(absl::StrCat(what, ": ", reproc_strerror(error)));
}

// Kill the child right away; used once the deadline expired and as the stop
// action reproc_destroy() applies to a child that is still running. (A
// function, not a constant: reproc's REPROC_* values are extern variables.)
reproc_stop_actions KillNow() {
  return {
      {REPROC_STOP_KILL, REPROC_INFINITE},
      {REPROC_STOP_NOOP, 0},
      {REPROC_STOP_NOOP, 0},
  };
}

// Reads one chunk of `stream` into `out`, keeping at most `max_output_bytes`
// in total and setting `truncated` if it drops any. Clears `open` at EOF or on
// error. Only one read per poll event, so a child that floods its output can't
// hold off the deadline check.
void ReadChunk(reproc_t* process, REPROC_STREAM stream, std::string& out,
               size_t max_output_bytes, bool& truncated, bool& open) {
  uint8_t buffer[65536];
  int n = reproc_read(process, stream, buffer, sizeof(buffer));
  if (n == REPROC_EWOULDBLOCK) return;
  if (n < 0) {  // REPROC_EPIPE: the child closed the stream.
    open = false;
    return;
  }
  size_t room =
      max_output_bytes > out.size() ? max_output_bytes - out.size() : 0;
  if (static_cast<size_t>(n) > room) truncated = true;
  out.append(reinterpret_cast<const char*>(buffer),
             std::min(static_cast<size_t>(n), room));
}

}  // namespace

absl::StatusOr<ProcessResult> RunProcess(const std::vector<std::string>& argv,
                                         absl::string_view stdin_data,
                                         absl::Duration timeout,
                                         size_t max_output_bytes) {
  if (argv.empty()) {
    return absl::InvalidArgumentError("argv must not be empty");
  }
  std::vector<const char*> c_argv;
  c_argv.reserve(argv.size() + 1);
  for (const std::string& arg : argv) c_argv.push_back(arg.c_str());
  c_argv.push_back(nullptr);

  reproc_options options = {};
  options.redirect.in.type = REPROC_REDIRECT_PIPE;
  options.redirect.out.type = REPROC_REDIRECT_PIPE;
  // reproc sends stderr to the parent's stderr by default, and Oak forwards
  // the container's logs to the untrusted host. Always capture it instead.
  options.redirect.err.type = REPROC_REDIRECT_PIPE;
  // Non-blocking pipes plus reproc_poll() so that a child that stops reading
  // stdin or writing output can't block us past the deadline. (`input` isn't
  // used because reproc requires it to fit in the 64 KiB pipe buffer.)
  options.nonblocking = true;
  // A deadline of 0 means "none" in reproc, so round up to at least 1 ms.
  options.deadline = static_cast<int>(std::clamp<int64_t>(
      absl::ToInt64Milliseconds(timeout), 1, std::numeric_limits<int>::max()));
  options.stop = KillNow();

  std::unique_ptr<reproc_t, ReprocDeleter> process(reproc_new());
  if (process == nullptr) {
    return absl::ResourceExhaustedError("reproc_new failed");
  }
  // The child closes every inherited fd below RLIMIT_NOFILE before exec, and
  // reproc refuses to start if that limit exceeds 2^20 (the bundle uses 1024).
  int r = reproc_start(process.get(), c_argv.data(), options);
  if (r < 0) return ReprocError("reproc_start", r);

  ProcessResult result;
  // Writing to a child that closed its stdin returns REPROC_EPIPE instead of
  // raising SIGPIPE because main() ignores SIGPIPE.
  bool stdin_open = !stdin_data.empty();
  if (!stdin_open) reproc_close(process.get(), REPROC_STREAM_IN);
  bool stdout_open = true;
  bool stderr_open = true;
  size_t written = 0;
  while (stdout_open || stderr_open) {
    int interests = (stdin_open ? REPROC_EVENT_IN : 0) |
                    (stdout_open ? REPROC_EVENT_OUT : 0) |
                    (stderr_open ? REPROC_EVENT_ERR : 0);
    reproc_event_source source = {process.get(), interests, /*events=*/0};
    r = reproc_poll(&source, 1, REPROC_INFINITE);
    if (r == -EINTR) continue;     // reproc errors are negated errno values.
    if (r == REPROC_EPIPE) break;  // No streams left to poll.
    if (r < 0) return ReprocError("reproc_poll", r);
    if (source.events & REPROC_EVENT_DEADLINE) {
      result.timed_out = true;
      break;
    }
    if (source.events & REPROC_EVENT_IN) {
      int n = reproc_write(
          process.get(),
          reinterpret_cast<const uint8_t*>(stdin_data.data()) + written,
          stdin_data.size() - written);
      if (n >= 0) written += static_cast<size_t>(n);
      if ((n < 0 && n != REPROC_EWOULDBLOCK) || written == stdin_data.size()) {
        // Done, or the child closed its stdin: signal EOF either way.
        reproc_close(process.get(), REPROC_STREAM_IN);
        stdin_open = false;
      }
    }
    if (source.events & REPROC_EVENT_OUT) {
      ReadChunk(process.get(), REPROC_STREAM_OUT, result.stdout_data,
                max_output_bytes, result.stdout_truncated, stdout_open);
    }
    if (source.events & REPROC_EVENT_ERR) {
      ReadChunk(process.get(), REPROC_STREAM_ERR, result.stderr_data,
                max_output_bytes, result.stderr_truncated, stderr_open);
    }
  }

  // Both output streams are closed (or the deadline expired). Reap the child,
  // killing it if it is still running at the deadline.
  if (!result.timed_out) {
    do {
      r = reproc_wait(process.get(), REPROC_DEADLINE);
    } while (r == -EINTR);
    result.timed_out = r == REPROC_ETIMEDOUT;
  }
  if (result.timed_out) r = reproc_stop(process.get(), KillNow());
  if (r < 0) return ReprocError("reproc_wait", r);
  // reproc reports 128 + signal number for a child killed by a signal.
  result.exit_code = r;
  return result;
}

}  // namespace confidential_federated_compute::gvisor
