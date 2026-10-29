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

#include "gvisor_do_fn.h"

#include <sys/stat.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "containers/fns/do_fn.h"
#include "containers/fns/fn.h"
#include "containers/fns/fn_factory.h"
#include "containers/session.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "gvisor_report.pb.h"
#include "process_runner.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"

namespace confidential_federated_compute::gvisor {
namespace {

using ::confidential_federated_compute::fns::DoFn;
using ::confidential_federated_compute::fns::Fn;
using ::confidential_federated_compute::fns::FnFactory;
using ::confidential_federated_compute::fns::WriteConfigurationMap;
using ::tensorflow_federated::aggregation::CheckpointBuilder;
using ::tensorflow_federated::aggregation::CheckpointParser;
using ::tensorflow_federated::aggregation::DT_STRING;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::Tensor;

constexpr char kRunscPath[] = "/usr/local/gvisor/runsc";
constexpr char kBsdtarPath[] = "/usr/bin/bsdtar";
constexpr char kTruePath[] = "/bin/true";
constexpr char kWorkDir[] = "/tmp/gvisor";
constexpr absl::Duration kSetupTimeout = absl::Minutes(5);
constexpr absl::Duration kUnitTimeout = absl::Minutes(2);
constexpr size_t kMaxReportedBytes = 16384;
constexpr size_t kMaxUnitOutputBytes = 1 << 20;
// The string tensor that holds the records, in the input and in the output,
// like in other containers (for example sentence_transformers).
// TODO: Take the tensor names from the configuration.
constexpr char kDataTensorName[] = "data";

// OCI runtime spec for the per-unit sandbox. The enforcer writes this itself;
// the customer only provides filesystem contents. No network, read-only root,
// writes go to an in-sandbox tmpfs that disappears with the sandbox. The
// sandbox runs /entrypoint from the customer layer: any executable, or a
// symlink to one.
// TODO: Take the entrypoint from the configuration.
constexpr char kInnerConfigJson[] = R"json({
  "ociVersion": "1.0.0",
  "process": {
    "terminal": false,
    "user": {"uid": 0, "gid": 0},
    "args": ["/entrypoint"],
    "env": ["PATH=/usr/bin:/bin", "HOME=/tmp", "TMPDIR=/tmp"],
    "cwd": "/tmp",
    "noNewPrivileges": true
  },
  "root": {"path": "rootfs", "readonly": true},
  "hostname": "unit",
  "mounts": [
    {"destination": "/proc", "type": "proc", "source": "proc"},
    {"destination": "/tmp", "type": "tmpfs", "source": "tmpfs",
     "options": ["nosuid", "nodev", "mode=1777"]}
  ],
  "linux": {
    "namespaces": [
      {"type": "pid"}, {"type": "network"}, {"type": "ipc"},
      {"type": "uts"}, {"type": "mount"}
    ]
  }
}
)json";

std::string Tail(absl::string_view s, size_t n) {
  return std::string(s.size() <= n ? s : s.substr(s.size() - n));
}

// Single-line text format, for logs and error messages.
std::string ToText(const google::protobuf::Message& message) {
  google::protobuf::TextFormat::Printer printer;
  printer.SetSingleLineMode(true);
  std::string text;
  printer.PrintToString(message, &text);
  return text;
}

std::string ReadFileForReport(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return absl::StrCat("<unreadable: ", path, ">");
  std::stringstream buffer;
  buffer << in.rdbuf();
  return Tail(buffer.str(), kMaxReportedBytes);
}

// Keeps only the lines of `text` that contain one of `needles`.
std::string FilterLines(absl::string_view text,
                        const std::vector<absl::string_view>& needles) {
  std::vector<absl::string_view> kept;
  for (absl::string_view line : absl::StrSplit(text, '\n')) {
    for (absl::string_view needle : needles) {
      if (absl::StrContains(line, needle)) {
        kept.push_back(line);
        break;
      }
    }
  }
  return absl::StrJoin(kept, "\n");
}

ProcessReport ToProcessReport(const absl::StatusOr<ProcessResult>& result) {
  ProcessReport report;
  if (!result.ok()) {
    report.set_spawn_error(result.status().ToString());
    return report;
  }
  report.set_exit_code(result->exit_code);
  report.set_timed_out(result->timed_out);
  report.set_stdout_tail(Tail(result->stdout_data, kMaxReportedBytes));
  report.set_stderr_tail(Tail(result->stderr_data, kMaxReportedBytes));
  return report;
}

// Facts about the Oak container that decide whether gVisor can nest inside it.
void CollectEnvironment(SetupReport& setup) {
  utsname uts{};
  uname(&uts);
  std::string status = ReadFileForReport("/proc/self/status");
  std::string mountinfo = ReadFileForReport("/proc/self/mountinfo");
  auto& env = *setup.mutable_environment();
  env["uname"] = absl::StrCat(uts.sysname, " ", uts.release, " ", uts.version,
                              " ", uts.machine);
  env["uid_map"] = ReadFileForReport("/proc/self/uid_map");
  env["gid_map"] = ReadFileForReport("/proc/self/gid_map");
  env["max_user_namespaces"] =
      ReadFileForReport("/proc/sys/user/max_user_namespaces");
  env["unprivileged_userns_clone"] =
      ReadFileForReport("/proc/sys/kernel/unprivileged_userns_clone");
  env["ptrace_scope"] = ReadFileForReport("/proc/sys/kernel/yama/ptrace_scope");
  env["status"] =
      FilterLines(status, {"Uid:", "Gid:", "Cap", "NoNewPrivs", "Seccomp"});
  env["proc_and_sys_mounts"] = FilterLines(mountinfo, {"proc", "sysfs", " / "});
  env["meminfo"] = FilterLines(ReadFileForReport("/proc/meminfo"),
                               {"MemTotal", "MemAvailable"});
  env["cpu_count"] = std::to_string(sysconf(_SC_NPROCESSORS_ONLN));
  env["dev_kvm_exists"] = access("/dev/kvm", F_OK) == 0 ? "true" : "false";
}

// Returns the records in the input checkpoint's `data` tensor.
absl::StatusOr<std::vector<std::string>> ReadRecords(std::string checkpoint) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<CheckpointParser> parser,
                        FederatedComputeCheckpointParserFactory().Create(
                            absl::Cord(std::move(checkpoint))));
  ABSL_ASSIGN_OR_RETURN(Tensor records, parser->GetTensor(kDataTensorName));
  if (records.dtype() != DT_STRING) {
    return absl::InvalidArgumentError(
        absl::StrCat("The `", kDataTensorName, "` tensor must hold strings."));
  }
  return records.ToStringVector();
}

// Returns a checkpoint whose `data` tensor holds `records`.
absl::StatusOr<std::string> WriteRecords(std::vector<std::string> records) {
  std::unique_ptr<CheckpointBuilder> builder =
      FederatedComputeCheckpointBuilderFactory().Create();
  ABSL_RETURN_IF_ERROR(
      builder->Add(kDataTensorName, Tensor(std::move(records))));
  ABSL_ASSIGN_OR_RETURN(absl::Cord checkpoint, builder->Build());
  return std::string(checkpoint);
}

// Global runsc flags shared by every run. systrap because KVM is not available
// inside a confidential VM; cgroups are owned by the outer runtime. No
// --rootless: the process is already root in the container's user namespace,
// and runsc creates its own nested user namespace for the gofer and sandbox.
std::vector<std::string> RunscBaseArgs(const std::string& state_dir) {
  return {kRunscPath, absl::StrCat("--root=", state_dir), "--network=none",
          "--platform=systrap", "--ignore-cgroups"};
}

// The sandbox environment shared by all DoFn instances. Thread-safe: every
// run uses a unique container ID.
class GvisorRunner {
 public:
  static absl::StatusOr<std::shared_ptr<GvisorRunner>> Create(
      const std::string& layer_path) {
    auto runner = std::shared_ptr<GvisorRunner>(new GvisorRunner());
    ABSL_RETURN_IF_ERROR(runner->Setup(layer_path));
    return runner;
  }

  // Runs the program once in a fresh sandbox, with `input` as its stdin.
  // Returns its stdout if it exited with status 0 before the deadline and
  // didn't write more than the output limit.
  std::optional<std::string> RunUnit(absl::string_view input) {
    std::string id = absl::StrCat("unit-", next_id_.fetch_add(1));
    std::vector<std::string> argv = RunscBaseArgs(state_dir_);
    argv.insert(argv.end(),
                {"run", absl::StrCat("--bundle=", bundle_dir_), id});
    absl::StatusOr<ProcessResult> result =
        RunProcess(argv, input, kUnitTimeout, kMaxUnitOutputBytes);
    // Best-effort cleanup of any leftover sandbox state.
    std::vector<std::string> delete_argv = RunscBaseArgs(state_dir_);
    delete_argv.insert(delete_argv.end(), {"delete", "--force", id});
    (void)RunProcess(delete_argv, "", absl::Seconds(30));

    if (!result.ok() || result->exit_code != 0 || result->timed_out ||
        result->stdout_truncated) {
      return std::nullopt;
    }
    return std::move(result->stdout_data);
  }

 private:
  GvisorRunner() = default;

  absl::Status Setup(const std::string& layer_path) {
    bundle_dir_ = absl::StrCat(kWorkDir, "/bundle");
    state_dir_ = absl::StrCat(kWorkDir, "/runsc-state");
    std::string rootfs_dir = absl::StrCat(bundle_dir_, "/rootfs");
    std::error_code ec;
    std::filesystem::create_directories(rootfs_dir, ec);
    if (ec) return absl::InternalError("create rootfs dir: " + ec.message());
    std::filesystem::create_directories(state_dir_, ec);
    if (ec) return absl::InternalError("create state dir: " + ec.message());

    SetupReport setup;
    CollectEnvironment(setup);
    LOG(INFO) << "gVisor environment: " << ToText(setup);

    absl::Time start = absl::Now();
    // bsdtar creates every directory owner-writable and applies the archived
    // modes at the end, so read-only (e.g. 0555) directories unpack without
    // CAP_DAC_OVERRIDE. Restore only modes and times: ownership stays with the
    // container's root, and xattrs, ACLs and file flags are skipped. bsdtar
    // refuses ".." paths and won't extract through symlinks by default.
    absl::StatusOr<ProcessResult> unpack =
        RunProcess({kBsdtarPath, "--no-same-owner", "--no-xattrs", "--no-acls",
                    "--no-fflags", "-xf", layer_path, "-C", rootfs_dir},
                   "", kSetupTimeout);
    setup.set_unpack_seconds(absl::ToDoubleSeconds(absl::Now() - start));
    if (!unpack.ok() || unpack->exit_code != 0) {
      return absl::InternalError(
          absl::StrCat("Unpacking the rootfs layer failed: ",
                       ToText(ToProcessReport(unpack))));
    }
    std::ofstream config(absl::StrCat(bundle_dir_, "/config.json"));
    config << kInnerConfigJson;
    config.close();
    if (!config) return absl::InternalError("Failed to write config.json");

    // Probes that isolate "can runsc start at all" from bundle problems.
    *setup.mutable_runsc_version() = ToProcessReport(
        RunProcess({kRunscPath, "--version"}, "", absl::Seconds(30)));
    std::vector<std::string> do_argv = RunscBaseArgs(state_dir_);
    do_argv.insert(do_argv.end(), {"do", kTruePath});
    *setup.mutable_runsc_do_true() =
        ToProcessReport(RunProcess(do_argv, "", absl::Seconds(60)));

    std::error_code size_ec;
    uintmax_t layer_size = std::filesystem::file_size(layer_path, size_ec);
    setup.set_layer_bytes(size_ec ? 0 : layer_size);
    setup.clear_environment();  // Already logged above.
    LOG(INFO) << "gVisor setup: " << ToText(setup);
    return absl::OkStatus();
  }

  std::string bundle_dir_;
  std::string state_dir_;
  std::atomic<int64_t> next_id_{0};
};

class GvisorDoFn : public DoFn {
 public:
  explicit GvisorDoFn(std::shared_ptr<GvisorRunner> runner)
      : runner_(std::move(runner)) {}

  absl::Status Do(KV input, DoContext& context) override {
    // Runs the program once per record. The records and the program's outputs
    // are opaque bytes: what they mean is up to the program and the stages
    // around this one. A run that fails contributes no output record.
    ABSL_ASSIGN_OR_RETURN(std::vector<std::string> records,
                          ReadRecords(std::move(input.data)));
    std::vector<std::string> outputs;
    outputs.reserve(records.size());
    for (const std::string& record : records) {
      std::optional<std::string> output = runner_->RunUnit(record);
      context.IncrementCounter(output.has_value() ? "gvisor-unit-ok"
                                                  : "gvisor-unit-failed");
      if (output.has_value()) outputs.push_back(*std::move(output));
    }
    ABSL_ASSIGN_OR_RETURN(std::string checkpoint,
                          WriteRecords(std::move(outputs)));
    // The output is derived from the input, so it must not leave the TEE in
    // plaintext: encrypt it for the pipeline's next node.
    if (!context.EmitEncrypted(/*reencryption_key_index=*/0,
                               KV(std::move(input.key), std::move(checkpoint),
                                  std::move(input.blob_id)))) {
      return absl::FailedPreconditionError(
          "Failed to emit the encrypted output.");
    }
    return absl::OkStatus();
  }

 private:
  std::shared_ptr<GvisorRunner> runner_;
};

class GvisorDoFnFactory : public FnFactory {
 public:
  explicit GvisorDoFnFactory(std::shared_ptr<GvisorRunner> runner)
      : runner_(std::move(runner)) {}

  absl::StatusOr<std::unique_ptr<Fn>> CreateFn() const override {
    return std::make_unique<GvisorDoFn>(runner_);
  }

 private:
  std::shared_ptr<GvisorRunner> runner_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<FnFactory>> ProvideGvisorDoFnFactory(
    const google::protobuf::Any& configuration,
    const google::protobuf::Any& /*config_constraints*/,
    const WriteConfigurationMap& write_configuration_map) {
  if (!configuration.type_url().empty()) {
    return absl::InvalidArgumentError(
        "The gVisor container doesn't take a configuration yet.");
  }
  auto it = write_configuration_map.find(kRootfsLayerConfigId);
  if (it == write_configuration_map.end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Missing configuration blob '", kRootfsLayerConfigId,
                     "' with the rootfs layer tar."));
  }
  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<GvisorRunner> runner,
                        GvisorRunner::Create(it->second));
  return std::make_unique<GvisorDoFnFactory>(std::move(runner));
}

}  // namespace confidential_federated_compute::gvisor
