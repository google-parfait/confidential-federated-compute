// Copyright 2026 Google LLC.
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

#include "program_executor_tee/program_context/cc/elastic_composing_executor.h"

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "federated_language/proto/computation.pb.h"
#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
// Textually include composing_executor.cc to access its file-scoped types
// (ExecutorValue, UnplacedInner, etc.) which are defined in an anonymous
// namespace and not accessible via the public header.  The guard prevents
// the base-class factory function from being redefined.
#define COMPOSING_EXECUTOR_INCLUDED_AS_HEADER
#include "tensorflow_federated/cc/core/impl/executors/composing_executor.cc"
#include "tensorflow_federated/cc/core/impl/executors/computations.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/federated_intrinsics.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/threading.h"
#include "tensorflow_federated/cc/core/impl/executors/value_validation.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::program_executor_tee {
namespace {

using ::tensorflow_federated::CentralClients;
using ::tensorflow_federated::ClientsData;
using ::tensorflow_federated::ComposingChild;
using ::tensorflow_federated::Executor;
using ::tensorflow_federated::ExecutorBase;
using ::tensorflow_federated::ExecutorValue;
using ::tensorflow_federated::FederatedKind;
using ::tensorflow_federated::kClientsUri;
using ::tensorflow_federated::kFederatedAggregateUri;
using ::tensorflow_federated::kFederatedMapAtClientsUri;
using ::tensorflow_federated::kServerUri;
using ::tensorflow_federated::OwnedValueId;
using ::tensorflow_federated::ParallelTasks;
using ::tensorflow_federated::ShareValueId;
using ::tensorflow_federated::ThreadPool;
using ::tensorflow_federated::ValueFuture;
namespace v0 = ::tensorflow_federated::v0;

// Thread-safe queue for elastic distribution of work items.
// Workers pop items from the front. If a worker fails, the items it was
// processing are re-queued so that a different worker can pick them up.
class WorkQueue {
 public:
  // Limits the number of times a specific work item is retried. This prevents
  // infinite loops when an item causes deterministic failures (e.g., poisoned
  // data). When this limit is hit, the item is dropped and ExceededRetries()
  // returns true, causing the overall operation to fail. Retries naturally
  // occur on different workers because a worker permanently marks itself as
  // failed when it encounters an error, leaving re-queued items for the
  // remaining healthy workers.
  static constexpr int32_t kMaxRetries = 4;

  explicit WorkQueue(int32_t num_items, int32_t num_workers)
      : active_workers_(num_workers), retry_counts_(num_items, 0) {
    for (int32_t i = 0; i < num_items; ++i) {
      items_.push_back(i);
    }
  }

  // Pops up to `max_batch_size` items from the queue.  If the queue is empty
  // and other workers are still active, blocks until work appears or all
  // workers finish.  Returns an empty vector when all work is done.
  //
  // active_workers_ tracks how many workers are currently processing items.
  // When a worker finds the queue empty, it goes idle (decrements
  // active_workers_) and waits.  If new work appears (via RequeueAndMarkFailed
  // from another worker), it re-activates and continues.  When the queue is
  // empty and active_workers_ == 0, all waiting workers wake up and return {}.
  std::vector<int32_t> PopBatchOrWait(int32_t max_batch_size) {
    absl::MutexLock lock(mu_);
    while (items_.empty()) {
      --active_workers_;
      cv_.SignalAll();
      while (items_.empty() && active_workers_ > 0) {
        cv_.Wait(&mu_);
      }
      if (items_.empty()) return {};  // All workers idle, no work left.
      ++active_workers_;
    }
    int32_t count =
        std::min(max_batch_size, static_cast<int32_t>(items_.size()));
    std::vector<int32_t> batch(items_.begin(), items_.begin() + count);
    items_.erase(items_.begin(), items_.begin() + count);
    return batch;
  }

  // Requeues the given items (if they haven't exceeded kMaxRetries) and
  // permanently removes this worker from the active pool.  Call with an
  // empty vector when only the worker failure (no requeue) is needed.
  void RequeueAndMarkFailed(const std::vector<int32_t>& items) {
    absl::MutexLock lock(mu_);
    for (int32_t item : items) {
      if (++retry_counts_[item] <= kMaxRetries) {
        items_.push_back(item);
      } else {
        exceeded_retries_ = true;
      }
    }
    --active_workers_;
    cv_.SignalAll();
  }

  bool Empty() const {
    absl::MutexLock lock(mu_);
    return items_.empty();
  }

  bool ExceededRetries() const {
    absl::MutexLock lock(mu_);
    return exceeded_retries_;
  }

 private:
  mutable absl::Mutex mu_;
  absl::CondVar cv_;
  // Number of workers currently processing items.
  int32_t active_workers_ ABSL_GUARDED_BY(mu_);
  // Pending work items (indices).
  std::deque<int32_t> items_ ABSL_GUARDED_BY(mu_);
  // Per-item retry counts to bound retries on deterministic failures.
  std::vector<int32_t> retry_counts_ ABSL_GUARDED_BY(mu_);
  // Set to true if any item exceeded kMaxRetries.
  bool exceeded_retries_ ABSL_GUARDED_BY(mu_) = false;
};

// Dynamically distributes per-client work across child worker executors using
// a shared work queue, rather than statically partitioning clients across
// children at value-creation time. Client data is stored centrally (as
// CentralClients) rather than being distributed to children at creation time,
// enabling dynamic redistribution when workers fail.
//
// Key differences from ComposingExecutor:
//   - Batch-based distribution: clients are grouped into batches and sent as
//     federated intrinsics to workers, amortising RPC overhead.
//   - Dynamic load balancing: faster workers pick up more batches.
//   - Fault tolerance: if a child executor fails mid-round, its unfinished
//     batches are re-queued and picked up by remaining healthy children.
//
// Types (UnplacedInner, ExecutorValue, etc.) are shared with ComposingExecutor
// via textual inclusion of composing_executor.cc.
class ElasticComposingExecutor
    : public tensorflow_federated::ComposingExecutor {
 public:
  ElasticComposingExecutor(std::shared_ptr<Executor> server,
                           std::vector<ComposingChild> children,
                           int32_t total_clients, int32_t num_children,
                           int32_t avg_batches_per_worker = 10)
      : ComposingExecutor(std::move(server), std::move(children), total_clients,
                          /*threadpool_size=*/
                          4 * std::max(static_cast<int32_t>(
                                           std::thread::hardware_concurrency()),
                                       num_children)),
        avg_batches_per_worker_(avg_batches_per_worker) {}

  ~ElasticComposingExecutor() override { ClearTracked(); }

 protected:
  absl::string_view ExecutorName() override {
    static constexpr absl::string_view kExecutorName =
        "ElasticComposingExecutor";
    return kExecutorName;
  }

 private:
  // Stores CLIENTS data centrally rather than distributing to children,
  // so that client work can be dynamically redistributed across workers
  // on failure or load imbalance.
  absl::StatusOr<ExecutorValue> CreateFederatedValue(
      FederatedKind kind, const v0::Value_Federated& federated) override {
    switch (kind) {
      case FederatedKind::SERVER: {
        return ComposingExecutor::CreateFederatedValue(kind, federated);
      }
      case FederatedKind::CLIENTS: {
        auto data = std::make_shared<ClientsData>();
        data->all_equal = false;
        data->values.reserve(federated.value_size());
        for (const auto& v : federated.value()) {
          data->values.push_back(v);
        }
        return ExecutorValue::CreateClientsPlaced(
            CentralClients(std::move(data)));
      }
      case FederatedKind::CLIENTS_ALL_EQUAL: {
        auto data = std::make_shared<ClientsData>();
        data->all_equal = true;
        data->values.push_back(federated.value(0));
        return ExecutorValue::CreateClientsPlaced(
            CentralClients(std::move(data)));
      }
    }
  }

  absl::Status MaterializeValue(const ExecutorValue& value, v0::Value* value_pb,
                                ParallelTasks& tasks) const override {
    if (value.type() == ExecutorValue::ValueType::CLIENTS) {
      if (!value.is_clients_central()) {
        return absl::InternalError(
            "ElasticComposingExecutor: expected central clients value.");
      }
      auto* fed = value_pb->mutable_federated();
      auto* type = fed->mutable_type();
      type->mutable_placement()->mutable_value()->mutable_uri()->assign(
          kClientsUri.data(), kClientsUri.size());
      const auto& data = value.clients_central();
      if (data->all_equal) {
        type->set_all_equal(true);
        *fed->add_value() = data->values[0];
      } else {
        type->set_all_equal(false);
        for (const auto& v : data->values) {
          *fed->add_value() = v;
        }
      }
      return absl::OkStatus();
    }
    return ComposingExecutor::MaterializeValue(value, value_pb, tasks);
  }

  absl::StatusOr<ExecutorValue> CallIntrinsicValueAtClients(
      ExecutorValue&& arg) override {
    if (arg.type() != ExecutorValue::ValueType::UNPLACED) {
      return absl::InvalidArgumentError(
          "federated_value_at_clients: arg must be unplaced.");
    }
    auto data = std::make_shared<ClientsData>();
    data->all_equal = true;
    data->values.push_back(*TFF_TRY(arg.unplaced()->Proto(*server_)));
    return ExecutorValue::CreateClientsPlaced(CentralClients(std::move(data)));
  }

  absl::StatusOr<ExecutorValue> CallIntrinsicBroadcast(
      ExecutorValue&& arg) override {
    if (arg.type() != ExecutorValue::ValueType::SERVER) {
      return absl::InvalidArgumentError(
          "Attempted to broadcast a value not placed at server.");
    }
    v0::Value server_val;
    TFF_TRY(server_->Materialize(arg.server()->ref(), &server_val));
    auto data = std::make_shared<ClientsData>();
    data->all_equal = true;
    data->values.push_back(std::move(server_val));
    return ExecutorValue::CreateClientsPlaced(CentralClients(std::move(data)));
  }

  // Build a federated {CLIENTS} value proto from a subset of client data
  // identified by index.  For all_equal data, emits a single value with
  // all_equal=true.  For non-all_equal data, emits one value per index.
  static v0::Value BuildFederatedClientsValue(
      const ClientsData& client_data,
      const std::vector<int32_t>& client_indices) {
    v0::Value fed;
    auto* federated = fed.mutable_federated();
    federated->mutable_type()
        ->mutable_placement()
        ->mutable_value()
        ->mutable_uri()
        ->assign(kClientsUri.data(), kClientsUri.size());
    if (client_data.all_equal) {
      federated->mutable_type()->set_all_equal(true);
      if (!client_data.values.empty()) {
        *federated->add_value() = client_data.values[0];
      }
    } else {
      federated->mutable_type()->set_all_equal(false);
      for (int32_t idx : client_indices) {
        *federated->add_value() = client_data.values[idx];
      }
    }
    return fed;
  }

  int32_t batch_size(int32_t num_clients) const {
    return std::max(
        1, num_clients / static_cast<int32_t>(children_.size() *
                                              avg_batches_per_worker_));
  }

  absl::StatusOr<v0::Value> ZipClientProtos(const ExecutorValue& arg,
                                            int32_t client_index) {
    switch (arg.type()) {
      case ExecutorValue::ValueType::CLIENTS: {
        if (!arg.is_clients_central()) {
          return absl::InvalidArgumentError(
              "ElasticComposingExecutor: expected central clients value in "
              "federated_zip_at_clients.");
        }
        const auto& data = arg.clients_central();
        if (data->all_equal) {
          return data->values.empty() ? v0::Value() : data->values[0];
        }
        if (client_index >= static_cast<int32_t>(data->values.size())) {
          return absl::InvalidArgumentError(
              absl::StrCat("ZipClientProtos index out of bounds: index ",
                           client_index, " vs size ", data->values.size()));
        }
        return data->values[client_index];
      }
      case ExecutorValue::ValueType::STRUCTURE: {
        v0::Value struct_val;
        for (const auto& elem : *arg.structure()) {
          *struct_val.mutable_struct_()->add_element()->mutable_value() =
              TFF_TRY(ZipClientProtos(elem, client_index));
        }
        return struct_val;
      }
      default:
        return absl::InvalidArgumentError(
            "federated_zip_at_clients: non-clients element.");
    }
  }

  absl::StatusOr<ExecutorValue> CallIntrinsicZipAtClients(
      ExecutorValue&& arg,
      const federated_language::FunctionType& type_pb) override {
    auto result = std::make_shared<ClientsData>();
    result->all_equal = false;
    result->values.resize(total_clients_);
    for (int32_t i = 0; i < total_clients_; i++) {
      result->values[i] = TFF_TRY(ZipClientProtos(arg, i));
    }
    return ExecutorValue::CreateClientsPlaced(
        CentralClients(std::move(result)));
  }

  // Distributes `num_work_units` items across all child worker executors using
  // a shared WorkQueue. Each worker pops batches, processes them via
  // `worker_fn`, and re-queues items on failure for other workers to retry.
  // The thread pool is initialized by the base class to
  // hardware_concurrency()*4, which is always >= children_.size().
  absl::Status RunOnWorkers(
      int32_t num_work_units,
      const std::function<absl::Status(Executor* child, WorkQueue& queue,
                                       int32_t batch_size)>& worker_fn,
      absl::string_view intrinsic_name) {
    WorkQueue queue(num_work_units, static_cast<int32_t>(children_.size()));
    int32_t bs = batch_size(num_work_units);

    {
      ParallelTasks tasks(&thread_pool_);
      for (size_t w = 0; w < children_.size(); w++) {
        TFF_TRY(tasks.add_task([&, w]() -> absl::Status {
          return worker_fn(children_[w].executor().get(), queue, bs);
        }));
      }
      TFF_TRY(tasks.WaitAll());
    }

    if (queue.ExceededRetries()) {
      return absl::UnavailableError(
          absl::StrCat(intrinsic_name, ": Items exceeded maximum retry count (",
                       WorkQueue::kMaxRetries, ")."));
    }
    if (!queue.Empty()) {
      return absl::UnavailableError(absl::StrCat(
          intrinsic_name, ": All workers failed to complete execution."));
    }
    return absl::OkStatus();
  }

  absl::StatusOr<ExecutorValue> CallIntrinsicMap(
      ExecutorValue&& arg,
      const federated_language::FunctionType& type_pb) override {
    TFF_TRY(arg.CheckLenForUseAsArgument("federated_map", 2));
    const auto& fn_val = arg.structure()->at(0);
    const auto& data_val = arg.structure()->at(1);

    if (data_val.type() == ExecutorValue::ValueType::SERVER) {
      auto fn_id = TFF_TRY(fn_val.Embed(*server_));
      auto res =
          TFF_TRY(server_->CreateCall(fn_id->ref(), data_val.server()->ref()));
      return ExecutorValue::CreateServerPlaced(ShareValueId(std::move(res)));
    }
    if (!data_val.is_clients_central()) {
      return absl::InternalError(
          "ElasticComposingExecutor: expected central clients value in "
          "federated_map.");
    }
    const auto& client_data = data_val.clients_central();
    auto fn_proto =
        TFF_TRY(fn_val.GetUnplacedFunctionProto("federated_map fn"));

    std::vector<v0::Value> results(total_clients_);

    v0::Value map_intrinsic;
    map_intrinsic.mutable_computation()
        ->mutable_intrinsic()
        ->mutable_uri()
        ->assign(kFederatedMapAtClientsUri.data(),
                 kFederatedMapAtClientsUri.size());
    *map_intrinsic.mutable_computation()->mutable_type()->mutable_function() =
        type_pb;

    TFF_TRY(RunOnWorkers(
        total_clients_,
        [&](Executor* child, WorkQueue& queue,
            int32_t batch_size) -> absl::Status {
          auto intrinsic_or = child->CreateValue(map_intrinsic);
          if (!intrinsic_or.ok()) {
            LOG(WARNING) << "Worker setup failed: " << intrinsic_or.status();
            queue.RequeueAndMarkFailed({});
            return absl::OkStatus();
          }
          OwnedValueId intrinsic_id = std::move(intrinsic_or.value());
          auto fn_or = child->CreateValue(*fn_proto);
          if (!fn_or.ok()) {
            LOG(WARNING) << "Worker setup failed: " << fn_or.status();
            queue.RequeueAndMarkFailed({});
            return absl::OkStatus();
          }
          OwnedValueId fn_id = std::move(fn_or.value());

          while (true) {
            auto batch = queue.PopBatchOrWait(batch_size);
            if (batch.empty()) break;

            v0::Value fed_data =
                BuildFederatedClientsValue(*client_data, batch);
            auto data_or = child->CreateValue(fed_data);
            if (!data_or.ok()) {
              LOG(WARNING) << "Batch failed: " << data_or.status();
              queue.RequeueAndMarkFailed(batch);
              break;
            }
            OwnedValueId data_id = std::move(data_or.value());
            auto args_or = child->CreateStruct({fn_id, data_id});
            if (!args_or.ok()) {
              LOG(WARNING) << "Batch failed: " << args_or.status();
              queue.RequeueAndMarkFailed(batch);
              break;
            }
            OwnedValueId args_id = std::move(args_or.value());
            auto call_or = child->CreateCall(intrinsic_id, args_id);
            if (!call_or.ok()) {
              LOG(WARNING) << "Batch failed: " << call_or.status();
              queue.RequeueAndMarkFailed(batch);
              break;
            }
            OwnedValueId call_id = std::move(call_or.value());

            v0::Value result;
            auto mat_status = child->Materialize(call_id, &result);
            if (!mat_status.ok()) {
              LOG(WARNING) << "Batch failed: " << mat_status;
              queue.RequeueAndMarkFailed(batch);
              break;
            }

            if (!result.has_federated()) {
              return absl::InternalError(
                  "Child executor returned non-federated value for "
                  "federated_map");
            }

            // Extract per-client results from the federated value.
            // Thread-safety: each batch contains disjoint client indices.
            // A failing worker bails out before reaching this point, so a
            // retry worker will never race with the original on the same
            // indices.
            for (size_t j = 0; j < batch.size(); ++j) {
              if (static_cast<int>(j) < result.federated().value_size()) {
                results[batch[j]] = result.federated().value(j);
              } else {
                results[batch[j]] = result.federated().value(0);
              }
            }
          }
          return absl::OkStatus();
        },
        "federated_map"));

    auto out = std::make_shared<ClientsData>();
    out->all_equal = false;
    out->values = std::move(results);
    return ExecutorValue::CreateClientsPlaced(CentralClients(std::move(out)));
  }

  absl::StatusOr<ExecutorValue> CallIntrinsicAggregate(
      ExecutorValue&& arg,
      const federated_language::FunctionType& type_pb) override {
    TFF_TRY(arg.CheckLenForUseAsArgument("federated_aggregate", 5));
    const auto& data_val = arg.structure()->at(0);
    if (!data_val.is_clients_central()) {
      return absl::InternalError(
          "ElasticComposingExecutor: expected central clients value in "
          "federated_aggregate.");
    }
    const auto& client_data = data_val.clients_central();

    v0::Value zero_proto;
    {
      ParallelTasks zero_tasks(&thread_pool_);
      TFF_TRY(
          MaterializeValue(arg.structure()->at(1), &zero_proto, zero_tasks));
      TFF_TRY(zero_tasks.WaitAll());
    }
    auto accumulate_proto =
        TFF_TRY(arg.structure()->at(2).GetUnplacedFunctionProto("accumulate"));
    auto merge_proto =
        TFF_TRY(arg.structure()->at(3).GetUnplacedFunctionProto("merge"));
    auto report_proto =
        TFF_TRY(arg.structure()->at(4).GetUnplacedFunctionProto("report"));

    // Zero-clients edge case: no data to aggregate, just report(zero).
    if (total_clients_ == 0) {
      OwnedValueId report_fn = TFF_TRY(server_->CreateValue(*report_proto));
      OwnedValueId zero_val = TFF_TRY(server_->CreateValue(zero_proto));
      OwnedValueId result = TFF_TRY(server_->CreateCall(report_fn, zero_val));
      return ExecutorValue::CreateServerPlaced(ShareValueId(std::move(result)));
    }

    // Distribute aggregate across workers in batches.  Each worker produces
    // partial accumulators that are merged into a single accumulator on the
    // server under a mutex, so there is no post-loop merge step.
    absl::Mutex merge_mu;
    // Starts as nullopt; the first partial initializes it.
    std::optional<OwnedValueId> current;
    OwnedValueId merge_fn = TFF_TRY(server_->CreateValue(*merge_proto));

    v0::Value agg_intrinsic;
    agg_intrinsic.mutable_computation()->mutable_intrinsic()->set_uri(
        std::string(kFederatedAggregateUri));
    *agg_intrinsic.mutable_computation()->mutable_type()->mutable_function() =
        type_pb;

    // Identity report: each batch returns the raw accumulator, not the
    // user-supplied report. The real report is applied once after merging.
    v0::Value identity_report;
    *identity_report.mutable_computation() =
        tensorflow_federated::IdentityComp();

    TFF_TRY(RunOnWorkers(
        total_clients_,
        [&](Executor* child, WorkQueue& queue,
            int32_t batch_size) -> absl::Status {
          // Embed shared values on this worker.  If any fail, mark the
          // worker as failed and let others pick up its work.
          auto try_embed =
              [&](const v0::Value& val) -> absl::StatusOr<OwnedValueId> {
            auto or_val = child->CreateValue(val);
            if (!or_val.ok()) {
              LOG(WARNING) << "Worker setup failed: " << or_val.status();
              queue.RequeueAndMarkFailed({});
            }
            return or_val;
          };
          auto intrinsic_or = try_embed(agg_intrinsic);
          if (!intrinsic_or.ok()) return absl::OkStatus();
          auto zero_or = try_embed(zero_proto);
          if (!zero_or.ok()) return absl::OkStatus();
          auto acc_or = try_embed(*accumulate_proto);
          if (!acc_or.ok()) return absl::OkStatus();
          auto merge_or = try_embed(*merge_proto);
          if (!merge_or.ok()) return absl::OkStatus();
          auto report_or = try_embed(identity_report);
          if (!report_or.ok()) return absl::OkStatus();
          OwnedValueId intrinsic_id = std::move(*intrinsic_or);
          OwnedValueId zero_id = std::move(*zero_or);
          OwnedValueId acc_id = std::move(*acc_or);
          OwnedValueId merge_id = std::move(*merge_or);
          OwnedValueId report_id = std::move(*report_or);

          while (true) {
            auto batch = queue.PopBatchOrWait(batch_size);
            if (batch.empty()) break;

            v0::Value fed_data =
                BuildFederatedClientsValue(*client_data, batch);
            auto data_or = child->CreateValue(fed_data);
            if (!data_or.ok()) {
              LOG(WARNING) << "Batch failed: " << data_or.status();
              queue.RequeueAndMarkFailed(batch);
              break;
            }
            OwnedValueId data_id = std::move(data_or.value());
            auto args_or = child->CreateStruct(
                {data_id, zero_id, acc_id, merge_id, report_id});
            if (!args_or.ok()) {
              LOG(WARNING) << "Batch failed: " << args_or.status();
              queue.RequeueAndMarkFailed(batch);
              break;
            }
            OwnedValueId args_id = std::move(args_or.value());
            auto call_or = child->CreateCall(intrinsic_id, args_id);
            if (!call_or.ok()) {
              LOG(WARNING) << "Batch failed: " << call_or.status();
              queue.RequeueAndMarkFailed(batch);
              break;
            }
            OwnedValueId call_id = std::move(call_or.value());

            v0::Value result;
            auto mat_status = child->Materialize(call_id, &result);
            if (!mat_status.ok()) {
              LOG(WARNING) << "Batch failed: " << mat_status;
              queue.RequeueAndMarkFailed(batch);
              break;
            }

            if (!result.has_federated() ||
                result.federated().value_size() == 0) {
              return absl::InternalError(
                  "Child executor returned non-federated or empty value for "
                  "federated_aggregate");
            }

            // Extract the partial accumulator and merge it into the shared
            // accumulator on the server under a mutex.
            v0::Value partial = result.federated().value(0);
            absl::MutexLock lock(merge_mu);
            OwnedValueId partial_id = TFF_TRY(server_->CreateValue(partial));
            if (!current.has_value()) {
              current = std::move(partial_id);
            } else {
              OwnedValueId pair =
                  TFF_TRY(server_->CreateStruct({*current, partial_id}));
              current = TFF_TRY(server_->CreateCall(merge_fn, pair));
            }
          }
          return absl::OkStatus();
        },
        "federated_aggregate"));

    if (!current.has_value()) {
      return absl::InternalError("No partial aggregates were produced.");
    }

    // Apply report to the fully-merged accumulator.
    OwnedValueId report_fn = TFF_TRY(server_->CreateValue(*report_proto));
    OwnedValueId result = TFF_TRY(server_->CreateCall(report_fn, *current));
    return ExecutorValue::CreateServerPlaced(ShareValueId(std::move(result)));
  }

  absl::StatusOr<ExecutorValue> CallIntrinsicEvalAtClients(
      ExecutorValue&& arg,
      const federated_language::FunctionType& type_pb) override {
    auto fn_proto =
        TFF_TRY(arg.GetUnplacedFunctionProto("federated_eval_at_clients_fn"));

    std::vector<v0::Value> results(total_clients_);
    TFF_TRY(RunOnWorkers(
        total_clients_,
        [&](Executor* child, WorkQueue& queue,
            int32_t batch_size) -> absl::Status {
          auto fn_or = child->CreateValue(*fn_proto);
          if (!fn_or.ok()) {
            LOG(WARNING) << "Worker setup failed: " << fn_or.status();
            queue.RequeueAndMarkFailed({});
            return absl::OkStatus();
          }
          OwnedValueId fn_id = std::move(fn_or.value());

          while (true) {
            auto batch = queue.PopBatchOrWait(batch_size);
            if (batch.empty()) break;

            for (size_t i = 0; i < batch.size(); ++i) {
              int32_t item = batch[i];
              auto res_or = child->CreateCall(fn_id, std::nullopt);
              if (!res_or.ok()) {
                queue.RequeueAndMarkFailed(
                    {batch.begin() + static_cast<int>(i), batch.end()});
                return absl::OkStatus();
              }
              OwnedValueId res_id = std::move(res_or.value());
              auto mat = child->Materialize(res_id, &results[item]);
              if (!mat.ok()) {
                queue.RequeueAndMarkFailed(
                    {batch.begin() + static_cast<int>(i), batch.end()});
                return absl::OkStatus();
              }
            }
          }
          return absl::OkStatus();
        },
        "federated_eval_at_clients"));

    auto out = std::make_shared<ClientsData>();
    out->all_equal = false;
    out->values = std::move(results);
    return ExecutorValue::CreateClientsPlaced(CentralClients(std::move(out)));
  }
  int32_t avg_batches_per_worker_;
};

}  // namespace

std::shared_ptr<tensorflow_federated::Executor> CreateElasticComposingExecutor(
    std::shared_ptr<tensorflow_federated::Executor> server,
    std::vector<tensorflow_federated::ComposingChild> children,
    int32_t total_clients, int32_t avg_batches_per_worker) {
  int32_t num_children = children.size();
  return std::make_shared<ElasticComposingExecutor>(
      std::move(server), std::move(children), total_clients, num_children,
      avg_batches_per_worker);
}

}  // namespace confidential_federated_compute::program_executor_tee
