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
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/executors/composing_executor.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/mock_executor.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace confidential_federated_compute::program_executor_tee {
namespace {

using ::tensorflow_federated::ComposingChild;
using ::tensorflow_federated::Executor;
using ::tensorflow_federated::MockExecutor;
using ::tensorflow_federated::OwnedValueId;
using ::tensorflow_federated::ValueId;
using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;
namespace v0 = ::tensorflow_federated::v0;

// ---------------------------------------------------------------------------
// Proto helpers
// ---------------------------------------------------------------------------

std::atomic<int32_t> g_next_id{1};

v0::Value MakeInt32(int32_t val) {
  v0::Value v;
  v.mutable_array()->set_dtype(federated_language::DataType::DT_INT32);
  v.mutable_array()->mutable_int32_list()->add_value(val);
  return v;
}

int32_t GetInt32(const v0::Value& v) { return v.array().int32_list().value(0); }

v0::Value MakeClientsFederated(const std::vector<int32_t>& values,
                               bool all_equal = false) {
  v0::Value v;
  auto* fed = v.mutable_federated();
  fed->mutable_type()->set_all_equal(all_equal);
  fed->mutable_type()->mutable_placement()->mutable_value()->set_uri("clients");
  for (int32_t val : values) {
    *fed->add_value() = MakeInt32(val);
  }
  return v;
}

v0::Value MakeEightClientsFederated() {
  return MakeClientsFederated({1, 2, 3, 4, 5, 6, 7, 8});
}

v0::Value MakeServerFederated(int32_t val) {
  v0::Value v;
  auto* fed = v.mutable_federated();
  fed->mutable_type()->set_all_equal(true);
  fed->mutable_type()->mutable_placement()->mutable_value()->set_uri("server");
  *fed->add_value() = MakeInt32(val);
  return v;
}

v0::Value MakeIntrinsic(const std::string& uri) {
  v0::Value v;
  v.mutable_computation()->mutable_intrinsic()->set_uri(uri);
  v.mutable_computation()
      ->mutable_type()
      ->mutable_function()
      ->mutable_parameter();
  v.mutable_computation()->mutable_type()->mutable_function()->mutable_result();
  return v;
}

v0::Value MakeLambda() {
  v0::Value v;
  v.mutable_computation()->mutable_lambda()->set_parameter_name("x");
  return v;
}

// ---------------------------------------------------------------------------
// Mock setup
// ---------------------------------------------------------------------------

// Sets up a mock to accept all calls with default behavior.
void SetupMock(std::shared_ptr<NiceMock<MockExecutor>>& mock) {
  std::weak_ptr<Executor> weak = mock;
  ON_CALL(*mock, CreateValue(_))
      .WillByDefault([weak](const v0::Value&) -> absl::StatusOr<OwnedValueId> {
        return OwnedValueId(weak.lock(), g_next_id++);
      });
  ON_CALL(*mock, CreateCall(_, _))
      .WillByDefault(
          [weak](ValueId,
                 std::optional<const ValueId>) -> absl::StatusOr<OwnedValueId> {
            return OwnedValueId(weak.lock(), g_next_id++);
          });
  ON_CALL(*mock, CreateStruct(_))
      .WillByDefault(
          [weak](absl::Span<const ValueId>) -> absl::StatusOr<OwnedValueId> {
            return OwnedValueId(weak.lock(), g_next_id++);
          });
  ON_CALL(*mock, CreateSelection(_, _))
      .WillByDefault([weak](ValueId, uint32_t) -> absl::StatusOr<OwnedValueId> {
        return OwnedValueId(weak.lock(), g_next_id++);
      });
  ON_CALL(*mock, Materialize(_, _))
      .WillByDefault([](ValueId, v0::Value* out) -> absl::Status {
        *out = MakeClientsFederated({0});
        return absl::OkStatus();
      });
  ON_CALL(*mock, Dispose(_)).WillByDefault(Return(absl::OkStatus()));
}

// Sets up a mock where all operations return UNAVAILABLE (failing worker).
void SetupFailingMock(std::shared_ptr<NiceMock<MockExecutor>>& mock) {
  ON_CALL(*mock, CreateValue(_))
      .WillByDefault([](const v0::Value&) -> absl::StatusOr<OwnedValueId> {
        return absl::UnavailableError("worker down");
      });
  ON_CALL(*mock, CreateCall(_, _))
      .WillByDefault(
          [](ValueId,
             std::optional<const ValueId>) -> absl::StatusOr<OwnedValueId> {
            return absl::UnavailableError("worker down");
          });
  ON_CALL(*mock, CreateStruct(_))
      .WillByDefault(
          [](absl::Span<const ValueId>) -> absl::StatusOr<OwnedValueId> {
            return absl::UnavailableError("worker down");
          });
  ON_CALL(*mock, Materialize(_, _))
      .WillByDefault([](ValueId, v0::Value*) -> absl::Status {
        return absl::UnavailableError("worker down");
      });
  ON_CALL(*mock, Dispose(_)).WillByDefault(Return(absl::OkStatus()));
}

// ---------------------------------------------------------------------------
// Executor creation helpers
// ---------------------------------------------------------------------------

struct TestExecutors {
  std::shared_ptr<NiceMock<MockExecutor>> server;
  std::vector<std::shared_ptr<NiceMock<MockExecutor>>> children;
  std::shared_ptr<Executor> executor;
  std::vector<std::shared_ptr<std::atomic<int32_t>>> create_call_counts;

  int32_t create_call_count(size_t worker_index) const {
    return create_call_counts[worker_index]->load();
  }
};

TestExecutors MakeExecutor(int num_workers = 2, int total_clients = 8,
                           int avg_batches_per_worker = 2) {
  TestExecutors t;
  t.server = std::make_shared<NiceMock<MockExecutor>>();
  SetupMock(t.server);

  std::vector<ComposingChild> composing_children;
  for (int i = 0; i < num_workers; i++) {
    auto child = std::make_shared<NiceMock<MockExecutor>>();
    SetupMock(child);
    auto call_count = std::make_shared<std::atomic<int32_t>>(0);
    t.create_call_counts.push_back(call_count);
    std::weak_ptr<Executor> weak = child;
    ON_CALL(*child, CreateCall(_, _))
        .WillByDefault([weak, call_count](ValueId, std::optional<const ValueId>)
                           -> absl::StatusOr<OwnedValueId> {
          (*call_count)++;
          // Sleep briefly to give other worker threads a chance to pick up
          // batches, preventing one worker from draining the entire work queue.
          absl::SleepFor(absl::Milliseconds(10));
          return OwnedValueId(weak.lock(), g_next_id++);
        });
    composing_children.push_back(
        ComposingChild::Make(child, {{"clients", 0}}).value());
    t.children.push_back(std::move(child));
  }
  t.executor =
      CreateElasticComposingExecutor(t.server, std::move(composing_children),
                                     total_clients, avg_batches_per_worker);
  return t;
}

// Helper: CreateValue an intrinsic, struct the args, CreateCall.
absl::StatusOr<OwnedValueId> CallIntrinsic(
    Executor& exec, const std::string& uri,
    const std::vector<OwnedValueId>& arg_ids) {
  auto intrinsic_id = exec.CreateValue(MakeIntrinsic(uri));
  if (!intrinsic_id.ok()) return intrinsic_id.status();

  std::vector<ValueId> refs;
  refs.reserve(arg_ids.size());
  for (const auto& id : arg_ids) {
    refs.push_back(id.ref());
  }
  auto struct_id = exec.CreateStruct(refs);
  if (!struct_id.ok()) return struct_id.status();

  return exec.CreateCall(*intrinsic_id, *struct_id);
}

// ===========================================================================
// Tests: Value creation and materialization
// ===========================================================================

TEST(ElasticComposingExecutorTest, UnplacedRoundTrip) {
  auto t = MakeExecutor();
  v0::Value input = MakeInt32(42);
  ON_CALL(*t.server, Materialize(_, _))
      .WillByDefault([&input](ValueId, v0::Value* out) -> absl::Status {
        *out = input;
        return absl::OkStatus();
      });

  auto id = t.executor->CreateValue(input);
  ASSERT_TRUE(id.ok()) << id.status();

  v0::Value output;
  ASSERT_TRUE(t.executor->Materialize(*id, &output).ok());
  EXPECT_EQ(GetInt32(output), 42);
}

TEST(ElasticComposingExecutorTest, ClientsFederatedRoundTrip) {
  auto t = MakeExecutor();

  v0::Value fed = MakeClientsFederated({10, 20, 30, 40, 50, 60, 70, 80});
  auto id = t.executor->CreateValue(fed);
  ASSERT_TRUE(id.ok()) << id.status();

  v0::Value output;
  ASSERT_TRUE(t.executor->Materialize(*id, &output).ok());

  ASSERT_TRUE(output.has_federated());
  EXPECT_FALSE(output.federated().type().all_equal());
  EXPECT_EQ(output.federated().type().placement().value().uri(), "clients");
  ASSERT_EQ(output.federated().value_size(), 8);
  EXPECT_EQ(GetInt32(output.federated().value(0)), 10);
  EXPECT_EQ(GetInt32(output.federated().value(1)), 20);
  EXPECT_EQ(GetInt32(output.federated().value(2)), 30);
}

TEST(ElasticComposingExecutorTest, ClientsAllEqualRoundTrip) {
  auto t = MakeExecutor();

  v0::Value fed = MakeClientsFederated({42}, /*all_equal=*/true);
  auto id = t.executor->CreateValue(fed);
  ASSERT_TRUE(id.ok()) << id.status();

  v0::Value output;
  ASSERT_TRUE(t.executor->Materialize(*id, &output).ok());

  ASSERT_TRUE(output.has_federated());
  EXPECT_TRUE(output.federated().type().all_equal());
  ASSERT_EQ(output.federated().value_size(), 1);
  EXPECT_EQ(GetInt32(output.federated().value(0)), 42);
}

TEST(ElasticComposingExecutorTest, FederatedBroadcast) {
  auto t = MakeExecutor();

  v0::Value server_val = MakeServerFederated(99);
  ON_CALL(*t.server, Materialize(_, _))
      .WillByDefault([](ValueId, v0::Value* out) -> absl::Status {
        *out = MakeInt32(99);
        return absl::OkStatus();
      });

  auto server_id = t.executor->CreateValue(server_val);
  ASSERT_TRUE(server_id.ok()) << server_id.status();

  auto intrinsic_id =
      t.executor->CreateValue(MakeIntrinsic("federated_broadcast"));
  ASSERT_TRUE(intrinsic_id.ok()) << intrinsic_id.status();

  auto broadcast_id = t.executor->CreateCall(*intrinsic_id, server_id->ref());
  ASSERT_TRUE(broadcast_id.ok()) << broadcast_id.status();

  v0::Value output;
  auto mat_status = t.executor->Materialize(*broadcast_id, &output);
  ASSERT_TRUE(mat_status.ok()) << mat_status;
  ASSERT_TRUE(output.has_federated());
  EXPECT_TRUE(output.federated().type().all_equal());
  EXPECT_EQ(output.federated().type().placement().value().uri(), "clients");
  ASSERT_EQ(output.federated().value_size(), 1);
  EXPECT_EQ(GetInt32(output.federated().value(0)), 99);
}

TEST(ElasticComposingExecutorTest, FederatedValueAtClients) {
  auto t = MakeExecutor();

  v0::Value unplaced = MakeInt32(123);
  auto unplaced_id = t.executor->CreateValue(unplaced);
  ASSERT_TRUE(unplaced_id.ok()) << unplaced_id.status();

  auto intrinsic_id =
      t.executor->CreateValue(MakeIntrinsic("federated_value_at_clients"));
  ASSERT_TRUE(intrinsic_id.ok()) << intrinsic_id.status();

  auto at_clients_id =
      t.executor->CreateCall(*intrinsic_id, unplaced_id->ref());
  ASSERT_TRUE(at_clients_id.ok()) << at_clients_id.status();

  v0::Value output;
  auto mat_status = t.executor->Materialize(*at_clients_id, &output);
  ASSERT_TRUE(mat_status.ok()) << mat_status;
  ASSERT_TRUE(output.has_federated());
  EXPECT_TRUE(output.federated().type().all_equal());
  EXPECT_EQ(output.federated().type().placement().value().uri(), "clients");
  ASSERT_EQ(output.federated().value_size(), 1);
  EXPECT_EQ(GetInt32(output.federated().value(0)), 123);
}

// ===========================================================================
// Tests: federated_map
// ===========================================================================

TEST(ElasticComposingExecutorTest, FederatedMapReturnsCorrectClientCount) {
  auto t = MakeExecutor();

  auto data_id = t.executor->CreateValue(MakeEightClientsFederated());
  ASSERT_TRUE(data_id.ok()) << data_id.status();

  auto fn_id = t.executor->CreateValue(MakeLambda());
  ASSERT_TRUE(fn_id.ok()) << fn_id.status();

  std::vector<OwnedValueId> args;
  args.push_back(std::move(*fn_id));
  args.push_back(std::move(*data_id));
  auto result_id = CallIntrinsic(*t.executor, "federated_map", args);
  ASSERT_TRUE(result_id.ok()) << result_id.status();

  v0::Value output;
  ASSERT_TRUE(t.executor->Materialize(*result_id, &output).ok());
  ASSERT_TRUE(output.has_federated());
  EXPECT_EQ(output.federated().type().placement().value().uri(), "clients");
  EXPECT_EQ(output.federated().value_size(), 8);

  // 4 batches total distributed dynamically across 2 workers.
  EXPECT_EQ(t.create_call_count(0) + t.create_call_count(1), 4);
  EXPECT_GT(t.create_call_count(0), 0);
  EXPECT_GT(t.create_call_count(1), 0);
}

TEST(ElasticComposingExecutorTest, FederatedMapWithAllEqualClientData) {
  auto t = MakeExecutor();

  // Create an all_equal client value (e.g. from broadcast or
  // federated_value_at_clients).
  v0::Value fed = MakeClientsFederated({42}, /*all_equal=*/true);
  auto data_id = t.executor->CreateValue(fed);
  ASSERT_TRUE(data_id.ok()) << data_id.status();

  auto fn_id = t.executor->CreateValue(MakeLambda());
  ASSERT_TRUE(fn_id.ok()) << fn_id.status();

  std::vector<OwnedValueId> args;
  args.push_back(std::move(*fn_id));
  args.push_back(std::move(*data_id));
  auto result_id = CallIntrinsic(*t.executor, "federated_map", args);
  ASSERT_TRUE(result_id.ok()) << result_id.status();

  v0::Value output;
  ASSERT_TRUE(t.executor->Materialize(*result_id, &output).ok());
  ASSERT_TRUE(output.has_federated());
  EXPECT_EQ(output.federated().type().placement().value().uri(), "clients");
  // Total clients is 8; all 8 should receive mapped results even when input was
  // all_equal.
  EXPECT_EQ(output.federated().value_size(), 8);

  EXPECT_EQ(t.create_call_count(0) + t.create_call_count(1), 4);
  EXPECT_GT(t.create_call_count(0), 0);
  EXPECT_GT(t.create_call_count(1), 0);
}

TEST(ElasticComposingExecutorTest, FederatedMapWorkerFailure) {
  auto t = MakeExecutor();

  // Kill worker 1.
  SetupFailingMock(t.children[1]);

  auto data_id = t.executor->CreateValue(MakeEightClientsFederated());
  ASSERT_TRUE(data_id.ok());
  auto fn_id = t.executor->CreateValue(MakeLambda());
  ASSERT_TRUE(fn_id.ok());

  std::vector<OwnedValueId> args;
  args.push_back(std::move(*fn_id));
  args.push_back(std::move(*data_id));
  auto result_id = CallIntrinsic(*t.executor, "federated_map", args);
  ASSERT_TRUE(result_id.ok()) << result_id.status();

  v0::Value output;
  auto status = t.executor->Materialize(*result_id, &output);
  ASSERT_TRUE(status.ok()) << "Surviving worker should handle all chunks: "
                           << status;
  ASSERT_TRUE(output.has_federated());
  EXPECT_EQ(output.federated().value_size(), 8);

  // Surviving worker 0 should handle all 4 batches; worker 1 failed.
  EXPECT_EQ(t.create_call_count(0), 4);
  EXPECT_EQ(t.create_call_count(1), 0);
}

TEST(ElasticComposingExecutorTest, AllWorkersFailMapReturnsError) {
  auto t = MakeExecutor();

  SetupFailingMock(t.children[0]);
  SetupFailingMock(t.children[1]);

  auto data_id = t.executor->CreateValue(MakeEightClientsFederated());
  ASSERT_TRUE(data_id.ok());
  auto fn_id = t.executor->CreateValue(MakeLambda());
  ASSERT_TRUE(fn_id.ok());

  std::vector<OwnedValueId> args;
  args.push_back(std::move(*fn_id));
  args.push_back(std::move(*data_id));
  auto result_id = CallIntrinsic(*t.executor, "federated_map", args);
  ASSERT_TRUE(result_id.ok());

  v0::Value output;
  EXPECT_FALSE(t.executor->Materialize(*result_id, &output).ok());
}

TEST(ElasticComposingExecutorTest, RetryExhaustionDuringMapReturnsError) {
  // 6 workers and 1 total client ensure chunk 0 is retried 5 times
  // across workers, exceeding kMaxRetries = 4.
  auto t = MakeExecutor(/*num_workers=*/6, /*total_clients=*/1,
                        /*avg_batches_per_worker=*/1);

  // All workers: CreateCall fails so chunks are requeued until retries exhaust.
  for (int i = 0; i < 6; i++) {
    ON_CALL(*t.children[i], CreateCall(_, _))
        .WillByDefault(
            [](ValueId,
               std::optional<const ValueId>) -> absl::StatusOr<OwnedValueId> {
              return absl::UnavailableError("poisoned chunk");
            });
  }

  auto data_id = t.executor->CreateValue(MakeClientsFederated({1}));
  ASSERT_TRUE(data_id.ok());

  auto fn_id = t.executor->CreateValue(MakeLambda());
  ASSERT_TRUE(fn_id.ok());

  std::vector<OwnedValueId> args;
  args.push_back(std::move(*fn_id));
  args.push_back(std::move(*data_id));
  auto result_id = CallIntrinsic(*t.executor, "federated_map", args);
  ASSERT_TRUE(result_id.ok());

  v0::Value output;
  auto status = t.executor->Materialize(*result_id, &output);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              testing::HasSubstr("Items exceeded maximum retry count"));
}

// ===========================================================================
// Tests: federated_aggregate
// ===========================================================================

TEST(ElasticComposingExecutorTest, FederatedAggregateProducesServerResult) {
  auto t = MakeExecutor();

  auto data_id = t.executor->CreateValue(MakeEightClientsFederated());
  ASSERT_TRUE(data_id.ok());
  auto zero_id = t.executor->CreateValue(MakeInt32(0));
  ASSERT_TRUE(zero_id.ok());
  auto accum_id = t.executor->CreateValue(MakeLambda());
  auto merge_id = t.executor->CreateValue(MakeLambda());
  auto report_id = t.executor->CreateValue(MakeLambda());
  ASSERT_TRUE(accum_id.ok());
  ASSERT_TRUE(merge_id.ok());
  ASSERT_TRUE(report_id.ok());

  // Server materialize returns the final aggregate result.
  ON_CALL(*t.server, Materialize(_, _))
      .WillByDefault([](ValueId, v0::Value* out) -> absl::Status {
        *out = MakeInt32(10);
        return absl::OkStatus();
      });

  std::vector<OwnedValueId> args;
  args.push_back(std::move(*data_id));
  args.push_back(std::move(*zero_id));
  args.push_back(std::move(*accum_id));
  args.push_back(std::move(*merge_id));
  args.push_back(std::move(*report_id));
  auto result_id = CallIntrinsic(*t.executor, "federated_aggregate", args);
  ASSERT_TRUE(result_id.ok()) << result_id.status();

  v0::Value output;
  ASSERT_TRUE(t.executor->Materialize(*result_id, &output).ok());
  // Aggregate returns a server-placed federated value.
  ASSERT_TRUE(output.has_federated());
  EXPECT_EQ(output.federated().type().placement().value().uri(), "server");

  // 4 batches total distributed dynamically across 2 workers.
  EXPECT_EQ(t.create_call_count(0) + t.create_call_count(1), 4);
  EXPECT_GT(t.create_call_count(0), 0);
  EXPECT_GT(t.create_call_count(1), 0);
}

TEST(ElasticComposingExecutorTest, FederatedAggregateWithAllEqualClientData) {
  auto t = MakeExecutor();

  // Aggregate an all_equal client input.
  auto data_id =
      t.executor->CreateValue(MakeClientsFederated({42}, /*all_equal=*/true));
  ASSERT_TRUE(data_id.ok());
  auto zero_id = t.executor->CreateValue(MakeInt32(0));
  ASSERT_TRUE(zero_id.ok());
  auto accum_id = t.executor->CreateValue(MakeLambda());
  auto merge_id = t.executor->CreateValue(MakeLambda());
  auto report_id = t.executor->CreateValue(MakeLambda());
  ASSERT_TRUE(accum_id.ok());
  ASSERT_TRUE(merge_id.ok());
  ASSERT_TRUE(report_id.ok());

  ON_CALL(*t.server, Materialize(_, _))
      .WillByDefault([](ValueId, v0::Value* out) -> absl::Status {
        *out = MakeInt32(10);
        return absl::OkStatus();
      });

  std::vector<OwnedValueId> args;
  args.push_back(std::move(*data_id));
  args.push_back(std::move(*zero_id));
  args.push_back(std::move(*accum_id));
  args.push_back(std::move(*merge_id));
  args.push_back(std::move(*report_id));
  auto result_id = CallIntrinsic(*t.executor, "federated_aggregate", args);
  ASSERT_TRUE(result_id.ok()) << result_id.status();

  v0::Value output;
  ASSERT_TRUE(t.executor->Materialize(*result_id, &output).ok());
  ASSERT_TRUE(output.has_federated());
  EXPECT_EQ(output.federated().type().placement().value().uri(), "server");

  EXPECT_EQ(t.create_call_count(0) + t.create_call_count(1), 4);
  EXPECT_GT(t.create_call_count(0), 0);
  EXPECT_GT(t.create_call_count(1), 0);
}

TEST(ElasticComposingExecutorTest, FederatedAggregateWorkerFailure) {
  auto t = MakeExecutor();

  // Kill worker 1.
  SetupFailingMock(t.children[1]);

  ON_CALL(*t.server, Materialize(_, _))
      .WillByDefault([](ValueId, v0::Value* out) -> absl::Status {
        *out = MakeInt32(99);
        return absl::OkStatus();
      });

  auto data_id = t.executor->CreateValue(MakeEightClientsFederated());
  ASSERT_TRUE(data_id.ok());
  auto zero_id = t.executor->CreateValue(MakeInt32(0));
  auto accum_id = t.executor->CreateValue(MakeLambda());
  auto merge_id = t.executor->CreateValue(MakeLambda());
  auto report_id = t.executor->CreateValue(MakeLambda());

  std::vector<OwnedValueId> args;
  args.push_back(std::move(*data_id));
  args.push_back(std::move(*zero_id));
  args.push_back(std::move(*accum_id));
  args.push_back(std::move(*merge_id));
  args.push_back(std::move(*report_id));
  auto result_id = CallIntrinsic(*t.executor, "federated_aggregate", args);
  ASSERT_TRUE(result_id.ok()) << result_id.status();

  v0::Value output;
  auto status = t.executor->Materialize(*result_id, &output);
  ASSERT_TRUE(status.ok()) << "Surviving worker should handle aggregate: "
                           << status;
  ASSERT_TRUE(output.has_federated());
  EXPECT_EQ(output.federated().type().placement().value().uri(), "server");

  // Surviving worker 0 should handle all 4 aggregate batches; worker 1 failed.
  EXPECT_EQ(t.create_call_count(0), 4);
  EXPECT_EQ(t.create_call_count(1), 0);
}

TEST(ElasticComposingExecutorTest, AllWorkersFailAggregateReturnsError) {
  auto t = MakeExecutor();

  SetupFailingMock(t.children[0]);
  SetupFailingMock(t.children[1]);

  auto data_id = t.executor->CreateValue(MakeEightClientsFederated());
  ASSERT_TRUE(data_id.ok());
  auto zero_id = t.executor->CreateValue(MakeInt32(0));
  auto accum_id = t.executor->CreateValue(MakeLambda());
  auto merge_id = t.executor->CreateValue(MakeLambda());
  auto report_id = t.executor->CreateValue(MakeLambda());

  std::vector<OwnedValueId> args;
  args.push_back(std::move(*data_id));
  args.push_back(std::move(*zero_id));
  args.push_back(std::move(*accum_id));
  args.push_back(std::move(*merge_id));
  args.push_back(std::move(*report_id));
  auto result_id = CallIntrinsic(*t.executor, "federated_aggregate", args);
  ASSERT_TRUE(result_id.ok());

  v0::Value output;
  EXPECT_FALSE(t.executor->Materialize(*result_id, &output).ok());
}

TEST(ElasticComposingExecutorTest, RetryExhaustionDuringAggregateReturnsError) {
  // 6 workers and 1 total client ensure chunk 0 is retried 5 times
  // across workers, exceeding kMaxRetries = 4.
  auto t = MakeExecutor(/*num_workers=*/6, /*total_clients=*/1,
                        /*avg_batches_per_worker=*/1);

  for (int i = 0; i < 6; i++) {
    ON_CALL(*t.children[i], CreateCall(_, _))
        .WillByDefault(
            [](ValueId,
               std::optional<const ValueId>) -> absl::StatusOr<OwnedValueId> {
              return absl::UnavailableError("poisoned chunk");
            });
  }

  auto data_id = t.executor->CreateValue(MakeClientsFederated({1}));
  ASSERT_TRUE(data_id.ok());
  auto zero_id = t.executor->CreateValue(MakeInt32(0));
  auto accum_id = t.executor->CreateValue(MakeLambda());
  auto merge_id = t.executor->CreateValue(MakeLambda());
  auto report_id = t.executor->CreateValue(MakeLambda());

  std::vector<OwnedValueId> args;
  args.push_back(std::move(*data_id));
  args.push_back(std::move(*zero_id));
  args.push_back(std::move(*accum_id));
  args.push_back(std::move(*merge_id));
  args.push_back(std::move(*report_id));
  auto result_id = CallIntrinsic(*t.executor, "federated_aggregate", args);
  ASSERT_TRUE(result_id.ok());

  v0::Value output;
  auto status = t.executor->Materialize(*result_id, &output);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              testing::HasSubstr("Items exceeded maximum retry count"));
}

TEST(ElasticComposingExecutorTest, FederatedAggregateZeroClients) {
  auto t = MakeExecutor(/*num_workers=*/2, /*total_clients=*/0);

  // With 0 clients, aggregate should just apply report(zero) on the server.
  ON_CALL(*t.server, Materialize(_, _))
      .WillByDefault([](ValueId, v0::Value* out) -> absl::Status {
        *out = MakeInt32(0);
        return absl::OkStatus();
      });

  auto data_id = t.executor->CreateValue(MakeClientsFederated({}));
  ASSERT_TRUE(data_id.ok());
  auto zero_id = t.executor->CreateValue(MakeInt32(0));
  auto accum_id = t.executor->CreateValue(MakeLambda());
  auto merge_id = t.executor->CreateValue(MakeLambda());
  auto report_id = t.executor->CreateValue(MakeLambda());

  std::vector<OwnedValueId> args;
  args.push_back(std::move(*data_id));
  args.push_back(std::move(*zero_id));
  args.push_back(std::move(*accum_id));
  args.push_back(std::move(*merge_id));
  args.push_back(std::move(*report_id));
  auto result_id = CallIntrinsic(*t.executor, "federated_aggregate", args);
  ASSERT_TRUE(result_id.ok()) << result_id.status();

  v0::Value output;
  auto status = t.executor->Materialize(*result_id, &output);
  ASSERT_TRUE(status.ok()) << status;
  // Result should be server-placed.
  ASSERT_TRUE(output.has_federated());
  EXPECT_EQ(output.federated().type().placement().value().uri(), "server");
}

// ===========================================================================
// Tests: federated_eval_at_clients
// ===========================================================================

TEST(ElasticComposingExecutorTest,
     FederatedEvalAtClientsReturnsCorrectClientCount) {
  auto t = MakeExecutor();

  auto fn_id = t.executor->CreateValue(MakeLambda());
  ASSERT_TRUE(fn_id.ok());

  auto intrinsic_id =
      t.executor->CreateValue(MakeIntrinsic("federated_eval_at_clients"));
  ASSERT_TRUE(intrinsic_id.ok());

  auto result_id = t.executor->CreateCall(*intrinsic_id, *fn_id);
  ASSERT_TRUE(result_id.ok()) << result_id.status();

  v0::Value output;
  auto status = t.executor->Materialize(*result_id, &output);
  ASSERT_TRUE(status.ok()) << status;
  ASSERT_TRUE(output.has_federated());
  EXPECT_EQ(output.federated().value_size(), 8);

  // 8 CreateCall calls total (1 per client) distributed dynamically across 2
  // workers.
  EXPECT_EQ(t.create_call_count(0) + t.create_call_count(1), 8);
  EXPECT_GT(t.create_call_count(0), 0);
  EXPECT_GT(t.create_call_count(1), 0);
}

TEST(ElasticComposingExecutorTest, FederatedEvalAtClientsWorkerFailure) {
  auto t = MakeExecutor();

  // Kill worker 1.
  SetupFailingMock(t.children[1]);

  auto fn_id = t.executor->CreateValue(MakeLambda());
  ASSERT_TRUE(fn_id.ok());

  auto intrinsic_id =
      t.executor->CreateValue(MakeIntrinsic("federated_eval_at_clients"));
  ASSERT_TRUE(intrinsic_id.ok());

  auto result_id = t.executor->CreateCall(*intrinsic_id, *fn_id);
  ASSERT_TRUE(result_id.ok()) << result_id.status();

  v0::Value output;
  auto status = t.executor->Materialize(*result_id, &output);
  ASSERT_TRUE(status.ok()) << "Surviving worker 0 should process all batches: "
                           << status;
  ASSERT_TRUE(output.has_federated());
  EXPECT_EQ(output.federated().value_size(), 8);

  // Worker 0 should handle all 8 clients.
  EXPECT_EQ(t.create_call_count(0), 8);
  EXPECT_EQ(t.create_call_count(1), 0);
}

}  // namespace
}  // namespace confidential_federated_compute::program_executor_tee
