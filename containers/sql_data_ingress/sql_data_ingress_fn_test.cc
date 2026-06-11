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

#include "containers/sql_data_ingress/sql_data_ingress_fn.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "containers/fns/fn.h"
#include "containers/fns/fn_factory.h"
#include "containers/session.h"
#include "containers/testing/mocks.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/sql_data_ingress_config.pb.h"
#include "fcp/protos/confidentialcompute/sql_query.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/any.pb.h"
#include "google/protobuf/descriptor.pb.h"
#include "gtest/gtest.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_parser.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute::sql_data_ingress {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::confidential_federated_compute::MockContext;
using ::confidential_federated_compute::fns::Fn;
using ::confidential_federated_compute::fns::FnFactory;
using ::fcp::confidentialcompute::
    SqlDataIngressContainerInitializeConfiguration;
using ::fcp::confidentialcompute::SqlQuery;
using ::fcp::confidentialcompute::WriteRequest;
using ::google::protobuf::Any;
using ::tensorflow_federated::aggregation::CreateTestData;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointParserFactory;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;
using ::testing::_;
using ::testing::Eq;
using ::testing::Return;
using ::testing::StrictMock;

Any CreateValidSqlConfig() {
  SqlQuery sql_query = PARSE_TEXT_PROTO(R"pb(
    raw_sql: "SELECT text AS result FROM input"
    database_schema {
      table {
        name: "input"
        column { name: "text" type: STRING }
        create_table_sql: "CREATE TABLE input (text TEXT)"
      }
    }
    output_columns { name: "result" type: STRING }
  )pb");
  SqlDataIngressContainerInitializeConfiguration init_config;
  *init_config.mutable_sql_query() = sql_query;
  Any config;
  config.PackFrom(init_config);
  return config;
}

absl::StatusOr<std::string> CreateInputCheckpoint(
    std::initializer_list<absl::string_view> text_values) {
  auto text_tensor =
      Tensor::Create(DataType::DT_STRING,
                     TensorShape({static_cast<int64_t>(text_values.size())}),
                     CreateTestData<absl::string_view>(text_values), "text");
  if (!text_tensor.ok()) return text_tensor.status();

  FederatedComputeCheckpointBuilderFactory factory;
  auto builder = factory.Create();
  auto status = builder->Add(text_tensor->name(), *text_tensor);
  if (!status.ok()) return status;
  auto ckpt = builder->Build();
  if (!ckpt.ok()) return ckpt.status();
  return std::string(ckpt->Flatten());
}

class SqlDataIngressFnTest : public testing::Test {
 protected:
  void SetUp() override {
    auto factory =
        ProvideSqlDataIngressFnFactory(CreateValidSqlConfig(), Any(), {});
    ASSERT_THAT(factory, IsOk());
    auto fn = (*factory)->CreateFn();
    ASSERT_THAT(fn, IsOk());
    fn_ = std::move(*fn);

    // Initialize with valid SQL config.
    Any config = CreateValidSqlConfig();
    ASSERT_THAT(fn_->Configure(
                    [&config]() {
                      fcp::confidentialcompute::ConfigureRequest req;
                      req.set_chunk_size(1000);
                      *req.mutable_configuration() = config;
                      return req;
                    }(),
                    context_),
                IsOk());
  }

  std::unique_ptr<Fn> fn_;
  StrictMock<MockContext> context_;
};

TEST(SqlDataIngressFnFactoryTest, CreateFnSucceeds) {
  auto factory =
      ProvideSqlDataIngressFnFactory(CreateValidSqlConfig(), Any(), {});
  ASSERT_THAT(factory, IsOk());
  EXPECT_THAT((*factory)->CreateFn(), IsOk());
}

TEST(SqlDataIngressFnConfigureTest, InvalidConfigFails) {
  auto factory = ProvideSqlDataIngressFnFactory(Any(), Any(), {});
  ASSERT_THAT(factory, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SqlDataIngressFnConfigureTest, NoTablesFails) {
  SqlQuery sql_query = PARSE_TEXT_PROTO(R"pb(
    raw_sql: "SELECT 1"
    database_schema {}
    output_columns { name: "result" type: INT64 }
  )pb");
  SqlDataIngressContainerInitializeConfiguration init_config;
  *init_config.mutable_sql_query() = sql_query;
  Any config;
  config.PackFrom(init_config);

  auto factory = ProvideSqlDataIngressFnFactory(config, Any(), {});
  ASSERT_THAT(factory, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SqlDataIngressFnConfigureTest, NoColumnsFails) {
  SqlQuery sql_query = PARSE_TEXT_PROTO(R"pb(
    raw_sql: "SELECT 1"
    database_schema {
      table { name: "input" create_table_sql: "CREATE TABLE input (x INT)" }
    }
    output_columns { name: "result" type: INT64 }
  )pb");
  SqlDataIngressContainerInitializeConfiguration init_config;
  *init_config.mutable_sql_query() = sql_query;
  Any config;
  config.PackFrom(init_config);

  auto factory = ProvideSqlDataIngressFnFactory(config, Any(), {});
  ASSERT_THAT(factory, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SqlDataIngressFnConfigureTest, NoOutputColumnsFails) {
  SqlQuery sql_query = PARSE_TEXT_PROTO(R"pb(
    raw_sql: "SELECT 1"
    database_schema {
      table {
        name: "input"
        column { name: "text" type: STRING }
        create_table_sql: "CREATE TABLE input (text TEXT)"
      }
    }
  )pb");
  SqlDataIngressContainerInitializeConfiguration init_config;
  *init_config.mutable_sql_query() = sql_query;
  Any config;
  config.PackFrom(init_config);

  auto factory = ProvideSqlDataIngressFnFactory(config, Any(), {});
  ASSERT_THAT(factory, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SqlDataIngressFnCustomConfigTest,
     EmitEncryptedCheckpointInt64ColumnSucceeds) {
  SqlQuery sql_query = PARSE_TEXT_PROTO(R"pb(
    raw_sql: "SELECT 1 AS result FROM input"
    database_schema {
      table {
        name: "input"
        column { name: "text" type: STRING }
        create_table_sql: "CREATE TABLE input (text TEXT)"
      }
    }
    output_columns { name: "result" type: INT64 }
  )pb");
  SqlDataIngressContainerInitializeConfiguration init_config;
  *init_config.mutable_sql_query() = sql_query;
  Any config;
  config.PackFrom(init_config);

  auto factory = ProvideSqlDataIngressFnFactory(config, Any(), {});
  ASSERT_THAT(factory, IsOk());
  auto fn = (*factory)->CreateFn();
  ASSERT_THAT(fn, IsOk());

  auto checkpoint = CreateInputCheckpoint({"example"});
  ASSERT_THAT(checkpoint, IsOk());
  std::string checkpoint_str = std::move(checkpoint).value();

  WriteRequest request;
  request.mutable_first_request_metadata()
      ->mutable_hpke_plus_aead_data()
      ->set_blob_id("blob_id");

  StrictMock<MockContext> context;
  std::string emitted_data;
  EXPECT_CALL(context, EmitEncrypted(Eq(0), _))
      .WillOnce([&emitted_data](int, Session::KV kv) {
        emitted_data = std::move(kv.data);
        EXPECT_EQ(kv.blob_id, "blob_id");
        return true;
      });

  auto result = (*fn)->Write(request, checkpoint_str, context);
  ASSERT_THAT(result, IsOk());

  // Parse the emitted checkpoint and verify the SQL result.
  FederatedComputeCheckpointParserFactory parser_factory;
  auto parser = parser_factory.Create(absl::Cord(emitted_data));
  ASSERT_THAT(parser, IsOk());

  auto result_tensor = (*parser)->GetTensor("result");
  ASSERT_THAT(result_tensor, IsOk());
  EXPECT_EQ(result_tensor->dtype(), DataType::DT_INT64);
  EXPECT_EQ(result_tensor->num_elements(), 1);
  EXPECT_THAT(result_tensor->AsSpan<int64_t>(), testing::ElementsAre(1));
}

TEST(SqlDataIngressFnCustomConfigTest,
     EmitEncryptedCheckpointMultipleColumnsSucceeds) {
  SqlQuery sql_query = PARSE_TEXT_PROTO(R"pb(
    raw_sql: "SELECT text AS result1, length(text) AS result2 FROM input"
    database_schema {
      table {
        name: "input"
        column { name: "text" type: STRING }
        create_table_sql: "CREATE TABLE input (text TEXT)"
      }
    }
    output_columns { name: "result1" type: STRING }
    output_columns { name: "result2" type: INT64 }
  )pb");
  SqlDataIngressContainerInitializeConfiguration init_config;
  *init_config.mutable_sql_query() = sql_query;
  Any config;
  config.PackFrom(init_config);

  auto factory = ProvideSqlDataIngressFnFactory(config, Any(), {});
  ASSERT_THAT(factory, IsOk());
  auto fn = (*factory)->CreateFn();
  ASSERT_THAT(fn, IsOk());

  auto checkpoint = CreateInputCheckpoint({"abc", "de"});
  ASSERT_THAT(checkpoint, IsOk());
  std::string checkpoint_str = std::move(checkpoint).value();

  WriteRequest request;
  request.mutable_first_request_metadata()
      ->mutable_hpke_plus_aead_data()
      ->set_blob_id("blob_id");

  StrictMock<MockContext> context;
  std::string emitted_data;
  EXPECT_CALL(context, EmitEncrypted(Eq(0), _))
      .WillOnce([&emitted_data](int, Session::KV kv) {
        emitted_data = std::move(kv.data);
        EXPECT_EQ(kv.blob_id, "blob_id");
        return true;
      });

  auto result = (*fn)->Write(request, checkpoint_str, context);
  ASSERT_THAT(result, IsOk());

  // Parse the emitted checkpoint and verify the SQL result.
  FederatedComputeCheckpointParserFactory parser_factory;
  auto parser = parser_factory.Create(absl::Cord(emitted_data));
  ASSERT_THAT(parser, IsOk());

  auto result_tensor1 = (*parser)->GetTensor("result1");
  ASSERT_THAT(result_tensor1, IsOk());
  EXPECT_EQ(result_tensor1->dtype(), DataType::DT_STRING);
  EXPECT_EQ(result_tensor1->num_elements(), 2);
  EXPECT_THAT(result_tensor1->AsSpan<absl::string_view>(),
              testing::ElementsAre("abc", "de"));

  auto result_tensor2 = (*parser)->GetTensor("result2");
  ASSERT_THAT(result_tensor2, IsOk());
  EXPECT_EQ(result_tensor2->dtype(), DataType::DT_INT64);
  EXPECT_EQ(result_tensor2->num_elements(), 2);
  EXPECT_THAT(result_tensor2->AsSpan<int64_t>(), testing::ElementsAre(3, 2));
}

TEST_F(SqlDataIngressFnTest, InvalidCheckpointDataFails) {
  WriteRequest request;
  request.mutable_first_request_metadata()
      ->mutable_hpke_plus_aead_data()
      ->set_blob_id("blob_id");

  auto result = fn_->Write(request, "invalid_checkpoint_data", context_);
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(SqlDataIngressFnTest, InputMissingBlobIdFails) {
  auto checkpoint = CreateInputCheckpoint({"example1"});
  ASSERT_THAT(checkpoint, IsOk());
  std::string checkpoint_str = std::move(checkpoint).value();

  WriteRequest request;
  request.mutable_first_request_metadata()->mutable_hpke_plus_aead_data();

  auto result = fn_->Write(request, checkpoint_str, context_);
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(SqlDataIngressFnTest, EmitEncryptedFails) {
  auto text_tensor = Tensor::Create(
      DataType::DT_STRING, TensorShape({2}),
      CreateTestData<absl::string_view>({"hello", "world"}), "text");
  ASSERT_THAT(text_tensor, IsOk());

  FederatedComputeCheckpointBuilderFactory factory;
  auto builder = factory.Create();
  ASSERT_THAT(builder->Add(text_tensor->name(), *text_tensor), IsOk());
  auto ckpt = builder->Build();
  ASSERT_THAT(ckpt, IsOk());
  std::string checkpoint = std::string(ckpt->Flatten());

  WriteRequest request;
  request.mutable_first_request_metadata()
      ->mutable_hpke_plus_aead_data()
      ->set_blob_id("blob_id");

  EXPECT_CALL(context_, EmitEncrypted(Eq(0), _)).WillOnce(Return(false));

  auto result = fn_->Write(request, checkpoint, context_);
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(SqlDataIngressFnTest, EmitEncryptedCheckpointSucceeds) {
  auto checkpoint = CreateInputCheckpoint({"example1", "example2", "example3"});
  ASSERT_THAT(checkpoint, IsOk());
  std::string checkpoint_str = std::move(checkpoint).value();

  WriteRequest request;
  request.mutable_first_request_metadata()
      ->mutable_hpke_plus_aead_data()
      ->set_blob_id("blob_id");

  std::string emitted_data;
  EXPECT_CALL(context_, EmitEncrypted(Eq(0), _))
      .WillOnce([&emitted_data](int, Session::KV kv) {
        emitted_data = std::move(kv.data);
        EXPECT_EQ(kv.blob_id, "blob_id");
        return true;
      });

  auto result = fn_->Write(request, checkpoint_str, context_);
  ASSERT_THAT(result, IsOk());

  // Parse the emitted checkpoint and verify the SQL result.
  FederatedComputeCheckpointParserFactory parser_factory;
  auto parser = parser_factory.Create(absl::Cord(emitted_data));
  ASSERT_THAT(parser, IsOk());

  auto result_tensor = (*parser)->GetTensor("result");
  ASSERT_THAT(result_tensor, IsOk());
  EXPECT_EQ(result_tensor->dtype(), DataType::DT_STRING);
  EXPECT_EQ(result_tensor->num_elements(), 3);
  EXPECT_THAT(result_tensor->AsSpan<absl::string_view>(),
              testing::ElementsAre("example1", "example2", "example3"));
}

TEST(ProvideSqlDataIngressFnFactoryTest, ParsesPrivateLoggerConfigSucceeds) {
  SqlQuery sql_query = PARSE_TEXT_PROTO(R"pb(
    raw_sql: "SELECT text AS result FROM input"
    database_schema {
      table {
        name: "input"
        column { name: "text" type: STRING }
        create_table_sql: "CREATE TABLE input (text TEXT)"
      }
    }
    output_columns { name: "result" type: STRING }
  )pb");
  SqlDataIngressContainerInitializeConfiguration init_config;
  *init_config.mutable_sql_query() = sql_query;

  auto* pl_config = init_config.mutable_private_logger_uploads_config();
  pl_config->set_on_device_query_name("test_query");

  google::protobuf::FileDescriptorSet fds;
  auto* fd = fds.add_file();
  fd->set_name("dummy.proto");
  auto* mt = fd->add_message_type();
  mt->set_name("dummy_name");

  auto* md = pl_config->mutable_message_description();
  md->set_message_descriptor_set(fds.SerializeAsString());
  md->set_message_name("dummy_name");

  Any config;
  config.PackFrom(init_config);

  auto factory = ProvideSqlDataIngressFnFactory(config, Any(), {});
  EXPECT_THAT(factory, IsOk());
}

}  // namespace
}  // namespace confidential_federated_compute::sql_data_ingress
