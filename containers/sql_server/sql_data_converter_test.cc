#include "containers/sql_server/sql_data_converter.h"

#include "absl/log/check.h"
#include "containers/sql_server/sql_data.pb.h"
#include "fcp/aggregation/core/mutable_vector_data.h"
#include "fcp/aggregation/protocol/federated_compute_checkpoint_builder.h"
#include "fcp/aggregation/testing/test_data.h"
#include "fcp/client/example_query_result.pb.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.grpc.pb.h"
#include "fcp/protos/confidentialcompute/pipeline_transform.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute::sql_server {
using ::fcp::aggregation::CheckpointBuilder;
using ::fcp::aggregation::CreateTestData;
using ::fcp::aggregation::DataType;
using ::fcp::aggregation::FederatedComputeCheckpointBuilderFactory;
using ::fcp::aggregation::FederatedComputeCheckpointParserFactory;
using ::fcp::aggregation::Tensor;
using ::fcp::aggregation::TensorShape;
using ::fcp::client::ExampleQueryResult_VectorData;
using ::fcp::client::ExampleQueryResult_VectorData_BoolValues;
using ::fcp::client::ExampleQueryResult_VectorData_Values;
using ::fcp::confidentialcompute::Record;
using ::fcp::confidentialcompute::TransformRequest;
using ::sql_data::ColumnSchema;
using ::sql_data::ColumnSchema_DataType;
using ::sql_data::SqlData;
using ::sql_data::TableSchema;
using ::testing::HasSubstr;
using ::testing::Test;

namespace sql_data_converter_internal {

namespace {

TEST(ConvertValuesToTensorTest, Int32ValuesConvertedCorrectly) {
  ExampleQueryResult_VectorData_Values values;
  values.mutable_int32_values()->add_value(42);
  auto tensor = ConvertValuesToTensor(values);
  ASSERT_TRUE(tensor.ok());
  ASSERT_EQ(tensor->dtype(), DataType::DT_INT32);
  ASSERT_EQ(tensor->shape(), TensorShape({1}));
}

TEST(ConvertValuesToTensorTest, Int64ValuesConvertedCorrectly) {
  ExampleQueryResult_VectorData_Values values;
  values.mutable_int64_values()->add_value(42);
  auto tensor = ConvertValuesToTensor(values);
  ASSERT_TRUE(tensor.ok());
  ASSERT_EQ(tensor->dtype(), DataType::DT_INT64);
  ASSERT_EQ(tensor->shape(), TensorShape({1}));
}

TEST(ConvertValuesToTensorTest, BoolValuesConvertedCorrectly) {
  ExampleQueryResult_VectorData_Values values;
  values.mutable_bool_values()->add_value(true);
  auto tensor = ConvertValuesToTensor(values);
  ASSERT_TRUE(tensor.ok());
  ASSERT_EQ(tensor->dtype(), DataType::DT_INT32);
  ASSERT_EQ(tensor->shape(), TensorShape({1}));
}

TEST(ConvertValuesToTensorTest, FloatValuesConvertedCorrectly) {
  ExampleQueryResult_VectorData_Values values;
  values.mutable_float_values()->add_value(1.1);
  auto tensor = ConvertValuesToTensor(values);
  ASSERT_TRUE(tensor.ok());
  ASSERT_EQ(tensor->dtype(), DataType::DT_FLOAT);
  ASSERT_EQ(tensor->shape(), TensorShape({1}));
}

TEST(ConvertValuesToTensorTest, DoubleValuesConvertedCorrectly) {
  ExampleQueryResult_VectorData_Values values;
  values.mutable_double_values()->add_value(1.1);
  auto tensor = ConvertValuesToTensor(values);
  ASSERT_TRUE(tensor.ok());
  ASSERT_EQ(tensor->dtype(), DataType::DT_DOUBLE);
  ASSERT_EQ(tensor->shape(), TensorShape({1}));
}

TEST(ConvertValuesToTensorTest, StringValuesConvertedCorrectly) {
  ExampleQueryResult_VectorData_Values values;
  values.mutable_string_values()->add_value("foo");
  auto tensor = ConvertValuesToTensor(values);
  ASSERT_TRUE(tensor.ok());
  ASSERT_EQ(tensor->dtype(), DataType::DT_STRING);
  ASSERT_EQ(tensor->shape(), TensorShape({1}));
}

TEST(ConvertValuesToTensorTest, BytesValuesConvertedCorrectly) {
  ExampleQueryResult_VectorData_Values values;
  values.mutable_bytes_values()->add_value("foo");
  auto tensor = ConvertValuesToTensor(values);
  ASSERT_TRUE(tensor.ok());
  ASSERT_EQ(tensor->dtype(), DataType::DT_STRING);
  ASSERT_EQ(tensor->shape(), TensorShape({1}));
}

TEST(ConvertValuesToTensorTest, EmptyValuesConvertedCorrectly) {
  ExampleQueryResult_VectorData_Values values;
  values.mutable_bytes_values();
  auto tensor = ConvertValuesToTensor(values);
  ASSERT_TRUE(tensor.ok());
  ASSERT_EQ(tensor->dtype(), DataType::DT_STRING);
  ASSERT_EQ(tensor->shape(), TensorShape({0}));
}

TEST(ConvertValuesToTensorTest, MultipleValuesConvertedCorrectly) {
  ExampleQueryResult_VectorData_Values values;
  auto* bool_values = values.mutable_bool_values();
  bool_values->add_value(false);
  bool_values->add_value(true);
  auto tensor = ConvertValuesToTensor(values);
  ASSERT_TRUE(tensor.ok());
  ASSERT_EQ(tensor->dtype(), DataType::DT_INT32);
  ASSERT_EQ(tensor->shape(), TensorShape({2}));
  for (const auto& [index, value] : tensor->AsAggVector<int32_t>()) {
    ASSERT_EQ(index, value);
  }
}

}  // namespace

}  // namespace sql_data_converter_internal

namespace {

class ConvertWireFormatRecordsToSqlDataTest : public Test {
 protected:
  void SetColumnNameAndType(ColumnSchema* col, std::string name,
                            ColumnSchema_DataType type) {
    col->set_name(name);
    col->set_type(type);
  }
};

TEST_F(ConvertWireFormatRecordsToSqlDataTest, BasicUsage) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> builder = builder_factory.Create();
  std::string col_name = "t1";

  absl::StatusOr<Tensor> t1 = Tensor::Create(
      DataType::DT_INT64, TensorShape({1}), CreateTestData<uint64_t>({42}));
  CHECK_OK(t1);
  CHECK_OK(builder->Add(col_name, *t1));
  auto checkpoint = builder->Build();
  CHECK_OK(checkpoint.status());

  TableSchema schema;
  SetColumnNameAndType(schema.add_column(), col_name, ColumnSchema::INT64);

  TransformRequest request;
  std::string checkpoint_string;
  absl::CopyCordToString(*checkpoint, &checkpoint_string);
  Record* record = request.add_inputs();
  record->set_unencrypted_data(checkpoint_string);

  auto sql_data = ConvertWireFormatRecordsToSqlData(&request, schema);
  ASSERT_TRUE(sql_data.ok());
  ASSERT_TRUE(sql_data->vector_data().vectors().contains(col_name));
  ASSERT_TRUE(
      sql_data->vector_data().vectors().at(col_name).has_int64_values());
  ASSERT_EQ(
      sql_data->vector_data().vectors().at(col_name).int64_values().value(0),
      42);
  ASSERT_EQ(sql_data->num_rows(), 1);
}

TEST_F(ConvertWireFormatRecordsToSqlDataTest, AllDataTypes) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> builder = builder_factory.Create();

  std::string int32_col_name = "int32_col";
  std::string int64_col_name = "int64_col";
  std::string str_col_name = "str_col";
  std::string double_col_name = "double_col";
  std::string float_col_name = "float_col";
  std::string bytes_col_name = "bytes_col";
  std::string bool_col_name = "bool_col";

  absl::StatusOr<Tensor> int32_tensor = Tensor::Create(
      DataType::DT_INT32, TensorShape({1}), CreateTestData<uint32_t>({32}));
  absl::StatusOr<Tensor> int64_tensor = Tensor::Create(
      DataType::DT_INT64, TensorShape({1}), CreateTestData<uint64_t>({64}));
  absl::StatusOr<Tensor> str_tensor =
      Tensor::Create(DataType::DT_STRING, TensorShape({1}),
                     CreateTestData<absl::string_view>({"a"}));
  absl::StatusOr<Tensor> double_tensor = Tensor::Create(
      DataType::DT_DOUBLE, TensorShape({1}), CreateTestData<double>({1.1}));
  absl::StatusOr<Tensor> float_tensor = Tensor::Create(
      DataType::DT_FLOAT, TensorShape({1}), CreateTestData<float>({2.2}));
  // Tensor doesn't currently have bytes or bool data type.
  absl::StatusOr<Tensor> bytes_tensor =
      Tensor::Create(DataType::DT_STRING, TensorShape({1}),
                     CreateTestData<absl::string_view>({"b"}));
  absl::StatusOr<Tensor> bool_tensor = Tensor::Create(
      DataType::DT_INT32, TensorShape({1}), CreateTestData<uint32_t>({0}));

  CHECK_OK(int32_tensor);
  CHECK_OK(int64_tensor);
  CHECK_OK(str_tensor);
  CHECK_OK(double_tensor);
  CHECK_OK(float_tensor);

  CHECK_OK(builder->Add(int32_col_name, *int32_tensor));
  CHECK_OK(builder->Add(int64_col_name, *int64_tensor));
  CHECK_OK(builder->Add(str_col_name, *str_tensor));
  CHECK_OK(builder->Add(double_col_name, *double_tensor));
  CHECK_OK(builder->Add(float_col_name, *float_tensor));
  CHECK_OK(builder->Add(bytes_col_name, *bytes_tensor));
  CHECK_OK(builder->Add(bool_col_name, *bool_tensor));

  auto checkpoint = builder->Build();
  CHECK_OK(checkpoint.status());

  TableSchema schema;
  SetColumnNameAndType(schema.add_column(), int32_col_name,
                       ColumnSchema::INT32);
  SetColumnNameAndType(schema.add_column(), int64_col_name,
                       ColumnSchema::INT64);
  SetColumnNameAndType(schema.add_column(), str_col_name, ColumnSchema::STRING);
  SetColumnNameAndType(schema.add_column(), double_col_name,
                       ColumnSchema::DOUBLE);
  SetColumnNameAndType(schema.add_column(), float_col_name,
                       ColumnSchema::FLOAT);
  SetColumnNameAndType(schema.add_column(), bytes_col_name,
                       ColumnSchema::BYTES);
  SetColumnNameAndType(schema.add_column(), bool_col_name, ColumnSchema::BOOL);

  TransformRequest request;

  std::string checkpoint_string;
  absl::CopyCordToString(*checkpoint, &checkpoint_string);
  Record* record = request.add_inputs();
  record->set_unencrypted_data(checkpoint_string);

  auto sql_data = ConvertWireFormatRecordsToSqlData(&request, schema);
  ASSERT_TRUE(sql_data.ok());
  auto vectors = sql_data->vector_data().vectors();
  ASSERT_TRUE(vectors.contains(int32_col_name));
  ASSERT_TRUE(vectors.contains(int64_col_name));
  ASSERT_TRUE(vectors.contains(str_col_name));
  ASSERT_TRUE(vectors.contains(double_col_name));
  ASSERT_TRUE(vectors.contains(float_col_name));
  ASSERT_TRUE(vectors.contains(bytes_col_name));
  ASSERT_TRUE(vectors.contains(bool_col_name));
}

TEST_F(ConvertWireFormatRecordsToSqlDataTest, EmptyCheckpoint) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> builder = builder_factory.Create();
  std::string col_name = "t1";

  absl::StatusOr<Tensor> t1 = Tensor::Create(
      DataType::DT_INT64, TensorShape({0}), CreateTestData<uint64_t>({}));
  CHECK_OK(t1);
  CHECK_OK(builder->Add(col_name, *t1));
  auto checkpoint = builder->Build();
  CHECK_OK(checkpoint.status());

  TableSchema schema;
  SetColumnNameAndType(schema.add_column(), col_name, ColumnSchema::INT64);

  TransformRequest request;
  std::string checkpoint_string;
  absl::CopyCordToString(*checkpoint, &checkpoint_string);
  Record* record = request.add_inputs();
  record->set_unencrypted_data(checkpoint_string);

  auto sql_data = ConvertWireFormatRecordsToSqlData(&request, schema);
  ASSERT_TRUE(sql_data.ok());
  ASSERT_TRUE(sql_data->vector_data().vectors().contains(col_name));
  ASSERT_FALSE(
      sql_data->vector_data().vectors().at(col_name).has_int64_values());
}

TEST_F(ConvertWireFormatRecordsToSqlDataTest, DifferentLengthTensors) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> builder = builder_factory.Create();
  std::string one_val_name = "one_val";
  std::string two_vals_name = "two_vals";

  absl::StatusOr<Tensor> t1 = Tensor::Create(
      DataType::DT_INT64, TensorShape({1}), CreateTestData<uint64_t>({42}));
  absl::StatusOr<Tensor> t2 = Tensor::Create(
      DataType::DT_FLOAT, TensorShape({2}), CreateTestData<float>({1.1, 2.2}));
  CHECK_OK(t1);
  CHECK_OK(t2);
  CHECK_OK(builder->Add(one_val_name, *t1));
  CHECK_OK(builder->Add(two_vals_name, *t2));
  auto checkpoint = builder->Build();
  CHECK_OK(checkpoint.status());

  TableSchema schema;
  SetColumnNameAndType(schema.add_column(), one_val_name, ColumnSchema::INT64);
  SetColumnNameAndType(schema.add_column(), two_vals_name, ColumnSchema::FLOAT);

  TransformRequest request;
  std::string checkpoint_string;
  absl::CopyCordToString(*checkpoint, &checkpoint_string);
  Record* record = request.add_inputs();
  record->set_unencrypted_data(checkpoint_string);

  auto sql_data = ConvertWireFormatRecordsToSqlData(&request, schema);
  ASSERT_TRUE(absl::IsInvalidArgument(sql_data.status()));
  ASSERT_THAT(sql_data.status().message(),
              HasSubstr("Record has columns with differing numbers of rows."));
}

TEST_F(ConvertWireFormatRecordsToSqlDataTest, TableSchemaTypeMismatch) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> builder = builder_factory.Create();
  std::string col_name = "col_name";

  absl::StatusOr<Tensor> t1 = Tensor::Create(
      DataType::DT_INT64, TensorShape({1}), CreateTestData<uint64_t>({42}));
  CHECK_OK(t1);
  CHECK_OK(builder->Add(col_name, *t1));
  auto checkpoint = builder->Build();
  CHECK_OK(checkpoint.status());

  TableSchema schema;
  SetColumnNameAndType(schema.add_column(), col_name, ColumnSchema::STRING);
  TransformRequest request;
  std::string checkpoint_string;
  absl::CopyCordToString(*checkpoint, &checkpoint_string);
  Record* record = request.add_inputs();
  record->set_unencrypted_data(checkpoint_string);

  auto sql_data = ConvertWireFormatRecordsToSqlData(&request, schema);
  ASSERT_TRUE(absl::IsInvalidArgument(sql_data.status()));
  ASSERT_THAT(sql_data.status().message(),
              HasSubstr("Checkpoint column type does not match the column "
                        "type specified in the TableSchema."));
}

TEST_F(ConvertWireFormatRecordsToSqlDataTest, MultipleRecords) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> builder1 = builder_factory.Create();
  std::unique_ptr<CheckpointBuilder> builder2 = builder_factory.Create();
  std::string col_name = "col_name";

  absl::StatusOr<Tensor> t1 = Tensor::Create(
      DataType::DT_INT64, TensorShape({1}), CreateTestData<uint64_t>({42}));
  absl::StatusOr<Tensor> t2 = Tensor::Create(
      DataType::DT_INT64, TensorShape({2}), CreateTestData<uint64_t>({1, 2}));
  CHECK_OK(t1);
  CHECK_OK(t2);
  CHECK_OK(builder1->Add(col_name, *t1));
  CHECK_OK(builder2->Add(col_name, *t2));
  auto checkpoint1 = builder1->Build();
  auto checkpoint2 = builder2->Build();
  CHECK_OK(checkpoint1.status());

  TableSchema schema;
  SetColumnNameAndType(schema.add_column(), col_name, ColumnSchema::INT64);

  TransformRequest request;
  std::string checkpoint_string1;
  std::string checkpoint_string2;
  absl::CopyCordToString(*checkpoint1, &checkpoint_string1);
  absl::CopyCordToString(*checkpoint2, &checkpoint_string2);
  Record* record1 = request.add_inputs();
  record1->set_unencrypted_data(checkpoint_string1);
  Record* record2 = request.add_inputs();
  record2->set_unencrypted_data(checkpoint_string2);

  auto sql_data = ConvertWireFormatRecordsToSqlData(&request, schema);
  ASSERT_TRUE(sql_data.ok());
  auto vectors = sql_data->vector_data().vectors();
  ASSERT_TRUE(vectors.contains(col_name));
  ASSERT_TRUE(vectors.at(col_name).has_int64_values());
  ASSERT_EQ(vectors.at(col_name).int64_values().value(0), 42);
  ASSERT_EQ(vectors.at(col_name).int64_values().value(1), 1);
  ASSERT_EQ(vectors.at(col_name).int64_values().value(2), 2);
  ASSERT_EQ(vectors.at(col_name).int64_values().value_size(), 3);
  ASSERT_EQ(sql_data->num_rows(), 3);
}

TEST_F(ConvertWireFormatRecordsToSqlDataTest, ByteColumnConvertedCorrectly) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> builder = builder_factory.Create();
  std::string col_name = "col_name";

  // Tensor doesn't have a bytes data type.
  absl::StatusOr<Tensor> t1 =
      Tensor::Create(DataType::DT_STRING, TensorShape({1}),
                     CreateTestData<absl::string_view>({"bytes"}));
  CHECK_OK(t1);
  CHECK_OK(builder->Add(col_name, *t1));
  auto checkpoint = builder->Build();
  CHECK_OK(checkpoint.status());

  TableSchema schema;
  SetColumnNameAndType(schema.add_column(), col_name, ColumnSchema::BYTES);

  TransformRequest request;
  std::string checkpoint_string;
  absl::CopyCordToString(*checkpoint, &checkpoint_string);
  Record* record = request.add_inputs();
  record->set_unencrypted_data(checkpoint_string);

  auto sql_data = ConvertWireFormatRecordsToSqlData(&request, schema);
  ASSERT_TRUE(sql_data.ok());
  auto vectors = sql_data->vector_data().vectors();
  ASSERT_TRUE(vectors.contains(col_name));
  ASSERT_TRUE(vectors.at(col_name).has_bytes_values());
  ASSERT_EQ(vectors.at(col_name).bytes_values().value(0), "bytes");
}

TEST_F(ConvertWireFormatRecordsToSqlDataTest, BoolColumnConvertedCorrectly) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> builder = builder_factory.Create();
  std::string col_name = "col_name";

  // Tensor doesn't currently have a bool data type.
  absl::StatusOr<Tensor> t1 = Tensor::Create(
      DataType::DT_INT32, TensorShape({2}), CreateTestData<uint32_t>({0, 1}));
  CHECK_OK(t1);
  CHECK_OK(builder->Add(col_name, *t1));
  auto checkpoint = builder->Build();
  CHECK_OK(checkpoint.status());

  TableSchema schema;
  SetColumnNameAndType(schema.add_column(), col_name, ColumnSchema::BOOL);

  TransformRequest request;
  std::string checkpoint_string;
  absl::CopyCordToString(*checkpoint, &checkpoint_string);
  Record* record = request.add_inputs();
  record->set_unencrypted_data(checkpoint_string);

  auto sql_data = ConvertWireFormatRecordsToSqlData(&request, schema);
  ASSERT_TRUE(sql_data.ok());
  auto vectors = sql_data->vector_data().vectors();
  ASSERT_TRUE(vectors.contains(col_name));
  ASSERT_TRUE(vectors.at(col_name).has_bool_values());
  ASSERT_EQ(vectors.at(col_name).bool_values().value(0), false);
  ASSERT_EQ(vectors.at(col_name).bool_values().value(1), true);
}

TEST(ConvertSqlDataToWireFormatTest, EmptyColumn) {
  SqlData sql_data;
  ExampleQueryResult_VectorData* vector_data = sql_data.mutable_vector_data();
  ExampleQueryResult_VectorData_Values empty_val;
  empty_val.mutable_bool_values();
  (*vector_data->mutable_vectors())["empty_col"] = empty_val;
  FederatedComputeCheckpointParserFactory parser_factory;

  auto response = ConvertSqlDataToWireFormat(std::move(sql_data));
  ASSERT_TRUE(response.ok());
  ASSERT_EQ(response->outputs_size(), 1);

  absl::Cord ckpt(response->outputs(0).unencrypted_data());
  auto parser = parser_factory.Create(ckpt);
  ASSERT_TRUE(parser.ok());
  auto empty_col_values = (*parser)->GetTensor("empty_col");
  ASSERT_EQ(empty_col_values->num_elements(), 0);
}

TEST(ConvertSqlDataToWireFormatTest, EmptySqlData) {
  SqlData sql_data;
  FederatedComputeCheckpointParserFactory parser_factory;

  auto response = ConvertSqlDataToWireFormat(std::move(sql_data));
  ASSERT_TRUE(response.ok());
  ASSERT_EQ(response->outputs_size(), 1);

  absl::Cord ckpt(response->outputs(0).unencrypted_data());
  auto parser = parser_factory.Create(ckpt);
  ASSERT_TRUE(parser.ok());
}

TEST(ConvertSqlDataToWireFormatTest, MultipleColumns) {
  SqlData sql_data;
  ExampleQueryResult_VectorData* vector_data = sql_data.mutable_vector_data();

  ExampleQueryResult_VectorData_Values int_val;
  int_val.mutable_int64_values()->add_value(100);

  ExampleQueryResult_VectorData_Values float_val;
  float_val.mutable_float_values()->add_value(2.2);

  (*vector_data->mutable_vectors())["int_val"] = int_val;
  (*vector_data->mutable_vectors())["float_val"] = float_val;
  FederatedComputeCheckpointParserFactory parser_factory;

  auto response = ConvertSqlDataToWireFormat(std::move(sql_data));
  ASSERT_TRUE(response.ok());
  ASSERT_EQ(response->outputs_size(), 1);

  absl::Cord ckpt(response->outputs(0).unencrypted_data());
  auto parser = parser_factory.Create(ckpt);
  auto int_col_values = (*parser)->GetTensor("int_val");
  ASSERT_EQ(int_col_values->num_elements(), 1);
  ASSERT_EQ(int_col_values->dtype(), DataType::DT_INT64);
  auto float_col_values = (*parser)->GetTensor("float_val");
  ASSERT_EQ(float_col_values->num_elements(), 1);
  ASSERT_EQ(float_col_values->dtype(), DataType::DT_FLOAT);
}

}  // namespace

}  // namespace confidential_federated_compute::sql_server
