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

#include "program_executor_tee/program_context/cc/data_parser.h"

#include "absl/log/check.h"
#include "containers/crypto.h"
#include "containers/crypto_test_utils.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/data_read_write.pb.h"
#include "fcp/testing/testing.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "program_executor_tee/program_context/cc/generate_checkpoint.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/core/impl/executors/array_shape_test_utils.h"
#include "tensorflow_federated/cc/core/impl/executors/array_test_utils.h"

namespace confidential_federated_compute::program_executor_tee {
namespace {

using ::confidential_federated_compute::crypto_test_utils::MockSigningKeyHandle;
using ::fcp::EqualsProto;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::outgoing::ReadResponse;
using ::tensorflow_federated::aggregation::CreateTestData;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::testing::CreateArray;
using ::tensorflow_federated::testing::CreateArrayShape;
using ::tensorflow_federated::v0::Value;
using ::testing::HasSubstr;
using ::testing::NiceMock;

template <typename T>
std::pair<Tensor, Value> CreateAggTensorAndExpectedVal(
    ::tensorflow_federated::aggregation::DataType agg_datatype,
    ::federated_language::DataType value_datatype,
    std::initializer_list<int64_t> dims, std::initializer_list<T> test_data) {
  auto agg_tensor =
      Tensor::Create(agg_datatype, dims, CreateTestData<T>(test_data));
  CHECK_OK(agg_tensor);

  auto expected_val = tensorflow_federated::v0::Value();
  auto expected_array =
      CreateArray(value_datatype, CreateArrayShape(dims), test_data);
  CHECK_OK(expected_array);
  expected_val.mutable_array()->Swap(&*expected_array);

  return {std::move(*agg_tensor), std::move(expected_val)};
}

TEST(DataParserTest, ConvertAggCoreTensorToValue_Float) {
  auto [tensor, expected_val] = CreateAggTensorAndExpectedVal<float>(
      tensorflow_federated::aggregation::DT_FLOAT,
      federated_language::DataType::DT_FLOAT, {3}, {1, 2, 3});
  auto val = DataParser::ConvertAggCoreTensorToValue(tensor);
  ASSERT_OK(val);
  EXPECT_THAT(*val, EqualsProto(expected_val));
}

TEST(DataParserTest, ConvertAggCoreTensorToValue_Double) {
  auto [tensor, expected_val] = CreateAggTensorAndExpectedVal<double>(
      tensorflow_federated::aggregation::DT_DOUBLE,
      federated_language::DataType::DT_DOUBLE, {2, 2}, {1.0, 2.0, 3.0, 4.0});
  auto val = DataParser::ConvertAggCoreTensorToValue(tensor);
  ASSERT_OK(val);
  EXPECT_THAT(*val, EqualsProto(expected_val));
}

TEST(DataParserTest, ConvertAggCoreTensorToValue_Int32) {
  auto [tensor, expected_val] = CreateAggTensorAndExpectedVal<int32_t>(
      tensorflow_federated::aggregation::DT_INT32,
      federated_language::DataType::DT_INT32, {}, {1});
  auto val = DataParser::ConvertAggCoreTensorToValue(tensor);
  ASSERT_OK(val);
  EXPECT_THAT(*val, EqualsProto(expected_val));
}

TEST(DataParserTest, ConvertAggCoreTensorToValue_Int64) {
  auto [tensor, expected_val] = CreateAggTensorAndExpectedVal<int64_t>(
      tensorflow_federated::aggregation::DT_INT64,
      federated_language::DataType::DT_INT64, {2}, {1, 2});
  auto val = DataParser::ConvertAggCoreTensorToValue(tensor);
  ASSERT_OK(val);
  EXPECT_THAT(*val, EqualsProto(expected_val));
}

TEST(DataParserTest, ConvertAggCoreTensorToValue_Uint64) {
  auto [tensor, expected_val] = CreateAggTensorAndExpectedVal<uint64_t>(
      tensorflow_federated::aggregation::DT_UINT64,
      federated_language::DataType::DT_UINT64, {3}, {1, 2, 3});
  auto val = DataParser::ConvertAggCoreTensorToValue(tensor);
  ASSERT_OK(val);
  EXPECT_THAT(*val, EqualsProto(expected_val));
}

TEST(DataParserTest, ConvertAggCoreTensorToValue_String) {
  // The CreateAggTensorAndExpectedVal helper function cannot be used for this
  // test because the AggCore tensor construction requires a absl::string_view
  // type whereas the Array construction requires a std::string type.
  auto tensor =
      Tensor::Create(tensorflow_federated::aggregation::DT_STRING, {2},
                     CreateTestData<absl::string_view>({"hello", "bye"}));
  CHECK_OK(tensor);

  auto expected_val = tensorflow_federated::v0::Value();
  auto expected_array = CreateArray(federated_language::DataType::DT_STRING,
                                    CreateArrayShape({2}), {"hello", "bye"});
  CHECK_OK(expected_array);
  expected_val.mutable_array()->Swap(&*expected_array);

  auto val = DataParser::ConvertAggCoreTensorToValue(*tensor);
  ASSERT_OK(val);
  EXPECT_THAT(*val, EqualsProto(expected_val));
}

TEST(DataParserTest, ConvertAggCoreTensorToValue_InvalidType) {
  // The default AggCore tensor constructor uses DT_INVALID.
  Tensor tensor;
  auto val = DataParser::ConvertAggCoreTensorToValue(tensor);
  EXPECT_EQ(val.status(),
            absl::UnimplementedError("Unexpected DataType found: 0"));
}

TEST(DataParserTest, ParseReadResponseToValue_PlaintextIntCheckpoint) {
  std::vector<int> input_values = {4, 5, 6};
  std::string tensor_name = "tensor_name";
  std::string checkpoint =
      BuildClientCheckpointFromInts(input_values, tensor_name);

  ReadResponse response;
  BlobMetadata blob_metadata;
  blob_metadata.set_total_size_bytes(checkpoint.size());
  blob_metadata.mutable_unencrypted();
  blob_metadata.set_compression_type(BlobMetadata::COMPRESSION_TYPE_NONE);
  *response.mutable_first_response_metadata() = std::move(blob_metadata);
  response.set_finish_read(true);
  response.set_data(checkpoint);

  DataParser data_parser(/*blob_decryptor=*/nullptr);
  auto val = data_parser.ParseReadResponseToValue(response, "unused nonce",
                                                  tensor_name);
  ASSERT_OK(val);

  auto expected_val = tensorflow_federated::v0::Value();
  auto expected_array = CreateArray(federated_language::DataType::DT_INT32,
                                    CreateArrayShape({3}), {4, 5, 6});
  CHECK_OK(expected_array);
  expected_val.mutable_array()->Swap(&*expected_array);
  EXPECT_THAT(*val, EqualsProto(expected_val));
}

TEST(DataParserTest, ParseReadResponseToValue_EncryptedIntCheckpoint) {
  std::initializer_list<int> input_values = {4, 5, 6};
  std::string tensor_name = "tensor_name";
  std::string checkpoint =
      BuildClientCheckpointFromInts(input_values, tensor_name);

  NiceMock<MockSigningKeyHandle> mock_signing_key_handle;
  BlobDecryptor blob_decryptor(mock_signing_key_handle);
  std::string nonce = "nonce";
  ReadResponse response;
  std::tie(*response.mutable_first_response_metadata(),
           *response.mutable_data()) =
      crypto_test_utils::CreateRewrappedBlob(
          checkpoint, "ciphertext associated data",
          *blob_decryptor.GetPublicKey(), nonce, "reencryption_public_key")
          .value();
  response.set_finish_read(true);

  DataParser data_parser(&blob_decryptor);
  auto val = data_parser.ParseReadResponseToValue(response, nonce, tensor_name);
  ASSERT_OK(val);

  auto expected_val = tensorflow_federated::v0::Value();
  auto expected_array = CreateArray(federated_language::DataType::DT_INT32,
                                    CreateArrayShape({3}), {4, 5, 6});
  CHECK_OK(expected_array);
  expected_val.mutable_array()->Swap(&*expected_array);
  EXPECT_THAT(*val, EqualsProto(expected_val));
}

TEST(DataParserTest, ParseReadResponseToValue_EncryptedStringCheckpoint) {
  std::vector<std::string> input_values = {"serialized_example_1",
                                           "serialized_example_2"};
  std::string tensor_name = "tensor_name";
  std::string checkpoint =
      BuildClientCheckpointFromStrings(input_values, tensor_name);

  NiceMock<MockSigningKeyHandle> mock_signing_key_handle;
  BlobDecryptor blob_decryptor(mock_signing_key_handle);
  std::string nonce = "nonce";
  ReadResponse response;
  std::tie(*response.mutable_first_response_metadata(),
           *response.mutable_data()) =
      crypto_test_utils::CreateRewrappedBlob(
          checkpoint, "ciphertext associated data",
          *blob_decryptor.GetPublicKey(), nonce, "reencryption_public_key")
          .value();
  response.set_finish_read(true);

  DataParser data_parser(&blob_decryptor);
  auto val = data_parser.ParseReadResponseToValue(response, nonce, tensor_name);
  ASSERT_OK(val);

  auto expected_val = tensorflow_federated::v0::Value();
  auto expected_array = CreateArray(
      federated_language::DataType::DT_STRING, CreateArrayShape({2}),
      {"serialized_example_1", "serialized_example_2"});
  CHECK_OK(expected_array);
  expected_val.mutable_array()->Swap(&*expected_array);
  EXPECT_THAT(*val, EqualsProto(expected_val));
}

TEST(DataParserTest, ParseReadResponseToValue_MismatchedNonce) {
  std::vector<int> input_values = {4, 5, 6};
  std::string tensor_name = "tensor_name";
  std::string checkpoint =
      BuildClientCheckpointFromInts(input_values, tensor_name);

  NiceMock<MockSigningKeyHandle> mock_signing_key_handle;
  BlobDecryptor blob_decryptor(mock_signing_key_handle);
  ReadResponse response;
  std::tie(*response.mutable_first_response_metadata(),
           *response.mutable_data()) =
      crypto_test_utils::CreateRewrappedBlob(
          checkpoint, "ciphertext associated data",
          *blob_decryptor.GetPublicKey(), "nonce", "reencryption_public_key")
          .value();
  response.set_finish_read(true);

  DataParser data_parser(&blob_decryptor);
  auto val = data_parser.ParseReadResponseToValue(response, "different nonce",
                                                  tensor_name);
  EXPECT_EQ(val.status(),
            absl::InvalidArgumentError("ReadResponse nonce does not match"));
}

TEST(DataParserTest, ParseReadResponseToValue_IncorrectCheckpointFormat) {
  std::string message = "not a fc checkpoint";

  NiceMock<MockSigningKeyHandle> mock_signing_key_handle;
  BlobDecryptor blob_decryptor(mock_signing_key_handle);
  std::string nonce = "nonce";
  ReadResponse response;
  std::tie(*response.mutable_first_response_metadata(),
           *response.mutable_data()) =
      crypto_test_utils::CreateRewrappedBlob(
          message, "ciphertext associated data", *blob_decryptor.GetPublicKey(),
          nonce, "reencryption_public_key")
          .value();
  response.set_finish_read(true);

  DataParser data_parser(&blob_decryptor);
  auto val = data_parser.ParseReadResponseToValue(response, nonce,
                                                  "unused_tensor_name");
  EXPECT_EQ(val.status().code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(val.status().message(),
              HasSubstr("Unsupported checkpoint format"));
}

TEST(DataParserTest, ParseReadResponseToValue_IncorrectTensorName) {
  std::vector<int> input_values = {4, 5, 6};
  std::string checkpoint =
      BuildClientCheckpointFromInts(input_values, "tensor_name");

  NiceMock<MockSigningKeyHandle> mock_signing_key_handle;
  BlobDecryptor blob_decryptor(mock_signing_key_handle);
  std::string nonce = "nonce";
  ReadResponse response;
  std::tie(*response.mutable_first_response_metadata(),
           *response.mutable_data()) =
      crypto_test_utils::CreateRewrappedBlob(
          checkpoint, "ciphertext associated data",
          *blob_decryptor.GetPublicKey(), nonce, "reencryption_public_key")
          .value();
  response.set_finish_read(true);

  DataParser data_parser(&blob_decryptor);
  auto val = data_parser.ParseReadResponseToValue(response, nonce,
                                                  "different_tensor_name");
  EXPECT_EQ(val.status(),
            absl::NotFoundError(
                "No aggregation tensor found for name different_tensor_name"));
}

}  // namespace
}  // namespace confidential_federated_compute::program_executor_tee