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

#include "containers/fns/global_reduce_fn.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "containers/session.h"
#include "containers/testing/mocks.h"
#include "fcp/protos/confidentialcompute/blob_header.pb.h"
#include "fcp/protos/confidentialcompute/confidential_transform.pb.h"
#include "fcp/protos/confidentialcompute/construct_user_session.pb.h"
#include "gmock/gmock.h"
#include "google/protobuf/any.pb.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute::fns {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::fcp::confidentialcompute::AssociatedMetadata;
using ::fcp::confidentialcompute::BlobHeader;
using ::fcp::confidentialcompute::BlobMetadata;
using ::fcp::confidentialcompute::CommitRequest;
using ::fcp::confidentialcompute::CommitResponse;
using ::fcp::confidentialcompute::ConfigureRequest;
using ::fcp::confidentialcompute::FinalizeRequest;
using ::fcp::confidentialcompute::FinalizeResponse;
using ::fcp::confidentialcompute::PrivacyIdMetadata;
using ::fcp::confidentialcompute::ProtectedMetadata;
using ::fcp::confidentialcompute::WriteFinishedResponse;
using ::fcp::confidentialcompute::WriteRequest;
using ::testing::_;
using ::testing::AllOf;
using ::testing::DoAll;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::Optional;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::SaveArg;
using ::testing::SizeIs;
using ::testing::StrictMock;
using Element = GlobalReduceFn::Element;
using ReduceContext = GlobalReduceFn::ReduceContext;
using ReleasableOutput = GlobalReduceFn::ReleasableOutput;

class MockGlobalReduceFn : public GlobalReduceFn {
 public:
  MOCK_METHOD(absl::StatusOr<ReleasableOutput>, Reduce,
              (std::vector<Element> inputs, ReduceContext& context),
              (override));
};

// A MockGlobalReduceFn that releases its output with a non-default
// reencryption key.
class MockGlobalReduceFnWithReencryptionKeyIndex : public MockGlobalReduceFn {
 public:
  int GetReencryptionKeyIndex() const override { return 2; }
};

// Returns a WriteRequest for an unencrypted blob with the given blob ID.
WriteRequest CreateWriteRequest(absl::string_view blob_id) {
  WriteRequest request;
  request.mutable_first_request_metadata()->mutable_unencrypted()->set_blob_id(
      std::string(blob_id));
  return request;
}

// Returns protected metadata containing a PrivacyIdMetadata entry with the
// given privacy ID.
ProtectedMetadata CreatePrivacyIdMetadata(absl::string_view privacy_id) {
  PrivacyIdMetadata privacy_id_metadata;
  privacy_id_metadata.set_privacy_id(std::string(privacy_id));
  ProtectedMetadata protected_metadata;
  protected_metadata.add_metadata()->PackFrom(privacy_id_metadata);
  return protected_metadata;
}

ReleasableOutput CreateOutput(
    absl::string_view data = "output",
    std::optional<std::string> src_state = "src_state",
    std::string dst_state = "dst_state") {
  return ReleasableOutput{.kv = Session::KV(std::string(data)),
                          .src_state = std::move(src_state),
                          .dst_state = std::move(dst_state)};
}

class GlobalReduceFnTest : public testing::Test {
 protected:
  // Commits the Fn and returns the result of finalizing it.
  absl::StatusOr<FinalizeResponse> CommitAndFinalize(GlobalReduceFn& fn) {
    EXPECT_THAT(fn.Commit(CommitRequest(), context_), IsOk());
    return fn.Finalize(FinalizeRequest(), BlobMetadata(), context_);
  }

  StrictMock<MockGlobalReduceFn> fn_;
  StrictMock<MockContext> context_;
};

TEST_F(GlobalReduceFnTest, WriteReturnsCommittedSize) {
  absl::StatusOr<WriteFinishedResponse> response =
      fn_.Write(WriteRequest(), "data_1", context_);
  ASSERT_THAT(response, IsOk());
  EXPECT_EQ(response->committed_size_bytes(), 6);
}

TEST_F(GlobalReduceFnTest, CommitReportsNumberOfInputsWithoutCallingReduce) {
  ASSERT_THAT(fn_.Write(WriteRequest(), "data_1", context_), IsOk());
  ASSERT_THAT(fn_.Write(WriteRequest(), "data_2", context_), IsOk());

  // Reduce() isn't called until Finalize().
  absl::StatusOr<CommitResponse> response =
      fn_.Commit(CommitRequest(), context_);
  ASSERT_THAT(response, IsOk());
  EXPECT_EQ(response->stats().num_inputs_committed(), 2);
}

TEST_F(GlobalReduceFnTest, FinalizeCallsReduceOnceWithAllInputsInWriteOrder) {
  // An unencrypted input with a key.
  WriteRequest request_1 = CreateWriteRequest("blob_1");
  request_1.mutable_first_request_configuration()->set_type_url("key_1");
  ASSERT_THAT(fn_.Write(request_1, "data_1", context_), IsOk());

  // An encrypted input with associated and protected metadata. Its blob ID is
  // taken from the BlobHeader in its associated metadata.
  BlobHeader header;
  header.set_blob_id("blob_2");
  header.set_key_id("key_id_2");
  AssociatedMetadata associated_metadata;
  associated_metadata.add_metadata()->PackFrom(header);
  WriteRequest request_2;
  request_2.mutable_first_request_metadata()
      ->mutable_hpke_plus_aead_data()
      ->set_blob_id("unused_blob_id");
  request_2.mutable_first_request_metadata()
      ->mutable_hpke_plus_aead_data()
      ->mutable_kms_symmetric_key_associated_data()
      ->mutable_associated_metadata()
      ->PackFrom(associated_metadata);
  BlobHeader protected_header;
  protected_header.set_blob_id("protected_blob_id");
  ProtectedMetadata protected_metadata;
  protected_metadata.add_metadata()->PackFrom(protected_header);
  ASSERT_THAT(fn_.Write(request_2, "data_2", protected_metadata, context_),
              IsOk());

  ASSERT_THAT(fn_.Write(CreateWriteRequest("blob_3"), "data_3", context_),
              IsOk());

  std::vector<Element> inputs;
  EXPECT_CALL(fn_, Reduce(_, _))
      .WillOnce(DoAll(SaveArg<0>(&inputs), Return(CreateOutput())));
  EXPECT_CALL(context_, EmitReleasable(0, _, _, _, _)).WillOnce(Return(true));
  ASSERT_THAT(CommitAndFinalize(fn_), IsOk());

  ASSERT_THAT(inputs, SizeIs(3));
  EXPECT_EQ(inputs[0].kv.data, "data_1");
  EXPECT_EQ(inputs[0].kv.blob_id, "blob_1");
  EXPECT_EQ(inputs[0].kv.key.type_url(), "key_1");
  EXPECT_EQ(inputs[0].associated_metadata.metadata_size(), 0);
  EXPECT_EQ(inputs[0].protected_metadata.metadata_size(), 0);

  EXPECT_EQ(inputs[1].kv.data, "data_2");
  EXPECT_EQ(inputs[1].kv.blob_id, "blob_2");
  BlobHeader unpacked_header;
  ASSERT_EQ(inputs[1].associated_metadata.metadata_size(), 1);
  ASSERT_TRUE(
      inputs[1].associated_metadata.metadata(0).UnpackTo(&unpacked_header));
  EXPECT_EQ(unpacked_header.key_id(), "key_id_2");
  ASSERT_EQ(inputs[1].protected_metadata.metadata_size(), 1);
  ASSERT_TRUE(
      inputs[1].protected_metadata.metadata(0).UnpackTo(&unpacked_header));
  EXPECT_EQ(unpacked_header.blob_id(), "protected_blob_id");

  EXPECT_EQ(inputs[2].kv.data, "data_3");
  EXPECT_EQ(inputs[2].kv.blob_id, "blob_3");
}

TEST_F(GlobalReduceFnTest, FinalizeWithoutWritesCallsReduceWithNoInputs) {
  // Reduce() is still called; it's up to the subclass to decide whether empty
  // input is an error.
  EXPECT_CALL(fn_, Reduce(IsEmpty(), _)).WillOnce(Return(CreateOutput()));
  EXPECT_CALL(context_, EmitReleasable(0, _, _, _, _)).WillOnce(Return(true));
  EXPECT_THAT(CommitAndFinalize(fn_), IsOk());
}

TEST_F(GlobalReduceFnTest, FinalizeReturnsReduceErrorWithoutEmitting) {
  ASSERT_THAT(fn_.Write(WriteRequest(), "data", context_), IsOk());
  // Nothing is emitted, so EmitReleasable() isn't expected to be called.
  EXPECT_CALL(fn_, Reduce(_, _))
      .WillOnce(Return(absl::InternalError("computation failed")));
  EXPECT_THAT(
      CommitAndFinalize(fn_),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("computation failed")));
}

TEST_F(GlobalReduceFnTest, ReduceCanIncrementCounters) {
  EXPECT_CALL(context_, GetCounters())
      .WillRepeatedly(ReturnRef(context_.counters_));
  EXPECT_CALL(fn_, Reduce(_, _))
      .WillOnce([](std::vector<Element> inputs,
                   ReduceContext& context) -> absl::StatusOr<ReleasableOutput> {
        context.IncrementCounter("reduce-calls");
        context.IncrementCounterBy("reduce-inputs", inputs.size());
        return CreateOutput();
      });
  EXPECT_CALL(context_, EmitReleasable(0, _, _, _, _)).WillOnce(Return(true));

  ASSERT_THAT(fn_.Write(WriteRequest(), "data_1", context_), IsOk());
  ASSERT_THAT(fn_.Write(WriteRequest(), "data_2", context_), IsOk());
  ASSERT_THAT(CommitAndFinalize(fn_), IsOk());
  EXPECT_EQ(context_.counters_["reduce-calls"], 1);
  EXPECT_EQ(context_.counters_["reduce-inputs"], 2);
}

TEST_F(GlobalReduceFnTest, WriteAfterCommitFails) {
  ASSERT_THAT(fn_.Commit(CommitRequest(), context_), IsOk());
  EXPECT_THAT(fn_.Write(WriteRequest(), "data", context_),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(GlobalReduceFnTest, SecondCommitFails) {
  ASSERT_THAT(fn_.Commit(CommitRequest(), context_), IsOk());
  EXPECT_THAT(fn_.Commit(CommitRequest(), context_),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(GlobalReduceFnTest, FinalizeEmitsOutputAsReleasable) {
  EXPECT_CALL(fn_, Reduce(_, _))
      .WillOnce(Return(CreateOutput("result", "src_state", "dst_state")));
  ASSERT_THAT(fn_.Write(WriteRequest(), "data", context_), IsOk());

  Session::KV released_kv;
  EXPECT_CALL(context_, EmitReleasable(0, _, Optional(Eq("src_state")),
                                       Eq("dst_state"), _))
      .WillOnce([&released_kv](int, Session::KV kv,
                               std::optional<absl::string_view>,
                               absl::string_view, std::string& release_token) {
        released_kv = std::move(kv);
        release_token = "release_token";
        return true;
      });
  absl::StatusOr<FinalizeResponse> response = CommitAndFinalize(fn_);
  ASSERT_THAT(response, IsOk());
  EXPECT_EQ(response->release_token(), "release_token");
  EXPECT_EQ(released_kv.data, "result");
}

TEST_F(GlobalReduceFnTest, FinalizeEmitsOutputWithoutSrcState) {
  EXPECT_CALL(fn_, Reduce(_, _))
      .WillOnce(Return(CreateOutput("result", std::nullopt, "dst_state")));
  EXPECT_CALL(context_,
              EmitReleasable(0, _, Eq(std::nullopt), Eq("dst_state"), _))
      .WillOnce(Return(true));
  EXPECT_THAT(CommitAndFinalize(fn_), IsOk());
}

TEST_F(GlobalReduceFnTest, FinalizeUsesReencryptionKeyIndex) {
  StrictMock<MockGlobalReduceFnWithReencryptionKeyIndex> fn;
  EXPECT_CALL(fn, Reduce(_, _)).WillOnce(Return(CreateOutput()));
  EXPECT_CALL(context_, EmitReleasable(2, _, _, _, _)).WillOnce(Return(true));
  EXPECT_THAT(CommitAndFinalize(fn), IsOk());
}

TEST_F(GlobalReduceFnTest, FinalizeWithoutCommitFails) {
  // Reduce() isn't called.
  EXPECT_THAT(fn_.Finalize(FinalizeRequest(), BlobMetadata(), context_),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(GlobalReduceFnTest, SecondFinalizeFails) {
  // Reduce() is only called once.
  EXPECT_CALL(fn_, Reduce(_, _)).WillOnce(Return(CreateOutput()));
  EXPECT_CALL(context_, EmitReleasable(0, _, _, _, _)).WillOnce(Return(true));
  ASSERT_THAT(CommitAndFinalize(fn_), IsOk());
  EXPECT_THAT(fn_.Finalize(FinalizeRequest(), BlobMetadata(), context_),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(GlobalReduceFnTest, FinalizeFailsWhenEmitReleasableFails) {
  EXPECT_CALL(fn_, Reduce(_, _)).WillOnce(Return(CreateOutput()));
  EXPECT_CALL(context_, EmitReleasable(0, _, _, _, _)).WillOnce(Return(false));
  EXPECT_THAT(CommitAndFinalize(fn_), StatusIs(absl::StatusCode::kInternal));
}

TEST_F(GlobalReduceFnTest, FullLifecycleViaConfigure) {
  ASSERT_THAT(fn_.Configure(ConfigureRequest(), context_), IsOk());
  ASSERT_THAT(fn_.Write(CreateWriteRequest("blob_1"), "data_1",
                        CreatePrivacyIdMetadata("privacy_id_1"), context_),
              IsOk());
  ASSERT_THAT(fn_.Write(CreateWriteRequest("blob_2"), "data_2",
                        CreatePrivacyIdMetadata("privacy_id_2"), context_),
              IsOk());

  EXPECT_CALL(fn_, Reduce(SizeIs(2), _))
      .WillOnce(Return(CreateOutput("result")));
  EXPECT_CALL(context_, EmitReleasable(0, _, _, _, _))
      .WillOnce([](int, Session::KV kv, std::optional<absl::string_view>,
                   absl::string_view, std::string& release_token) {
        EXPECT_EQ(kv.data, "result");
        release_token = "release_token";
        return true;
      });
  absl::StatusOr<FinalizeResponse> response = CommitAndFinalize(fn_);
  ASSERT_THAT(response, IsOk());
  EXPECT_EQ(response->release_token(), "release_token");
}

TEST_F(GlobalReduceFnTest, WriteAcceptsDuplicateBlobIdsWithoutPrivacyIds) {
  // Inputs without privacy IDs aren't deduplicated by GlobalReduceFn.
  EXPECT_THAT(fn_.Write(CreateWriteRequest("blob_1"), "data_1", context_),
              IsOk());
  EXPECT_THAT(fn_.Write(CreateWriteRequest("blob_1"), "data_2", context_),
              IsOk());
}

TEST_F(GlobalReduceFnTest, WriteAcceptsDistinctPrivacyIdsWithSameBlobId) {
  // Inputs with privacy IDs are deduplicated by privacy ID only.
  EXPECT_THAT(fn_.Write(CreateWriteRequest("blob_1"), "data_1",
                        CreatePrivacyIdMetadata("privacy_id_1"), context_),
              IsOk());
  EXPECT_THAT(fn_.Write(CreateWriteRequest("blob_1"), "data_2",
                        CreatePrivacyIdMetadata("privacy_id_2"), context_),
              IsOk());
}

TEST_F(GlobalReduceFnTest, WriteRejectsDuplicatePrivacyId) {
  ASSERT_THAT(fn_.Write(CreateWriteRequest("blob_1"), "data_1",
                        CreatePrivacyIdMetadata("privacy_id_1"), context_),
              IsOk());
  // A different blob from the same privacy unit.
  EXPECT_THAT(
      fn_.Write(CreateWriteRequest("blob_2"), "data_2",
                CreatePrivacyIdMetadata("privacy_id_1"), context_),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Duplicate privacy id"),
                     // The privacy ID is sensitive and must not be leaked.
                     Not(HasSubstr("privacy_id_1")))));
}

TEST_F(GlobalReduceFnTest,
       WriteRejectsMissingPrivacyIdAfterInputWithPrivacyId) {
  ASSERT_THAT(fn_.Write(CreateWriteRequest("blob_1"), "data_1",
                        CreatePrivacyIdMetadata("privacy_id_1"), context_),
              IsOk());
  EXPECT_THAT(fn_.Write(CreateWriteRequest("blob_2"), "data_2", context_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Missing PrivacyIdMetadata")));
}

TEST_F(GlobalReduceFnTest, WriteRejectsPrivacyIdAfterInputWithoutPrivacyId) {
  ASSERT_THAT(fn_.Write(CreateWriteRequest("blob_1"), "data_1", context_),
              IsOk());
  EXPECT_THAT(fn_.Write(CreateWriteRequest("blob_2"), "data_2",
                        CreatePrivacyIdMetadata("privacy_id_1"), context_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unexpected PrivacyIdMetadata")));
}

TEST_F(GlobalReduceFnTest, WriteFindsPrivacyIdAmongOtherProtectedMetadata) {
  BlobHeader header;
  header.set_blob_id("protected_blob_id");
  ProtectedMetadata protected_metadata;
  protected_metadata.add_metadata()->PackFrom(header);
  *protected_metadata.add_metadata() =
      CreatePrivacyIdMetadata("privacy_id_1").metadata(0);
  ASSERT_THAT(fn_.Write(CreateWriteRequest("blob_1"), "data_1",
                        protected_metadata, context_),
              IsOk());

  EXPECT_THAT(fn_.Write(CreateWriteRequest("blob_2"), "data_2",
                        CreatePrivacyIdMetadata("privacy_id_1"), context_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Duplicate privacy id")));
}

TEST_F(GlobalReduceFnTest, RejectedInputIsNotPassedToReduce) {
  ASSERT_THAT(fn_.Write(CreateWriteRequest("blob_1"), "data_1",
                        CreatePrivacyIdMetadata("privacy_id_1"), context_),
              IsOk());
  ASSERT_THAT(fn_.Write(CreateWriteRequest("blob_2"), "data_2",
                        CreatePrivacyIdMetadata("privacy_id_1"), context_),
              Not(IsOk()));
  ASSERT_THAT(fn_.Write(CreateWriteRequest("blob_3"), "data_3",
                        CreatePrivacyIdMetadata("privacy_id_3"), context_),
              IsOk());

  std::vector<Element> inputs;
  EXPECT_CALL(fn_, Reduce(_, _))
      .WillOnce(DoAll(SaveArg<0>(&inputs), Return(CreateOutput())));
  EXPECT_CALL(context_, EmitReleasable(0, _, _, _, _)).WillOnce(Return(true));
  ASSERT_THAT(CommitAndFinalize(fn_), IsOk());
  ASSERT_THAT(inputs, SizeIs(2));
  EXPECT_EQ(inputs[0].kv.data, "data_1");
  EXPECT_EQ(inputs[1].kv.data, "data_3");
}

}  // namespace
}  // namespace confidential_federated_compute::fns
