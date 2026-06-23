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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_INPUT_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_INPUT_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "containers/common/row_view.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.pb.h"
#include "google/protobuf/dynamic_message.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_parser.h"

namespace confidential_federated_compute {

// Interface for creating new protobuf messages.
class MessageFactory {
 public:
  virtual ~MessageFactory() = default;

  // Creates a new message instance.
  virtual std::unique_ptr<google::protobuf::Message> NewMessage() const = 0;
};

// Implementation of MessageFactory that creates messages from a
// FileDescriptorSet and message name.
class FileDescriptorSetMessageFactory : public MessageFactory {
 public:
  static absl::StatusOr<std::unique_ptr<MessageFactory>> Create(
      const google::protobuf::FileDescriptorSet& file_descriptor_set,
      absl::string_view message_name);

  std::unique_ptr<google::protobuf::Message> NewMessage() const override {
    return std::unique_ptr<google::protobuf::Message>(prototype_->New());
  }

 private:
  explicit FileDescriptorSetMessageFactory(
      std::unique_ptr<google::protobuf::DescriptorPool> descriptor_pool,
      std::unique_ptr<google::protobuf::DynamicMessageFactory>
          dynamic_message_factory,
      const google::protobuf::Message* prototype)
      : descriptor_pool_(std::move(descriptor_pool)),
        dynamic_message_factory_(std::move(dynamic_message_factory)),
        prototype_(prototype) {}
  // Holds the descriptor of the logged message. Must outlive the factory and
  // prototype.
  std::unique_ptr<google::protobuf::DescriptorPool> descriptor_pool_;
  // Factory for creating instances of the logged Message, whose type we don't
  // know at compile time. Must outlive the prototype.
  std::unique_ptr<google::protobuf::DynamicMessageFactory>
      dynamic_message_factory_;
  // Template for creating instances of the logged Message. It is only used by
  // calling the New() method to get a new, mutable Message*.
  const google::protobuf::Message* prototype_;
};

// Represents the contents of a single table, which may be backed by
// different underlying storage types (e.g., Tensors, Messages). This class uses
// absl::variant to abstract the specific storage mechanism.
//
// An Input can carry an opaque metadata string that is propagated through the
// abstraction layer without being interpreted by Input itself. Callers may use
// this to attach context (e.g., a key ID) that is needed downstream.
class Input {
 public:
  // Creates an Input from Tensors.
  //
  // The optional `privacy_id` Tensor must be a scalar tensor of type STRING.
  static absl::StatusOr<Input> CreateFromTensors(
      std::vector<tensorflow_federated::aggregation::Tensor> contents,
      std::string metadata = "",
      std::optional<tensorflow_federated::aggregation::Tensor> privacy_id =
          std::nullopt);

  // Creates an Input from a list of Messages and a list of system column
  // Tensors.
  //
  // Each row is composed of a single Message at index i in `messages` and the
  // element held at index i in each system column Tensor. Each field in the
  // Message corresponds to a column in the input table.
  //
  // Each system column Tensor must be a 1-dimensional Tensor with the same
  // number of elements (rows) as the number of messages.
  //
  // The optional `privacy_id` Tensor must be a scalar tensor of type STRING.
  static absl::StatusOr<Input> CreateFromMessages(
      std::vector<std::unique_ptr<google::protobuf::Message>> messages,
      std::vector<tensorflow_federated::aggregation::Tensor> system_columns,
      std::string metadata = "",
      std::optional<tensorflow_federated::aggregation::Tensor> privacy_id =
          std::nullopt);

  Input(Input&&) = default;
  Input& operator=(Input&&) = default;

  Input(const Input&) = delete;
  Input& operator=(const Input&) = delete;

  absl::Status AddColumn(
      tensorflow_federated::aggregation::Tensor&& new_column);

  absl::Span<const std::string> GetColumnNames() const;

  // Returns a view of the row at the specified index.
  //
  // The returned RowView will provide access to the values of the columns in
  // the row, in the order of the column names returned by GetColumnNames().
  absl::StatusOr<RowView> GetRow(uint32_t row_index) const;

  size_t GetRowCount() const;

  // Returns the opaque metadata string associated with this Input.
  const std::string& GetMetadata() const { return metadata_; }

  const std::optional<std::string>& GetPrivacyId() const { return privacy_id_; }

  absl::StatusOr<std::vector<tensorflow_federated::aggregation::Tensor>>
  MoveToTensors() &&;

 private:
  // Type trait to check if a type T conforms to the input contents interface.
  template <typename T, typename = void>
  struct has_input_contents_interface : std::false_type {};

  template <typename T>
  struct has_input_contents_interface<
      T, std::void_t<decltype(std::declval<const T&>().GetRowCount()),
                     decltype(std::declval<const T&>().GetRow(0)),
                     decltype(std::declval<T&&>().MoveToTensors({})),
                     decltype(std::declval<T&&>().AddColumn(
                         tensorflow_federated::aggregation::Tensor()))>>
      : std::true_type {};

  // Input contents backed by Tensors.
  class TensorContents {
   public:
    TensorContents(
        std::vector<tensorflow_federated::aggregation::Tensor> contents)
        : contents_(std::move(contents)) {}

    void AddColumn(tensorflow_federated::aggregation::Tensor&& column) {
      contents_.push_back(std::move(column));
    }

    absl::StatusOr<std::vector<tensorflow_federated::aggregation::Tensor>>
    MoveToTensors(absl::Span<const std::string> column_names) && {
      return std::move(contents_);
    }

    absl::StatusOr<RowView> GetRow(uint32_t row_index) const {
      return RowView::CreateFromTensors(contents_, row_index);
    }

    size_t GetRowCount() const;

   private:
    std::vector<tensorflow_federated::aggregation::Tensor> contents_;
  };

  static_assert(has_input_contents_interface<TensorContents>::value,
                "TensorContents does not conform to the input interface.");

  // Input contents backed by Message rows and Tensor system columns.
  class MessageContents {
   public:
    MessageContents(
        std::vector<std::unique_ptr<google::protobuf::Message>> messages,
        std::vector<tensorflow_federated::aggregation::Tensor> system_columns,
        FieldPathList field_paths)
        : messages_(std::move(messages)),
          system_columns_(std::move(system_columns)),
          field_paths_(std::move(field_paths)) {}

    void AddColumn(tensorflow_federated::aggregation::Tensor&& column) {
      system_columns_.push_back(std::move(column));
    }

    // Unfortunately, this method copies the underlying Message data, since the
    // reflection API doesn't support moving the data out of a Message.
    absl::StatusOr<std::vector<tensorflow_federated::aggregation::Tensor>>
    MoveToTensors(absl::Span<const std::string> column_names) &&;

    absl::StatusOr<RowView> GetRow(uint32_t row_index) const;

    size_t GetRowCount() const { return messages_.size(); };

   private:
    std::vector<std::unique_ptr<google::protobuf::Message>> messages_;
    std::vector<tensorflow_federated::aggregation::Tensor> system_columns_;
    FieldPathList field_paths_;
  };

  static_assert(has_input_contents_interface<MessageContents>::value,
                "MessageContents does not conform to the input interface.");

  using ContentsVariant = absl::variant<TensorContents, MessageContents>;

  Input(ContentsVariant contents, std::string metadata,
        std::vector<std::string> column_names,
        std::optional<std::string> privacy_id);

  ContentsVariant contents_;
  std::string metadata_;
  std::vector<std::string> column_names_;
  std::optional<std::string> privacy_id_;
};

absl::StatusOr<Input> CreateFromMessageCheckpoint(
    tensorflow_federated::aggregation::CheckpointParser* checkpoint,
    MessageFactory& message_factory, absl::string_view on_device_query_name);

}  // namespace confidential_federated_compute

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_INPUT_H_
