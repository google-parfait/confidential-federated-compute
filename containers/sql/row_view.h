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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SQL_ROW_VIEW_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SQL_ROW_VIEW_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "fcp/base/monitoring.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.pb.h"
#include "google/protobuf/message.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace confidential_federated_compute::sql {

// Represents a direct mapping from a root message level to a leaf field.
// Each `FieldDescriptor` in the path describes navigation through a single
// nested message field, sequentially resolving values to mapped leaf fields.
// Safety: Pointers to FieldDescriptors are owned by static DescriptorPools
// and remain valid across runtime executions.
using FieldPath = std::vector<const google::protobuf::FieldDescriptor*>;

// An ordered flat array of field paths matching mapped table columns.
// The list maps 1:1 with flat columns computed once per table schema execution
// by Input::CreateFromMessages during initialization via GetFlattenedSchema.
//
// Safety: Pointers referencing this list are safe to pass because lists are
// owned by Input::MessageContents which is guaranteed to outlive all RowView
// instances generated during operations.
using FieldPathList = std::vector<FieldPath>;

// A non-owning view of a single row of data, abstracting the underlying
// storage mechanism (e.g., Tensors, Messages) via absl::variant.
class RowView {
 public:
  // Creates a RowView from a span of columns and a row index.
  // A RowView created this way will provide access to the elements of the row
  // by index, in the order of the `columns` span.
  static absl::StatusOr<RowView> CreateFromTensors(
      absl::Span<const tensorflow_federated::aggregation::Tensor> columns,
      uint32_t row_index);

  // Creates a RowView from a Message, a list of system columns, and a row
  // index.
  // The row index identifies the row within the system columns.
  // A RowView created this way will provide access to the elements of the
  // row by index, in the order of the message's field numbers followed by the
  // system columns in order of the `system_columns` span.
  static absl::StatusOr<RowView> CreateFromMessage(
      const google::protobuf::Message* message ABSL_ATTRIBUTE_LIFETIME_BOUND,
      absl::Span<const tensorflow_federated::aggregation::Tensor>
          system_columns,
      uint32_t row_index, const FieldPathList* field_paths);

  // Returns the data type of a column.
  tensorflow_federated::aggregation::DataType GetColumnType(
      int column_index) const {
    return absl::visit(
        [column_index](const auto& view) {
          return view.GetColumnType(column_index);
        },
        row_view_variant_);
  }

  // Returns the value of an element in the row.
  template <typename T>
  T GetValue(int column_index) const {
    return absl::visit(
        [column_index](const auto& view) {
          return view.template GetValue<T>(column_index);
        },
        row_view_variant_);
  }

  // Returns the number of columns in the row.
  size_t GetColumnCount() const {
    return absl::visit([](const auto& view) { return view.GetColumnCount(); },
                       row_view_variant_);
  }

 private:
  // Type trait to check if a type T conforms to the RowView interface.
  template <typename T, typename = void>
  struct has_row_view_interface : std::false_type {};

  template <typename T>
  struct has_row_view_interface<
      T, std::void_t<
             decltype(std::declval<const T&>().GetColumnType(0)),
             decltype(std::declval<const T&>().template GetValue<int32_t>(0)),
             decltype(std::declval<const T&>().GetColumnCount())>>
      : std::true_type {};

  // A RowView backed by Tensors.
  class TensorRowView {
   public:
    static absl::StatusOr<TensorRowView> Create(
        absl::Span<const tensorflow_federated::aggregation::Tensor> columns,
        uint32_t row_index);

    tensorflow_federated::aggregation::DataType GetColumnType(
        int column_index) const {
      return columns_[column_index].dtype();
    }

    template <typename T>
    T GetValue(int column_index) const {
      const auto& column = columns_[column_index];
      // This will CHECK-fail if T does not match the column's dtype.
      return column.AsSpan<T>().at(row_index_);
    }

    size_t GetColumnCount() const { return columns_.size(); }

   private:
    TensorRowView(
        absl::Span<const tensorflow_federated::aggregation::Tensor> columns,
        uint32_t row_index)
        : columns_(columns), row_index_(row_index) {};

    absl::Span<const tensorflow_federated::aggregation::Tensor> columns_;
    uint32_t row_index_;
  };

  static_assert(has_row_view_interface<TensorRowView>::value,
                "TensorRowView does not conform to the RowView interface.");

  // A RowView backed by a Message and a list of system columns.
  class MessageRowView {
   public:
    MessageRowView(const google::protobuf::Message* message
                       ABSL_ATTRIBUTE_LIFETIME_BOUND,
                   absl::Span<const tensorflow_federated::aggregation::Tensor>
                       system_columns,
                   uint32_t row_index, const FieldPathList* field_paths);

    tensorflow_federated::aggregation::DataType GetColumnType(
        int column_index) const;

    template <typename T>
    T GetValue(int column_index) const;

    size_t GetColumnCount() const;

   private:
    size_t GetSystemColumnIndex(int column_index) const;

    template <typename T>
    T GetMessageValue(const google::protobuf::Message& msg,
                      const google::protobuf::FieldDescriptor* field) const;

    tensorflow_federated::aggregation::DataType GetMessageColumnType(
        int column_index) const;

    const google::protobuf::Message* message_;
    absl::Span<const tensorflow_federated::aggregation::Tensor> system_columns_;
    // The index of the row within the system columns.
    uint32_t row_index_;
    // Flattened list of field paths for message columns. Owned by Input.
    const FieldPathList* field_paths_;
  };

  static_assert(has_row_view_interface<MessageRowView>::value,
                "MessageRowView does not conform to the RowView interface.");

  using RowViewVariant = absl::variant<TensorRowView, MessageRowView>;

  explicit RowView(RowViewVariant row_view_variant)
      : row_view_variant_(std::move(row_view_variant)) {}

  RowViewVariant row_view_variant_;
};

template <typename T>
T RowView::MessageRowView::GetMessageValue(
    const google::protobuf::Message& msg,
    const google::protobuf::FieldDescriptor* field) const {
  FCP_LOG(FATAL) << "Unsupported column type " << field->cpp_type_name();
}

template <>
inline int32_t RowView::MessageRowView::GetMessageValue<int32_t>(
    const google::protobuf::Message& msg,
    const google::protobuf::FieldDescriptor* field) const {
  const google::protobuf::Reflection* reflection = msg.GetReflection();
  if (field->cpp_type() == google::protobuf::FieldDescriptor::CPPTYPE_ENUM) {
    return reflection->GetEnumValue(msg, field);
  }
  FCP_CHECK(field->cpp_type() ==
            google::protobuf::FieldDescriptor::CPPTYPE_INT32)
      << "Field " << field->name() << " has type " << field->cpp_type_name()
      << " but expected int32";
  return reflection->GetInt32(msg, field);
}

template <>
inline int64_t RowView::MessageRowView::GetMessageValue<int64_t>(
    const google::protobuf::Message& msg,
    const google::protobuf::FieldDescriptor* field) const {
  FCP_CHECK(field->cpp_type() ==
            google::protobuf::FieldDescriptor::CPPTYPE_INT64)
      << "Field " << field->name() << " has type " << field->cpp_type_name()
      << " but expected int64";
  return msg.GetReflection()->GetInt64(msg, field);
}

template <>
inline float RowView::MessageRowView::GetMessageValue<float>(
    const google::protobuf::Message& msg,
    const google::protobuf::FieldDescriptor* field) const {
  FCP_CHECK(field->cpp_type() ==
            google::protobuf::FieldDescriptor::CPPTYPE_FLOAT)
      << "Field " << field->name() << " has type " << field->cpp_type_name()
      << " but expected float";
  return msg.GetReflection()->GetFloat(msg, field);
}

template <>
inline double RowView::MessageRowView::GetMessageValue<double>(
    const google::protobuf::Message& msg,
    const google::protobuf::FieldDescriptor* field) const {
  FCP_CHECK(field->cpp_type() ==
            google::protobuf::FieldDescriptor::CPPTYPE_DOUBLE)
      << "Field " << field->name() << " has type " << field->cpp_type_name()
      << " but expected double";
  return msg.GetReflection()->GetDouble(msg, field);
}

template <>
inline absl::string_view
RowView::MessageRowView::GetMessageValue<absl::string_view>(
    const google::protobuf::Message& msg,
    const google::protobuf::FieldDescriptor* field) const {
  FCP_CHECK(field->cpp_type() ==
            google::protobuf::FieldDescriptor::CPPTYPE_STRING)
      << "Field " << field->name() << " has type " << field->cpp_type_name()
      << " but expected string";
  FCP_CHECK(field->options().ctype() == google::protobuf::FieldOptions::STRING)
      << "Field " << field->name() << " has unsupported ctype "
      << field->options().ctype();
  // GetStringReference copies the field into `unused` if the field is
  // not stored as a string (e.g. it's stored as absl::Cord). Since we check
  // that ctype == STRING, `unused` won't be used and GetStringReference
  // will return a reference to the underlying field.
  std::string unused;
  return msg.GetReflection()->GetStringReference(msg, field, &unused);
}

template <typename T>
T RowView::MessageRowView::GetValue(int column_index) const {
  if (column_index < field_paths_->size()) {
    // Navigate the pre-computed path of field descriptors to retrieve the
    // value from the correct nested message instance.
    const auto& path = (*field_paths_)[column_index];
    const google::protobuf::Message* current_msg = message_;
    for (size_t i = 0; i < path.size() - 1; ++i) {
      current_msg =
          &current_msg->GetReflection()->GetMessage(*current_msg, path[i]);
    }
    return GetMessageValue<T>(*current_msg, path.back());
  }
  // This will CHECK-fail if T does not match the column's dtype.
  return system_columns_[GetSystemColumnIndex(column_index)].AsSpan<T>().at(
      row_index_);
}

}  // namespace confidential_federated_compute::sql

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_SQL_ROW_VIEW_H_
