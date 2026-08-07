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
#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_ROW_VIEW_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_ROW_VIEW_H_

// Overview
// --------
// The table abstraction stack is organized into four layers across row_view.h,
// input.h, and row_set.h:
//
// Column Schema
//   MessageColumnSchema (std::vector<ColumnDescriptor>) describes every column
//   in order and encodes how to read it. Proto columns store a FieldPath and
//   read via reflection; system columns have an empty FieldPath and read from a
//   Tensor using system_column_index. column_type is set once at schema-build
//   time and never re-derived from FieldDescriptor::cpp_type() at read time,
//   allowing enum fields to produce both a DT_INT32 integer column and a
//   DT_STRING name column.
//
// Single-Row Cursor
//   RowView is a lightweight, non-owning view over a single row. It holds no
//   data of its own, only references into the owning storage plus a row index.
//   - TensorRowView reads directly from flat column-major Tensor arrays.
//   - MessageRowView uses MessageColumnSchema to route reads to either proto
//     reflection or system Tensors.
//
// See input.h for Table Storage (Input) and row_set.h for Cross-Table View
// (RowSet).

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/descriptor.pb.h"
#include "google/protobuf/message.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

namespace confidential_federated_compute {

// Represents the sequence of FieldDescriptors that navigates from the root
// message to a leaf field.  Each entry describes a single hop through a
// nested message field.
//
// Safety: Pointers to FieldDescriptors are owned by static DescriptorPools
// and remain valid across runtime executions.
using FieldPath = std::vector<const google::protobuf::FieldDescriptor*>;

// Unified descriptor for one logical column in a message-backed Input.
//
// Covers both kinds of columns:
//   - Proto columns:        proto_path is non-empty; value is read via proto
//                           reflection by navigating the field path.
//   - System tensor columns: proto_path is empty; value is read from the
//                           system_columns span at system_tensor_index.
//
// name and column_type are always present and are the single source of truth
// for column identity and type — no secondary name arrays are kept in sync.
//
// Two proto columns may share the same proto_path but carry different
// column_types and names (e.g. an enum's integer column and its "_as_str"
// string column). column_type is never re-inferred from cpp_type() at read
// time; it is set once at schema-build time.
struct ColumnDescriptor {
  // The flat column name, e.g. "event_type" or "nested__sub_col1".
  std::string name;

  // The output DataType for this column. Set once at schema-build time
  // (GetFlattenedSchema) and read directly by GetColumnType().
  tensorflow_federated::aggregation::DataType column_type;

  // Path of FieldDescriptors from the root message to the leaf field.
  // Non-empty for proto columns; empty for system tensor columns.
  FieldPath proto_path;

  // For system tensor columns (proto_path.empty()), the index into the
  // system_columns span passed to CreateFromMessage (and held by
  // MessageContents::system_columns_).
  size_t system_tensor_index = 0;
};

// Ordered schema describing all logical columns for a message-backed Input.
// Computed once by Input::CreateFromMessages; covers both proto columns
// (appended by GetFlattenedSchema) and system tensor columns (appended
// immediately after).  Shared by pointer with every RowView this Input
// produces.
using MessageColumnSchema = std::vector<ColumnDescriptor>;

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

  // Creates a RowView from a Message, a span of system tensors, and a row
  // index.
  //
  // `schema` describes every column — both proto columns (non-empty
  // proto_path) and system tensor columns (empty proto_path, with
  // system_tensor_index pointing into `system_columns`).
  // `schema` must outlive this RowView.
  static absl::StatusOr<RowView> CreateFromMessage(
      const google::protobuf::Message* message ABSL_ATTRIBUTE_LIFETIME_BOUND,
      absl::Span<const tensorflow_federated::aggregation::Tensor>
          system_columns,
      uint32_t row_index,
      const MessageColumnSchema* schema ABSL_ATTRIBUTE_LIFETIME_BOUND);

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

  // A RowView backed by a Message and a unified MessageColumnSchema.
  //
  // The schema covers both proto columns (non-empty proto_path) and system
  // columns (empty proto_path, backed by a Tensor at system_column_index).
  class MessageRowView {
   public:
    MessageRowView(
        const google::protobuf::Message* message ABSL_ATTRIBUTE_LIFETIME_BOUND,
        absl::Span<const tensorflow_federated::aggregation::Tensor>
            system_columns,
        uint32_t row_index,
        const MessageColumnSchema* schema ABSL_ATTRIBUTE_LIFETIME_BOUND);

    // Returns ColumnDescriptor::column_type, set at schema-build time.
    // This is what ensures enum-as-str columns return DT_STRING.
    tensorflow_federated::aggregation::DataType GetColumnType(
        int column_index) const;

    template <typename T>
    T GetValue(int column_index) const;

    size_t GetColumnCount() const;

   private:
    template <typename T>
    T GetMessageValue(const google::protobuf::Message& msg,
                      const google::protobuf::FieldDescriptor* field) const;

    const google::protobuf::Message* message_;
    absl::Span<const tensorflow_federated::aggregation::Tensor> system_columns_;
    // The index of the row within the system columns.
    uint32_t row_index_;
    // Owned by Input::MessageContents, outlives all RowViews from that Input.
    const MessageColumnSchema* schema_;
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
  LOG(FATAL) << "Unsupported column type " << field->cpp_type_name();
}

template <>
inline int32_t RowView::MessageRowView::GetMessageValue<int32_t>(
    const google::protobuf::Message& msg,
    const google::protobuf::FieldDescriptor* field) const {
  const google::protobuf::Reflection* reflection = msg.GetReflection();
  if (field->cpp_type() == google::protobuf::FieldDescriptor::CPPTYPE_ENUM) {
    return reflection->GetEnumValue(msg, field);
  }
  if (field->cpp_type() == google::protobuf::FieldDescriptor::CPPTYPE_BOOL) {
    return static_cast<int32_t>(reflection->GetBool(msg, field));
  }
  if (field->cpp_type() == google::protobuf::FieldDescriptor::CPPTYPE_UINT32) {
    uint32_t val = reflection->GetUInt32(msg, field);
    CHECK_LE(val, static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))
        << "uint32 field " << field->name() << " value overflows int32";
    return static_cast<int32_t>(val);
  }
  CHECK(field->cpp_type() == google::protobuf::FieldDescriptor::CPPTYPE_INT32)
      << "Field " << field->name() << " has type " << field->cpp_type_name()
      << " but expected int32";
  return reflection->GetInt32(msg, field);
}

template <>
inline int64_t RowView::MessageRowView::GetMessageValue<int64_t>(
    const google::protobuf::Message& msg,
    const google::protobuf::FieldDescriptor* field) const {
  const google::protobuf::Reflection* reflection = msg.GetReflection();
  if (field->cpp_type() == google::protobuf::FieldDescriptor::CPPTYPE_UINT64) {
    uint64_t val = reflection->GetUInt64(msg, field);
    CHECK_LE(val, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        << "uint64 field " << field->name() << " value overflows int64";
    return static_cast<int64_t>(val);
  }
  CHECK(field->cpp_type() == google::protobuf::FieldDescriptor::CPPTYPE_INT64)
      << "Field " << field->name() << " has type " << field->cpp_type_name()
      << " but expected int64";
  return msg.GetReflection()->GetInt64(msg, field);
}

template <>
inline float RowView::MessageRowView::GetMessageValue<float>(
    const google::protobuf::Message& msg,
    const google::protobuf::FieldDescriptor* field) const {
  CHECK(field->cpp_type() == google::protobuf::FieldDescriptor::CPPTYPE_FLOAT)
      << "Field " << field->name() << " has type " << field->cpp_type_name()
      << " but expected float";
  return msg.GetReflection()->GetFloat(msg, field);
}

template <>
inline double RowView::MessageRowView::GetMessageValue<double>(
    const google::protobuf::Message& msg,
    const google::protobuf::FieldDescriptor* field) const {
  CHECK(field->cpp_type() == google::protobuf::FieldDescriptor::CPPTYPE_DOUBLE)
      << "Field " << field->name() << " has type " << field->cpp_type_name()
      << " but expected double";
  return msg.GetReflection()->GetDouble(msg, field);
}

template <>
inline absl::string_view
RowView::MessageRowView::GetMessageValue<absl::string_view>(
    const google::protobuf::Message& msg,
    const google::protobuf::FieldDescriptor* field) const {
  const google::protobuf::Reflection* reflection = msg.GetReflection();
  // Enum fields read as string_view return the value's name (e.g. "SHOWN").
  // This is used when column_type == DT_STRING for an enum column spec.
  if (field->cpp_type() == google::protobuf::FieldDescriptor::CPPTYPE_ENUM) {
    const google::protobuf::EnumValueDescriptor* enum_val =
        reflection->GetEnum(msg, field);
    if (enum_val != nullptr) {
      return enum_val->name();
    }
    // enum_val is null for unknown enum values (e.g. values not present in the
    // descriptor, which can happen with proto3 open enums or mismatched
    // schemas). Return empty string rather than crashing.
    LOG(WARNING) << "Unknown enum value for field " << field->name()
                 << "; returning empty string";
    return "";
  }
  CHECK(field->cpp_type() == google::protobuf::FieldDescriptor::CPPTYPE_STRING)
      << "Field " << field->name() << " has type " << field->cpp_type_name()
      << " but expected string";
  CHECK(field->options().ctype() == google::protobuf::FieldOptions::STRING)
      << "Field " << field->name() << " has unsupported ctype "
      << field->options().ctype();
  // GetStringReference copies the field into `unused` if the field is
  // not stored as a string (e.g. it's stored as absl::Cord). Since we check
  // that ctype == STRING, `unused` won't be used and GetStringReference
  // will return a reference to the underlying field.
  std::string unused;
  return reflection->GetStringReference(msg, field, &unused);
}

template <typename T>
T RowView::MessageRowView::GetValue(int column_index) const {
  const ColumnDescriptor& desc = (*schema_)[column_index];
  if (!desc.proto_path.empty()) {
    // Navigate the pre-computed path of field descriptors to retrieve the
    // value from the correct nested message instance.
    const google::protobuf::Message* current_msg = message_;
    for (size_t i = 0; i < desc.proto_path.size() - 1; ++i) {
      current_msg = &current_msg->GetReflection()->GetMessage(
          *current_msg, desc.proto_path[i]);
    }
    return GetMessageValue<T>(*current_msg, desc.proto_path.back());
  }
  // System tensor column: read from the pre-indexed tensor.
  // This will CHECK-fail if T does not match the column's dtype.
  return system_columns_[desc.system_tensor_index].AsSpan<T>().at(row_index_);
}

}  // namespace confidential_federated_compute

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_CONTAINERS_COMMON_ROW_VIEW_H_
