/*
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CONFIDENTIAL_FEDERATED_COMPUTE_TESTING_MATCHERS_H_
#define CONFIDENTIAL_FEDERATED_COMPUTE_TESTING_MATCHERS_H_

#include <iostream>
#include <string>
#include <type_traits>

#include "google/protobuf/util/message_differencer.h"
#include "testing/parse_text_proto.h"

namespace confidential_federated_compute {

// Proto matchers forked from FCP.

using RepeatedFieldComparison =
    google::protobuf::util::MessageDifferencer::RepeatedFieldComparison;

template <typename T>
class ProtoMatcherImpl : public ::testing::MatcherInterface<T> {
 public:
  explicit ProtoMatcherImpl(const google::protobuf::Message& arg,
                            RepeatedFieldComparison repeated_field_comparison)
      : arg_(CloneMessage(arg)),
        repeated_field_comparison_(repeated_field_comparison) {}

  explicit ProtoMatcherImpl(const std::string& arg,
                            RepeatedFieldComparison repeated_field_comparison)
      : arg_(ParseMessage(arg)),
        repeated_field_comparison_(repeated_field_comparison) {}

  void DescribeTo(::std::ostream* os) const override {
    *os << "is " << arg_->DebugString();
  }
  void DescribeNegationTo(::std::ostream* os) const override {
    *os << "is not " << arg_->DebugString();
  }
  bool MatchAndExplain(
      T x, ::testing::MatchResultListener* listener) const override {
    if (x.GetDescriptor()->full_name() != arg_->GetDescriptor()->full_name()) {
      *listener << "Argument proto is of type "
                << arg_->GetDescriptor()->full_name()
                << " but expected proto of type "
                << x.GetDescriptor()->full_name();
      return false;
    }

    google::protobuf::util::MessageDifferencer differencer;
    differencer.set_repeated_field_comparison(repeated_field_comparison_);
    std::string reported_differences;
    differencer.ReportDifferencesToString(&reported_differences);
    if (!differencer.Compare(*arg_, x)) {
      *listener << reported_differences;
      return false;
    }
    return true;
  }

 private:
  static std::unique_ptr<google::protobuf::Message> CloneMessage(
      const google::protobuf::Message& message) {
    std::unique_ptr<google::protobuf::Message> copy_of_message =
        absl::WrapUnique(message.New());
    copy_of_message->CopyFrom(message);
    return copy_of_message;
  }

  static std::unique_ptr<google::protobuf::Message> ParseMessage(
      const std::string& proto_text) {
    using V = std::remove_cv_t<std::remove_reference_t<T>>;
    std::unique_ptr<V> message = std::make_unique<V>();
    *message = PARSE_TEXT_PROTO(proto_text);
    return message;
  }

  std::unique_ptr<google::protobuf::Message> arg_;
  RepeatedFieldComparison repeated_field_comparison_;
};

template <typename T>
class ProtoMatcher {
 public:
  explicit ProtoMatcher(const T& arg,
                        RepeatedFieldComparison repeated_field_comparison =
                            RepeatedFieldComparison::AS_LIST)
      : arg_(arg), repeated_field_comparison_(repeated_field_comparison) {}

  template <typename U>
  operator testing::Matcher<U>() const {  // NOLINT
    using V = std::remove_cv_t<std::remove_reference_t<U>>;
    static_assert(std::is_base_of<google::protobuf::Message, V>::value &&
                  !std::is_same<google::protobuf::Message, V>::value);
    return ::testing::MakeMatcher(
        new ProtoMatcherImpl<U>(arg_, repeated_field_comparison_));
  }

 private:
  T arg_;
  RepeatedFieldComparison repeated_field_comparison_;
};

// Proto matcher that takes another proto message reference as an argument.
template <class T, typename std::enable_if<
                       std::is_base_of<google::protobuf::Message, T>::value &&
                           !std::is_same<google::protobuf::Message, T>::value,
                       int>::type = 0>
inline ProtoMatcher<T> EqualsProto(const T& arg) {
  return ProtoMatcher<T>(arg);
}

// Proto matcher that takes another proto message reference as an argument and
// ignores repeated field order.
template <class T, typename std::enable_if<
                       std::is_base_of<google::protobuf::Message, T>::value &&
                           !std::is_same<google::protobuf::Message, T>::value,
                       int>::type = 0>
inline ProtoMatcher<T> EqualsProtoIgnoringRepeatedFieldOrder(const T& arg) {
  return ProtoMatcher<T>(arg, RepeatedFieldComparison::AS_SET);
}

// Proto matcher that takes a text proto as an argument.
inline ProtoMatcher<std::string> EqualsProto(const std::string& arg) {
  return ProtoMatcher<std::string>(arg);
}

// Proto matcher that takes a text proto as an argument and ignores repeated
// field order.
inline ProtoMatcher<std::string> EqualsProtoIgnoringRepeatedFieldOrder(
    const std::string& arg) {
  return ProtoMatcher<std::string>(arg, RepeatedFieldComparison::AS_SET);
}

}  // namespace confidential_federated_compute

#endif  // CONFIDENTIAL_FEDERATED_COMPUTE_TESTING_MATCHERS_H_
