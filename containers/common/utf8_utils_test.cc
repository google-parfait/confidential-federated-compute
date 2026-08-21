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

#include "containers/common/utf8_utils.h"

#include <string>

#include "gtest/gtest.h"

namespace confidential_federated_compute {
namespace {

TEST(Utf8UtilsTest, IsValidUtf8ValidStrings) {
  EXPECT_TRUE(IsValidUtf8(""));
  EXPECT_TRUE(IsValidUtf8("hello world"));
  EXPECT_TRUE(IsValidUtf8("café"));
  EXPECT_TRUE(IsValidUtf8("こんにちは"));
  EXPECT_TRUE(IsValidUtf8("🌍🌎🌏"));
}

TEST(Utf8UtilsTest, IsValidUtf8InvalidStrings) {
  // Truncated multi-byte character
  EXPECT_FALSE(IsValidUtf8("\xC3"));
  EXPECT_FALSE(IsValidUtf8("\xE4\xB8"));
  EXPECT_FALSE(IsValidUtf8("\xF0\x9F\x8C"));

  // Stray continuation byte
  EXPECT_FALSE(IsValidUtf8("\x80"));
  EXPECT_FALSE(IsValidUtf8("\xBF"));

  // Invalid leading byte
  EXPECT_FALSE(IsValidUtf8("\xFF"));
  EXPECT_FALSE(IsValidUtf8("\xC0\xAF"));  // Overlong ASCII
}

TEST(Utf8UtilsTest, TruncateUtf8NoOpWhenUnderLimit) {
  EXPECT_EQ(TruncateUtf8("", 10), "");
  EXPECT_EQ(TruncateUtf8("hello", 10), "hello");
  EXPECT_EQ(TruncateUtf8("hello", 5), "hello");
  EXPECT_EQ(TruncateUtf8("café", 5), "café");  // café is 5 bytes
}

TEST(Utf8UtilsTest, TruncateUtf8Ascii) {
  EXPECT_EQ(TruncateUtf8("hello world", 5), "hello");
  EXPECT_EQ(TruncateUtf8("hello world", 0), "");
}

TEST(Utf8UtilsTest, TruncateUtf8TwoByteCharacter) {
  // "café": c (1), a (1), f (1), é (\xC3\xA9 - 2 bytes) -> 5 bytes total
  std::string s = "café";
  ASSERT_EQ(s.size(), 5);

  // Truncate at 4 bytes (cuts middle of é) -> should truncate to "caf" (3
  // bytes)
  EXPECT_EQ(TruncateUtf8(s, 4), "caf");
  EXPECT_TRUE(IsValidUtf8(TruncateUtf8(s, 4)));

  // Truncate at 3 bytes -> "caf"
  EXPECT_EQ(TruncateUtf8(s, 3), "caf");
  EXPECT_TRUE(IsValidUtf8(TruncateUtf8(s, 3)));

  // Truncate at 5 bytes -> "café"
  EXPECT_EQ(TruncateUtf8(s, 5), "café");
  EXPECT_TRUE(IsValidUtf8(TruncateUtf8(s, 5)));
}

TEST(Utf8UtilsTest, TruncateUtf8ThreeByteCharacter) {
  // "世": \xE4\xB8\x96 (3 bytes)
  // "hi世": 2 + 3 = 5 bytes
  std::string s = "hi世";
  ASSERT_EQ(s.size(), 5);

  // Truncate at 4 bytes (cuts middle of 世) -> "hi" (2 bytes)
  EXPECT_EQ(TruncateUtf8(s, 4), "hi");
  EXPECT_TRUE(IsValidUtf8(TruncateUtf8(s, 4)));

  // Truncate at 3 bytes (cuts middle of 世) -> "hi" (2 bytes)
  EXPECT_EQ(TruncateUtf8(s, 3), "hi");
  EXPECT_TRUE(IsValidUtf8(TruncateUtf8(s, 3)));

  // Truncate at 2 bytes -> "hi"
  EXPECT_EQ(TruncateUtf8(s, 2), "hi");
  EXPECT_TRUE(IsValidUtf8(TruncateUtf8(s, 2)));

  // Truncate at 5 bytes -> "hi世"
  EXPECT_EQ(TruncateUtf8(s, 5), "hi世");
  EXPECT_TRUE(IsValidUtf8(TruncateUtf8(s, 5)));
}

TEST(Utf8UtilsTest, TruncateUtf8FourByteCharacter) {
  // "🌍": \xF0\x9F\x8C\x8D (4 bytes)
  // "A🌍B": 1 + 4 + 1 = 6 bytes
  std::string s = "A🌍B";
  ASSERT_EQ(s.size(), 6);

  // Truncate at 5 bytes (after 🌍) -> "A🌍" (5 bytes)
  EXPECT_EQ(TruncateUtf8(s, 5), "A🌍");
  EXPECT_TRUE(IsValidUtf8(TruncateUtf8(s, 5)));

  // Truncate at 4 bytes (cuts inside 🌍) -> "A" (1 byte)
  EXPECT_EQ(TruncateUtf8(s, 4), "A");
  EXPECT_TRUE(IsValidUtf8(TruncateUtf8(s, 4)));

  // Truncate at 3 bytes (cuts inside 🌍) -> "A" (1 byte)
  EXPECT_EQ(TruncateUtf8(s, 3), "A");
  EXPECT_TRUE(IsValidUtf8(TruncateUtf8(s, 3)));

  // Truncate at 2 bytes (cuts inside 🌍) -> "A" (1 byte)
  EXPECT_EQ(TruncateUtf8(s, 2), "A");
  EXPECT_TRUE(IsValidUtf8(TruncateUtf8(s, 2)));

  // Truncate at 1 byte -> "A" (1 byte)
  EXPECT_EQ(TruncateUtf8(s, 1), "A");
  EXPECT_TRUE(IsValidUtf8(TruncateUtf8(s, 1)));

  // Truncate at 0 bytes -> ""
  EXPECT_EQ(TruncateUtf8(s, 0), "");
  EXPECT_TRUE(IsValidUtf8(TruncateUtf8(s, 0)));
}

}  // namespace
}  // namespace confidential_federated_compute
