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

#include "containers/big_endian.h"

#include "absl/numeric/int128.h"
#include "gtest/gtest.h"

namespace confidential_federated_compute {
namespace {

TEST(BigEndianTest, Load64) {
  EXPECT_EQ(LoadBigEndian<uint64_t>("\0\0\0\0\0\0\0\x1", 8), 1);
  EXPECT_EQ(LoadBigEndian<uint64_t>("\0\0\0\0\0\x2\0\x12", 8), 131090);
  EXPECT_EQ(LoadBigEndian<uint64_t>("", 0), 0);
  EXPECT_EQ(LoadBigEndian<uint64_t>("\x1", 1), 0x100000000000000);
}

TEST(BigEndianTest, Store64) {
  EXPECT_EQ(StoreBigEndian<uint64_t>(1), std::string("\0\0\0\0\0\0\0\x01", 8));
  EXPECT_EQ(StoreBigEndian<uint64_t>(131090),
            std::string("\0\0\0\0\0\x2\0\x12", 8));
  EXPECT_EQ(StoreBigEndian<uint64_t>(0), std::string("\0\0\0\0\0\0\0\0", 8));
  EXPECT_EQ(StoreBigEndian<uint64_t>(0x100000000000000),
            std::string("\x1\0\0\0\0\0\0\0", 8));
}

TEST(BigEndianTest, Load128) {
  EXPECT_EQ(LoadBigEndian<absl::uint128>("", 0), 0);
  EXPECT_EQ(
      LoadBigEndian<absl::uint128>("\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x1", 16), 1);
  EXPECT_EQ(
      LoadBigEndian<absl::uint128>("\0\0\0\0\0\0\0\x1\0\0\0\0\0\0\0\0", 16),
      absl::MakeUint128(1, 0));
  EXPECT_EQ(LoadBigEndian<absl::uint128>("\x1", 1),
            absl::MakeUint128(0x100000000000000, 0));
}

TEST(BigEndianTest, Store128) {
  EXPECT_EQ(StoreBigEndian<absl::uint128>(0),
            std::string("\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0", 16));
  EXPECT_EQ(StoreBigEndian<absl::uint128>(1),
            std::string("\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\x1", 16));
  EXPECT_EQ(StoreBigEndian<absl::uint128>(absl::MakeUint128(1, 0)),
            std::string("\0\0\0\0\0\0\0\x1\0\0\0\0\0\0\0\0", 16));
  EXPECT_EQ(
      StoreBigEndian<absl::uint128>(absl::MakeUint128(0x100000000000000, 0)),
      std::string("\x1\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0", 16));
}

}  // namespace
}  // namespace confidential_federated_compute
