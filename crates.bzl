# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rust crates required by this workspace."""

load("@rules_rust//crate_universe:defs.bzl", "crate")

# Crates used by both std and no_std builds.
_COMMON_PACKAGES = {
    "aes-gcm-siv": crate.spec(
        version = "0.11.1",
    ),
    "bitmask": crate.spec(
        default_features = False,
        version = "0.5.0",
    ),
    "byteorder": crate.spec(
        default_features = False,
        version = "1.4.3",
    ),
    "core2": crate.spec(
        default_features = False,
        version = "0.4.0",
    ),
    "libflate": crate.spec(
        default_features = False,
        version = "2.0.0",
    ),
    "rangemap": crate.spec(
        default_features = False,
        version = "1.5.1",
    ),
}

# Crates used for std builds.
CFC_PACKAGES = _COMMON_PACKAGES | {
    "assert_cmd": crate.spec(
        version = "2.0.14",
    ),
    "insta": crate.spec(
        version = "1.38.0",
    ),
}

# Crates used for no_std builds.
CFC_NO_STD_PACKAGES = _COMMON_PACKAGES
