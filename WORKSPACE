# Copyright 2023 Google LLC.
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

################################################################################
# Directly managed dependencies
################################################################################

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//:stub_repo.bzl", "stub_repo")
load("//:stub_tf_proto_library.bzl", "stub_tf_proto_library")

# go/keep-sorted start block=yes newline_separated=yes ignore_prefixes=git_repository,http_archive,stub_repo,stub_tf_proto_library
http_archive(
    name = "bazel_toolchains",
    sha256 = "294cdd859e57fcaf101d4301978c408c88683fbc46fbc1a3829da92afbea55fb",
    strip_prefix = "bazel-toolchains-8c717f8258cd5f6c7a45b97d974292755852b658",
    url = "https://github.com/bazelbuild/bazel-toolchains/archive/8c717f8258cd5f6c7a45b97d974292755852b658.tar.gz",
)

http_archive(
    name = "boringssl",
    integrity = "sha256-YfmrrozMej0Kon25c2SrYYc4NEHI0R3+MVBlzWEsXo8=",
    patches = ["//third_party/boringssl:rust.patch"],
    strip_prefix = "boringssl-34492c89a8e381e0e856a686cc71b1eb5bd728db",
    urls = ["https://github.com/google/boringssl/archive/34492c89a8e381e0e856a686cc71b1eb5bd728db.tar.gz"],
)

http_archive(
    name = "com_github_grpc_grpc",
    integrity = "sha256-rhSg3iIkhf1uO69SAox0rL2a2NaFyBNYBAHTgyz66fE=",
    strip_prefix = "grpc-1.72.2",
    urls = ["https://github.com/grpc/grpc/archive/refs/tags/v1.72.2.tar.gz"],
)

# Avoid Kotlin dependencies.
stub_repo(
    name = "com_github_grpc_grpc_kotlin",
    rules = {":kt_jvm_grpc.bzl": ["kt_jvm_grpc_library"]},
)

http_archive(
    name = "com_google_absl",
    integrity = "sha256-9Q5awxGoE4Laf6dblzEOS5AGR0+VYKxG9UqZZ/B9SuM=",
    # Prevent googletest from being pulled in twice, causing ODR and strict
    # layering violations.
    repo_mapping = {"@com_google_googletest": "@googletest"},
    strip_prefix = "abseil-cpp-20240722.0",
    url = "https://github.com/abseil/abseil-cpp/archive/20240722.0.tar.gz",
)

# The following enables the use of the library functions in the differential-
# privacy github repo
http_archive(
    name = "com_google_cc_differential_privacy",
    sha256 = "6e6e1cd7a819695caae408f4fa938129ab7a86e83fe2410137c85e50131abbe0",
    strip_prefix = "differential-privacy-3.0.0/cc",
    url = "https://github.com/google/differential-privacy/archive/refs/tags/v3.0.0.tar.gz",
)

http_archive(
    name = "com_google_differential_privacy",
    sha256 = "6e6e1cd7a819695caae408f4fa938129ab7a86e83fe2410137c85e50131abbe0",
    strip_prefix = "differential-privacy-3.0.0",
    url = "https://github.com/google/differential-privacy/archive/refs/tags/v3.0.0.tar.gz",
)

http_archive(
    name = "com_google_googleapis",
    sha256 = "249d83abc5d50bf372c35c49d77f900bff022b2c21eb73aa8da1458b6ac401fc",
    strip_prefix = "googleapis-6b3fdcea8bc5398be4e7e9930c693f0ea09316a0",
    url = "https://github.com/googleapis/googleapis/archive/6b3fdcea8bc5398be4e7e9930c693f0ea09316a0.tar.gz",
)

http_archive(
    name = "com_google_protobuf",
    integrity = "sha256-oZHSr911mXuln2IBlCUBZwPa7TVqnZL3Ql9HQUOa5UQ=",
    strip_prefix = "protobuf-29.5",
    url = "https://github.com/protocolbuffers/protobuf/releases/download/v29.5/protobuf-29.5.tar.gz",
)

http_archive(
    name = "com_google_sentencepiece",
    build_file = "@gemma//bazel:sentencepiece.bazel",
    patch_args = ["-p1"],
    patches = ["@gemma//bazel:sentencepiece.patch"],
    repo_mapping = {"@abseil-cpp": "@com_google_absl"},
    sha256 = "8409b0126ebd62b256c685d5757150cf7fcb2b92a2f2b98efb3f38fc36719754",
    strip_prefix = "sentencepiece-0.1.96",
    urls = ["https://github.com/google/sentencepiece/archive/refs/tags/v0.1.96.zip"],
)

# The following enables the use of the library functions in the gemma.cpp
# github repo.
http_archive(
    name = "darts_clone",
    build_file_content = """
licenses(["notice"])
exports_files(["LICENSE"])
package(default_visibility = ["//visibility:public"])
cc_library(
    name = "darts_clone",
    hdrs = [
        "include/darts.h",
    ],
)
""",
    sha256 = "c97f55d05c98da6fcaf7f9ecc6a6dc6bc5b18b8564465f77abff8879d446491c",
    strip_prefix = "darts-clone-e40ce4627526985a7767444b6ed6893ab6ff8983",
    urls = [
        "https://github.com/s-yata/darts-clone/archive/e40ce4627526985a7767444b6ed6893ab6ff8983.zip",
    ],
)

http_archive(
    name = "federated-compute",
    integrity = "sha256-CH8uwoVD/9E2nxryZJto1+FW6lDTlwgt1viUZxvJLxU=",
    patches = ["//third_party/federated_compute:visibility.patch"],
    repo_mapping = {"@org_tensorflow": "@tf_proto_library"},
    strip_prefix = "federated-compute-5f25f0c0369fa5e6cc9fb05d044a2fd36953a868",
    url = "https://github.com/google/federated-compute/archive/5f25f0c0369fa5e6cc9fb05d044a2fd36953a868.tar.gz",
)

http_archive(
    name = "federated_language",
    repo_mapping = {
        "@protobuf": "@com_google_protobuf",
    },
    sha256 = "51e13f9ce23c9886f34e20c5f4bd7941b6335867405d3b4f7cbc704d6f89e820",
    strip_prefix = "federated-language-16e734b633e68b613bb92918e6f3304774853e9b",
    url = "https://github.com/google-parfait/federated-language/archive/16e734b633e68b613bb92918e6f3304774853e9b.tar.gz",
)

stub_repo(
    name = "flatbuffers",
    rules = {":build_defs.bzl": ["flatbuffer_cc_library"]},
)

http_archive(
    name = "gemma",
    sha256 = "9c9b1cb7d859d46fb6e8b3b814818111526bc236bb269d7496cba69f392fe5c1",
    strip_prefix = "gemma.cpp-73f1140dca92b42a5a5cf620ad3b6d9c0c35155e",
    url = "https://github.com/google/gemma.cpp/archive/73f1140dca92b42a5a5cf620ad3b6d9c0c35155e.tar.gz",
)

http_archive(
    name = "google_benchmark",
    sha256 = "6bc180a57d23d4d9515519f92b0c83d61b05b5bab188961f36ac7b06b0d9e9ce",
    strip_prefix = "benchmark-1.8.3",
    urls = ["https://github.com/google/benchmark/archive/refs/tags/v1.8.3.tar.gz"],
)

http_archive(
    name = "googletest",
    sha256 = "8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7",
    strip_prefix = "googletest-1.14.0",
    urls = ["https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz"],
)

http_archive(
    name = "highway",
    sha256 = "8bdfbe5c0fb548b284505aa52fe395b50142efd4a999ded20d672a268d8e28d1",
    strip_prefix = "highway-1d16731233de45a365b43867f27d0a5f73925300",
    urls = ["https://github.com/google/highway/archive/1d16731233de45a365b43867f27d0a5f73925300.tar.gz"],
)

# Avoid Java dependencies.
stub_repo(
    name = "io_grpc_grpc_java",
    rules = {":java_grpc_library.bzl": ["java_grpc_library"]},
)

git_repository(
    name = "libcppbor",
    build_file = "@federated-compute//third_party:libcppbor.BUILD.bzl",
    commit = "20d2be8672d24bfb441d075f82cc317d17d601f8",
    remote = "https://android.googlesource.com/platform/external/libcppbor",
)

http_archive(
    name = "llama_cpp",
    build_file = "//third_party/llama_cpp:llama_cpp.BUILD.bzl",
    integrity = "sha256-tE6qqF3MiQbSWpTfUDYqTK4KndvvJ0R27udIde7/LE4=",
    patch_args = ["-p1"],
    patches = ["//third_party/llama_cpp:add_build_info.patch"],
    strip_prefix = "llama.cpp-be48528b068111304e4a0bb82c028558b5705f05",
    url = "https://github.com/ggml-org/llama.cpp/archive/be48528b068111304e4a0bb82c028558b5705f05.tar.gz",
)

http_archive(
    name = "oak",
    integrity = "sha256-ZEZ+qDfHWNivUm3z9LkWG5q2F6udbjij8GTgUFmIl/k=",
    patches = [
        "//third_party/oak:oak_attestation_verification.patch",
        "@trusted_computations_platform//third_party/oak:session_binder.patch",
    ],
    strip_prefix = "oak-0beb0de86a546df46365cf4caef3ac8065e304e9",
    url = "https://github.com/project-oak/oak/archive/0beb0de86a546df46365cf4caef3ac8065e304e9.tar.gz",
)

http_archive(
    name = "org_tensorflow_federated",
    integrity = "sha256-bYba88quz4uRmGnwhCCR9FOJovhjRtlTZyYNY2KKp2Q=",
    strip_prefix = "tensorflow-federated-9eb9169473c67dc453507e708d487e3545a149c0",
    url = "https://github.com/google-parfait/tensorflow-federated/archive/9eb9169473c67dc453507e708d487e3545a149c0.tar.gz",
)

http_archive(
    name = "platforms",
    sha256 = "29742e87275809b5e598dc2f04d86960cc7a55b3067d97221c9abbc9926bff0f",
    url = "https://github.com/bazelbuild/platforms/releases/download/0.0.11/platforms-0.0.11.tar.gz",
)

http_archive(
    name = "raft_rs",
    patches = ["@trusted_computations_platform//third_party/raft_rs:bazel.patch"],
    sha256 = "e755de7613e7105c3bf90fb7742887cce7c7276723f16a9d1fe7c6053bd91973",
    strip_prefix = "raft-rs-10968a112dcc4143ad19a1b35b6dca6e30d2e439",
    url = "https://github.com/google-parfait/raft-rs/archive/10968a112dcc4143ad19a1b35b6dca6e30d2e439.tar.gz",
)

http_archive(
    name = "rules_cc",
    sha256 = "abc605dd850f813bb37004b77db20106a19311a96b2da1c92b789da529d28fe1",
    strip_prefix = "rules_cc-0.0.17",
    url = "https://github.com/bazelbuild/rules_cc/releases/download/0.0.17/rules_cc-0.0.17.tar.gz",
)

http_archive(
    name = "rules_oci",
    patches = ["//third_party/rules_oci:zot_rbe.patch"],
    sha256 = "4a276e9566c03491649eef63f27c2816cc222f41ccdebd97d2c5159e84917c3b",
    strip_prefix = "rules_oci-1.7.4",
    url = "https://github.com/bazel-contrib/rules_oci/releases/download/v1.7.4/rules_oci-v1.7.4.tar.gz",
)

http_archive(
    name = "rules_proto",
    sha256 = "6fb6767d1bef535310547e03247f7518b03487740c11b6c6adb7952033fe1295",
    strip_prefix = "rules_proto-6.0.2",
    url = "https://github.com/bazelbuild/rules_proto/releases/download/6.0.2/rules_proto-6.0.2.tar.gz",
)

http_archive(
    name = "rules_python",
    sha256 = "dc6e2756130fafb90273587003659cadd1a2dfef3f6464c227794cdc01ebf70e",
    strip_prefix = "rules_python-0.33.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.33.0/rules_python-0.33.0.tar.gz",
)

http_archive(
    name = "rules_rust",
    integrity = "sha256-CeF7R8AVBGVjGqMZ8nQnYKQ+3tqy6cAS+R0K4u/wImg=",
    urls = ["https://github.com/bazelbuild/rules_rust/releases/download/0.59.2/rules_rust-0.59.2.tar.gz"],
)

http_archive(
    name = "rules_rust_bindgen",
    integrity = "sha256-CeF7R8AVBGVjGqMZ8nQnYKQ+3tqy6cAS+R0K4u/wImg=",
    strip_prefix = "extensions/bindgen",
    url = "https://github.com/bazelbuild/rules_rust/releases/download/0.59.2/rules_rust-0.59.2.tar.gz",
)

http_archive(
    name = "rules_rust_prost",
    integrity = "sha256-CeF7R8AVBGVjGqMZ8nQnYKQ+3tqy6cAS+R0K4u/wImg=",
    strip_prefix = "extensions/prost",
    url = "https://github.com/bazelbuild/rules_rust/releases/download/0.59.2/rules_rust-0.59.2.tar.gz",
)

http_archive(
    name = "sqlite",
    build_file = "//third_party/sqlite:BUILD.sqlite",
    sha256 = "65230414820d43a6d1445d1d98cfe57e8eb9f7ac0d6a96ad6932e0647cce51db",
    strip_prefix = "sqlite-amalgamation-3450200",
    url = "https://www.sqlite.org/2024/sqlite-amalgamation-3450200.zip",
)

http_archive(
    name = "sysroot",
    # Build provenance available at https://search.sigstore.dev/?hash=<sha256>.
    # To make this lookup easier, we use `sha256` instead of `integrity`.
    sha256 = "f58c289b3ccb28895ad8ca408ac366e709037088e8b5c28aca18212adc18c31e",
    url = "https://github.com/google-parfait/confidential-federated-compute/releases/download/sysroot-20250618/sysroot.tar.xz",
)

stub_tf_proto_library(
    name = "tf_proto_library",
)

http_archive(
    name = "toolchains_llvm",
    sha256 = "fded02569617d24551a0ad09c0750dc53a3097237157b828a245681f0ae739f8",
    strip_prefix = "toolchains_llvm-v1.4.0",
    url = "https://github.com/bazel-contrib/toolchains_llvm/releases/download/v1.4.0/toolchains_llvm-v1.4.0.tar.gz",
)

http_archive(
    name = "trusted_computations_platform",
    integrity = "sha256-jnq5xas/fINCtqbQUJAB0PTm8cjDEwN7bdsRXiEKjjo=",
    strip_prefix = "trusted-computations-platform-c685296de0ded451244e1f9592fb0b0a2fc80775",
    url = "https://github.com/google-parfait/trusted-computations-platform/archive/c685296de0ded451244e1f9592fb0b0a2fc80775.tar.gz",
)
# go/keep-sorted end

################################################################################
# Transitive dependencies managed via macros
################################################################################

load("@rules_rust//rust:repositories.bzl", "rules_rust_dependencies")

rules_rust_dependencies()

# Call py_repositories() first so rules_python can setup any state
# subsequent things might need. See
# https://github.com/bazelbuild/rules_python/issues/1560
load("@rules_python//python:repositories.bzl", "py_repositories", "python_register_toolchains")

py_repositories()

python_register_toolchains(
    name = "python",
    ignore_root_user_error = True,
    python_version = "3.10",
)

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")

grpc_extra_deps()

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies")

rules_proto_dependencies()

load("@rules_proto//proto:setup.bzl", "rules_proto_setup")

rules_proto_setup()

load("@rules_proto//proto:toolchains.bzl", "rules_proto_toolchains")

rules_proto_toolchains()

load("@oak//bazel:repositories.bzl", "oak_toolchain_repositories")

oak_toolchain_repositories()

load("@oak//bazel/crates:patched_crates.bzl", "load_patched_crates")

load_patched_crates()

load("@oak//bazel/rust:defs.bzl", "setup_rust_dependencies")

setup_rust_dependencies()

load("@rules_rust_prost//:repositories.bzl", "rust_prost_dependencies")

rust_prost_dependencies()

load("@rules_rust_prost//:transitive_repositories.bzl", "rust_prost_transitive_repositories")

rust_prost_transitive_repositories()

register_toolchains("//toolchains:prost_toolchain")

load("@oak//bazel/crates:repositories.bzl", "create_oak_crate_repositories")
load("@trusted_computations_platform//bazel:crates.bzl", "TCP_NO_STD_PACKAGES", "TCP_PACKAGES")
load("//:crates.bzl", "CFC_ANNOTATIONS", "CFC_NO_STD_PACKAGES", "CFC_PACKAGES")

create_oak_crate_repositories(
    extra_annotations = CFC_ANNOTATIONS,
    extra_no_std_packages = TCP_NO_STD_PACKAGES | CFC_NO_STD_PACKAGES,
    extra_packages = TCP_PACKAGES | CFC_PACKAGES,
)

load("@oak//bazel/crates:crates.bzl", "load_oak_crate_repositories")

load_oak_crate_repositories()

register_toolchains("//toolchains:bindgen_toolchain")

load("@rules_oci//oci:dependencies.bzl", "rules_oci_dependencies")

rules_oci_dependencies()

load("@rules_oci//oci:repositories.bzl", "LATEST_CRANE_VERSION", "LATEST_ZOT_VERSION", "oci_register_toolchains")

oci_register_toolchains(
    name = "oci",
    crane_version = LATEST_CRANE_VERSION,
    zot_version = LATEST_ZOT_VERSION,
)

load("@rules_oci//oci:pull.bzl", "oci_pull")

oci_pull(
    name = "distroless_cc_debian12",
    digest = "sha256:6714977f9f02632c31377650c15d89a7efaebf43bab0f37c712c30fc01edb973",
    image = "gcr.io/distroless/cc-debian12",
    platforms = ["linux/amd64"],
)

# Install a hermetic GCC toolchain for nostd builds. This must be defined after
# rules_oci because it uses an older version of aspect_bazel_lib.
http_archive(
    name = "aspect_gcc_toolchain",
    sha256 = "3341394b1376fb96a87ac3ca01c582f7f18e7dc5e16e8cf40880a31dd7ac0e1e",
    strip_prefix = "gcc-toolchain-0.4.2",
    url = "https://github.com/aspect-build/gcc-toolchain/archive/refs/tags/0.4.2.tar.gz",
)

load("@aspect_gcc_toolchain//toolchain:repositories.bzl", "gcc_toolchain_dependencies")

gcc_toolchain_dependencies()

load("@aspect_gcc_toolchain//toolchain:defs.bzl", "ARCHS", "gcc_register_toolchain")

gcc_register_toolchain(
    name = "gcc_toolchain_x86_64_unknown_none",
    extra_ldflags = ["-nostdlib"],
    target_arch = ARCHS.x86_64,
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:none",
    ],
)

load("@toolchains_llvm//toolchain:deps.bzl", "bazel_toolchain_dependencies")

bazel_toolchain_dependencies()

load("@toolchains_llvm//toolchain:rules.bzl", "llvm_toolchain")

llvm_toolchain(
    name = "llvm_toolchain",
    # Use LLVM version 14 as version 13 has a bug which causes asan to fail:
    # https://github.com/llvm/llvm-project/issues/51620
    llvm_version = "14.0.0",
    sysroot = {"linux-x86_64": "@sysroot"},
)

load("@llvm_toolchain//:toolchains.bzl", "llvm_register_toolchains")

llvm_register_toolchains()
