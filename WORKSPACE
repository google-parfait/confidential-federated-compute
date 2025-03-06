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

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Pin a newer version of gRPC than the one provided by Tensorflow.
# 1.50.0 is used as it is the gRPC version used by FCP. It's not a requirement
# for the versions to stay in sync if we need a feature of a later gRPC version,
# but simplifies the configuration of the two projects if we can use the same
# version.
http_archive(
    name = "com_github_grpc_grpc",
    sha256 = "76900ab068da86378395a8e125b5cc43dfae671e09ff6462ddfef18676e2165a",
    strip_prefix = "grpc-1.50.0",
    urls = ["https://github.com/grpc/grpc/archive/refs/tags/v1.50.0.tar.gz"],
)

# TensorFlow pins an old version of upb that's compatible with their old
# version of gRPC, but not with the newer version we use. Pin the version that
# would be added by gRPC 1.50.0.
http_archive(
    name = "upb",
    sha256 = "017a7e8e4e842d01dba5dc8aa316323eee080cd1b75986a7d1f94d87220e6502",
    strip_prefix = "upb-e4635f223e7d36dfbea3b722a4ca4807a7e882e2",
    urls = [
        "https://storage.googleapis.com/grpc-bazel-mirror/github.com/protocolbuffers/upb/archive/e4635f223e7d36dfbea3b722a4ca4807a7e882e2.tar.gz",
        "https://github.com/protocolbuffers/upb/archive/e4635f223e7d36dfbea3b722a4ca4807a7e882e2.tar.gz",
    ],
)

http_archive(
    name = "rules_cc",
    sha256 = "2037875b9a4456dce4a79d112a8ae885bbc4aad968e6587dca6e64f3a0900cdf",
    strip_prefix = "rules_cc-0.0.9",
    # We intentionally chose an older version of rules_cc here because the newer
    # versions get cc_proto rule from protobuf, but the version of protobuf we use
    # doesn't have cc_proto rule. We can't easily update the version of protobuf
    # because we need to use a version compatible with TensorFlow.
    urls = ["https://github.com/bazelbuild/rules_cc/releases/download/0.0.9/rules_cc-0.0.9.tar.gz"],
)

http_archive(
    name = "rules_proto",
    sha256 = "6fb6767d1bef535310547e03247f7518b03487740c11b6c6adb7952033fe1295",
    strip_prefix = "rules_proto-6.0.2",
    url = "https://github.com/bazelbuild/rules_proto/releases/download/6.0.2/rules_proto-6.0.2.tar.gz",
)

# Initialize hermetic python prior to loading Tensorflow deps. See
# https://github.com/tensorflow/tensorflow/blob/v2.14.0-rc0/WORKSPACE#L6
http_archive(
    name = "rules_python",
    sha256 = "d71d2c67e0bce986e1c5a7731b4693226867c45bfe0b7c5e0067228a536fc580",
    strip_prefix = "rules_python-0.29.0",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.29.0/rules_python-0.29.0.tar.gz",
)

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

http_archive(
    name = "org_tensorflow_federated",
    patches = [
        # Patch to make TFF compatible with TF 2.18.
        "//third_party/org_tensorflow_federated:tensorflow_2_18.patch",
    ],
    sha256 = "8aeabf62a55860d89adefe71d0de6bb220afc707837760331bbfc0585b76b894",
    strip_prefix = "tensorflow-federated-d21797d7792cd8bcfcd5b326b2608fd0f8d082b3",
    url = "https://github.com/google-parfait/tensorflow-federated/archive/d21797d7792cd8bcfcd5b326b2608fd0f8d082b3.tar.gz",
)

http_archive(
    name = "federated_language",
    patches = [
        "@org_tensorflow_federated//third_party/federated_language:proto_library_loads.patch",
        "@org_tensorflow_federated//third_party/federated_language:python_deps.patch",
        "@org_tensorflow_federated//third_party/federated_language:structure_visibility.patch",
    ],
    repo_mapping = {
        "@protobuf": "@com_google_protobuf",
    },
    sha256 = "e2b13844d56233616d8ed664d15e155dbc6bb45743b6e5ce775a8553026b34a6",
    strip_prefix = "federated-language-b685d2243891f9d7ca3c5820cfd690b4ecdb9697",
    url = "https://github.com/google-parfait/federated-language/archive/b685d2243891f9d7ca3c5820cfd690b4ecdb9697.tar.gz",
)

# Use a newer version of BoringSSL than what TF gives us, so we can use
# functions like `EC_group_p256` (which was added in commit
# 417069f8b2fd6dd4f8c2f5f69de7c038a2397050).
http_archive(
    name = "boringssl",
    sha256 = "5d6be8b65198828b151e7e4a83c3e4d76b892c43c3fb01c63f03de62b420b70f",
    strip_prefix = "boringssl-47e850c41f43350699e1325a134ec88269cabe6b",
    urls = ["https://github.com/google/boringssl/archive/47e850c41f43350699e1325a134ec88269cabe6b.tar.gz"],
)

http_archive(
    name = "trusted_computations_platform",
    sha256 = "87f5ccaa551611816ab3799ce0c8ef58d5f780ed0f960c6954fd84a01a33b61d",
    strip_prefix = "trusted-computations-platform-7254e8c2d029609f5cb65ca72a01d3ce61198bd9",
    url = "https://github.com/google-parfait/trusted-computations-platform/archive/7254e8c2d029609f5cb65ca72a01d3ce61198bd9.tar.gz",
)

http_archive(
    name = "rules_pkg",
    patches = ["@trusted_computations_platform//third_party/rules_pkg:tar.patch"],
    sha256 = "d20c951960ed77cb7b341c2a59488534e494d5ad1d30c4818c736d57772a9fef",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/1.0.1/rules_pkg-1.0.1.tar.gz",
        "https://github.com/bazelbuild/rules_pkg/releases/download/1.0.1/rules_pkg-1.0.1.tar.gz",
    ],
)

http_archive(
    name = "org_tensorflow",
    patches = [
        "//third_party/org_tensorflow:internal_visibility.patch",
        "//third_party/org_tensorflow:protobuf.patch",
    ],
    sha256 = "d7876f4bb0235cac60eb6316392a7c48676729860da1ab659fb440379ad5186d",
    strip_prefix = "tensorflow-2.18.0",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.18.0.tar.gz",
    ],
)

# The following is copied from TensorFlow's own WORKSPACE, see
# https://github.com/tensorflow/tensorflow/blob/v2.14.0-rc0/WORKSPACE#L68
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

load(
    "@local_tsl//third_party/gpus/cuda/hermetic:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@local_tsl//third_party/gpus/cuda/hermetic:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@local_tsl//third_party/gpus/cuda/hermetic:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@local_tsl//third_party/nccl/hermetic:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@local_tsl//third_party/nccl/hermetic:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")

http_archive(
    name = "lingvo",
    patches = ["//third_party/lingvo:lingvo.bzl.patch"],
    sha256 = "6baf671a81ed747f2580eb4f549044093c48173cd88b392510fe6ddf5dce2ba2",
    strip_prefix = "lingvo-ccfa97995bea99a3c0bb47b7b0b8e34a757ecf39",
    url = "https://github.com/tensorflow/lingvo/archive/ccfa97995bea99a3c0bb47b7b0b8e34a757ecf39.tar.gz",
)

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies")

rules_proto_dependencies()

load("@rules_proto//proto:setup.bzl", "rules_proto_setup")

rules_proto_setup()

load("@rules_proto//proto:toolchains.bzl", "rules_proto_toolchains")

rules_proto_toolchains()

http_archive(
    name = "federated-compute",
    patches = [
        "//third_party/federated_compute:libcppbor.patch",
        "//third_party/federated_compute:visibility.patch",
    ],
    sha256 = "b0bba256d146d5cde15d4f3a3a524fda337c3250c3613516f7c2e0a822cc4ccf",
    strip_prefix = "federated-compute-978bded209ba5e836b3aa2d63eff93677b8c074d",
    url = "https://github.com/google/federated-compute/archive/978bded209ba5e836b3aa2d63eff93677b8c074d.tar.gz",
)

http_archive(
    name = "raft_rs",
    patches = ["@trusted_computations_platform//third_party/raft_rs:bazel.patch"],
    sha256 = "e755de7613e7105c3bf90fb7742887cce7c7276723f16a9d1fe7c6053bd91973",
    strip_prefix = "raft-rs-10968a112dcc4143ad19a1b35b6dca6e30d2e439",
    url = "https://github.com/google-parfait/raft-rs/archive/10968a112dcc4143ad19a1b35b6dca6e30d2e439.tar.gz",
)

git_repository(
    name = "libcppbor",
    build_file = "@federated-compute//third_party:libcppbor.BUILD.bzl",
    commit = "20d2be8672d24bfb441d075f82cc317d17d601f8",
    remote = "https://android.googlesource.com/platform/external/libcppbor",
)

http_archive(
    name = "oak",
    sha256 = "5bb692898705ae3dcdb2e7a385e7afc681e3f18866120aee6b06d1615f5a4cf9",
    strip_prefix = "oak-d889956d9503c0459b96579e75ba34583d0809ae",
    url = "https://github.com/project-oak/oak/archive/d889956d9503c0459b96579e75ba34583d0809ae.tar.gz",
)

load("@oak//bazel:repositories.bzl", "oak_toolchain_repositories")

oak_toolchain_repositories()

load("@oak//bazel/rust:deps.bzl", "load_rust_repositories")

load_rust_repositories()

load("@oak//bazel/rust:defs.bzl", "setup_rust_dependencies")

setup_rust_dependencies()

load("@rules_rust//proto/prost:repositories.bzl", "rust_prost_dependencies")

rust_prost_dependencies()

load("@rules_rust//proto/prost:transitive_repositories.bzl", "rust_prost_transitive_repositories")

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

http_archive(
    name = "googletest",
    sha256 = "8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7",
    strip_prefix = "googletest-1.14.0",
    urls = ["https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz"],
)

http_archive(
    name = "sqlite",
    build_file = "//third_party/sqlite:BUILD.sqlite",
    sha256 = "65230414820d43a6d1445d1d98cfe57e8eb9f7ac0d6a96ad6932e0647cce51db",
    strip_prefix = "sqlite-amalgamation-3450200",
    url = "https://www.sqlite.org/2024/sqlite-amalgamation-3450200.zip",
)

http_archive(
    name = "com_google_absl",
    sha256 = "aabf6c57e3834f8dc3873a927f37eaf69975d4b28117fc7427dfb1c661542a87",
    strip_prefix = "abseil-cpp-98eb410c93ad059f9bba1bf43f5bb916fc92a5ea",
    urls = ["https://github.com/abseil/abseil-cpp/archive/98eb410c93ad059f9bba1bf43f5bb916fc92a5ea.zip"],
)

http_archive(
    name = "google_benchmark",
    sha256 = "6bc180a57d23d4d9515519f92b0c83d61b05b5bab188961f36ac7b06b0d9e9ce",
    strip_prefix = "benchmark-1.8.3",
    urls = ["https://github.com/google/benchmark/archive/refs/tags/v1.8.3.tar.gz"],
)

http_archive(
    name = "rules_oci",
    patches = ["//third_party/rules_oci:zot_rbe.patch"],
    sha256 = "4a276e9566c03491649eef63f27c2816cc222f41ccdebd97d2c5159e84917c3b",
    strip_prefix = "rules_oci-1.7.4",
    url = "https://github.com/bazel-contrib/rules_oci/releases/download/v1.7.4/rules_oci-v1.7.4.tar.gz",
)

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

# Install a hermetic GCC toolchain. This must be defined after rules_oci
# because it uses an older version of aspect_bazel_lib.
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
    name = "gcc_toolchain_x86_64",
    target_arch = ARCHS.x86_64,
)

gcc_register_toolchain(
    name = "gcc_toolchain_x86_64_unknown_none",
    extra_ldflags = ["-nostdlib"],
    target_arch = ARCHS.x86_64,
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:none",
    ],
)

# Add a clang C++ toolchain for use with sanitizers, as the GCC toolchain does
# not easily enable sanitizers to be used with tests. The clang toolchain is not
# registered, so that the registered gcc toolchain is used by default, but can
# be specified on the command line with
# --extra_toolchains=@llvm_toolchain//:cc-toolchain-x86_64-linux (as is
# configured in the .bazelrc when asan, tsan, or ubsan are enabled.)
http_archive(
    name = "toolchains_llvm",
    sha256 = "b7cd301ef7b0ece28d20d3e778697a5e3b81828393150bed04838c0c52963a01",
    strip_prefix = "toolchains_llvm-0.10.3",
    url = "https://github.com/grailbio/bazel-toolchain/releases/download/0.10.3/toolchains_llvm-0.10.3.tar.gz",
)

load("@toolchains_llvm//toolchain:deps.bzl", "bazel_toolchain_dependencies")

bazel_toolchain_dependencies()

load("@toolchains_llvm//toolchain:rules.bzl", "llvm_toolchain")

llvm_toolchain(
    name = "llvm_toolchain",
    # Use LLVM version 14 as version 13 has a bug which causes asan to fail:
    # https://github.com/llvm/llvm-project/issues/51620
    llvm_version = "14.0.0",
    sysroot = {"linux-x86_64": "@clang_sysroot//:sysroot"},
)

# Use a sysroot to make clang builds hermetic. This sysroot may not be
# reproducible, but that's OK because it's only used by sanitizer builds.
http_archive(
    name = "clang_sysroot",
    build_file_content = """
filegroup(
    name = "sysroot",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)
    """,
    sha256 = "dec7a3a0fc5b83b909cba1b6d119077e0429a138eadef6bf5a0f2e03b1904631",
    type = "tar.xz",
    url = "https://commondatastorage.googleapis.com/chrome-linux-sysroot/dec7a3a0fc5b83b909cba1b6d119077e0429a138eadef6bf5a0f2e03b1904631",
)

# Stub out unneeded Java proto library rules used by various dependencies. This
# avoids needing to depend on a Java toolchain.
load("//:stub_repo.bzl", "stub_repo")

stub_repo(
    name = "io_grpc_grpc_java",
    rules = {":java_grpc_library.bzl": ["java_grpc_library"]},
)

stub_repo(
    name = "rules_java",
    rules = {"java:defs.bzl": [
        "java_lite_proto_library",
        "java_proto_library",
    ]},
)

stub_repo(
    name = "com_github_grpc_grpc_kotlin",
    rules = {":kt_jvm_grpc.bzl": ["kt_jvm_grpc_library"]},
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
    name = "com_google_sentencepiece",
    build_file = "@gemma//bazel:sentencepiece.bazel",
    patch_args = ["-p1"],
    patches = ["@gemma//bazel:sentencepiece.patch"],
    repo_mapping = {"@abseil-cpp": "@com_google_absl"},
    sha256 = "8409b0126ebd62b256c685d5757150cf7fcb2b92a2f2b98efb3f38fc36719754",
    strip_prefix = "sentencepiece-0.1.96",
    urls = ["https://github.com/google/sentencepiece/archive/refs/tags/v0.1.96.zip"],
)

http_archive(
    name = "highway",
    sha256 = "eb07b6c8b9fc23cca4e2e166e30937ff4a13811d48f59ba87f2d2499470d4a63",
    strip_prefix = "highway-2b565e87d50b151660494624af532ac0b6076c79",
    urls = ["https://github.com/google/highway/archive/2b565e87d50b151660494624af532ac0b6076c79.tar.gz"],
)

http_archive(
    name = "gemma",
    sha256 = "ee01ac7165fbacddc5509e8384900165b72f349c1421efdd5b6f9391c3c1729e",
    strip_prefix = "gemma.cpp-9f5159ff683d99a7d852a0e66782e718588937df",
    url = "https://github.com/google/gemma.cpp/archive/9f5159ff683d99a7d852a0e66782e718588937df.tar.gz",
)
