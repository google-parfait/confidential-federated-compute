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

# go/keep-sorted start block=yes newline_separated=yes ignore_prefixes=git_repository,http_archive,stub_repo
http_archive(
    name = "boringssl",
    integrity = "sha256-YfmrrozMej0Kon25c2SrYYc4NEHI0R3+MVBlzWEsXo8=",
    patches = ["//third_party/boringssl:rust.patch"],
    strip_prefix = "boringssl-34492c89a8e381e0e856a686cc71b1eb5bd728db",
    urls = ["https://github.com/google/boringssl/archive/34492c89a8e381e0e856a686cc71b1eb5bd728db.tar.gz"],
)

# Pin a newer version of gRPC than the one provided by Tensorflow.
# 1.50.0 is used as it is the gRPC version used by FCP. It's not a requirement
# for the versions to stay in sync if we need a feature of a later gRPC version,
# but simplifies the configuration of the two projects if we can use the same
# version.
http_archive(
    name = "com_github_grpc_grpc",
    patches = [
        "//third_party/grpc:noexcept.patch",
    ],
    sha256 = "76900ab068da86378395a8e125b5cc43dfae671e09ff6462ddfef18676e2165a",
    strip_prefix = "grpc-1.50.0",
    urls = ["https://github.com/grpc/grpc/archive/refs/tags/v1.50.0.tar.gz"],
)

# Avoid Kotlin dependencies.
stub_repo(
    name = "com_github_grpc_grpc_kotlin",
    rules = {":kt_jvm_grpc.bzl": ["kt_jvm_grpc_library"]},
)

http_archive(
    name = "com_google_absl",
    integrity = "sha256-AyBYaFZnTRawt6TUr7IhUb3HmEkLt/KV7d2PamK0b+o=",
    strip_prefix = "abseil-cpp-fb3621f4f897824c0dbe0615fa94543df6192f30",
    url = "https://github.com/abseil/abseil-cpp/archive/fb3621f4f897824c0dbe0615fa94543df6192f30.tar.gz",
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
    integrity = "sha256-OxuKW4Pq1QtHgv3YXkflZqk+k6RQKwA1ANfPWBcMObE=",
    patches = [
        "//third_party/federated_compute:libcppbor.patch",
        "//third_party/federated_compute:visibility.patch",
        "//third_party/federated_compute:executors.patch",
        "//third_party/federated_compute:min_sep.patch",
    ],
    strip_prefix = "federated-compute-5bcbbe6db0662f0a8db49c13ebdfd8c80ff46a1f",
    url = "https://github.com/google/federated-compute/archive/5bcbbe6db0662f0a8db49c13ebdfd8c80ff46a1f.tar.gz",
)

http_archive(
    name = "federated_language",
    patches = [
        "@org_tensorflow_federated//third_party/federated_language:numpy.patch",
        "@org_tensorflow_federated//third_party/federated_language:proto_library_loads.patch",
        "@org_tensorflow_federated//third_party/federated_language:python_deps.patch",
        "@org_tensorflow_federated//third_party/federated_language:structure_visibility.patch",
    ],
    repo_mapping = {
        "@protobuf": "@com_google_protobuf",
    },
    sha256 = "51e13f9ce23c9886f34e20c5f4bd7941b6335867405d3b4f7cbc704d6f89e820",
    strip_prefix = "federated-language-16e734b633e68b613bb92918e6f3304774853e9b",
    url = "https://github.com/google-parfait/federated-language/archive/16e734b633e68b613bb92918e6f3304774853e9b.tar.gz",
)

http_archive(
    name = "federated_language_jax",
    integrity = "sha256-KtH+Nfd3qYkvPcVjIH/NrJoXVgulCPo/dLzB7SGlCok=",
    patches = [
        "//third_party/federated_language_jax:eigen.patch",
        "//third_party/federated_language_jax:federated_language.patch",
        "//third_party/federated_language_jax:computation_visibility.patch",
    ],
    repo_mapping = {
        "@federated_language_jax_pypi": "@pypi",
        "@xla": "@local_xla",
    },
    strip_prefix = "federated-language-jax-182e5e9ec3cc0869a3588ea8048432c2c3922dfa",
    url = "https://github.com/google-parfait/federated-language-jax/archive/182e5e9ec3cc0869a3588ea8048432c2c3922dfa.tar.gz",
)

http_archive(
    name = "gemma",
    sha256 = "748eca98c85d0fe3e5134eb11c68ea8ec12a94b88f520fce1fafb4e03f0e11ff",
    strip_prefix = "gemma.cpp-0.1.4",
    url = "https://github.com/google/gemma.cpp/archive/refs/tags/v0.1.4.tar.gz",
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
    sha256 = "a9e04d39eba738e4265ed03bcc05562677d224bf16da6538ba7b236c8163a727",
    strip_prefix = "highway-c5bebf84ad01edec97e336f5c97ca4e0df6b4d06",
    urls = ["https://github.com/google/highway/archive/c5bebf84ad01edec97e336f5c97ca4e0df6b4d06.tar.gz"],
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
    patches = ["//third_party/libcppbor:libcppbor.patch"],
    remote = "https://android.googlesource.com/platform/external/libcppbor",
)

http_archive(
    name = "oak",
    integrity = "sha256-0IgL2etptmbLQ22HY73KG1Z0U53+e9MqBbzy40qDFkw=",
    patches = ["//third_party/oak:oak_attestation_verification.patch"],
    strip_prefix = "oak-205aec221ca6a7c5d223641076470b1fb2db75a1",
    url = "https://github.com/project-oak/oak/archive/205aec221ca6a7c5d223641076470b1fb2db75a1.tar.gz",
)

http_archive(
    name = "org_tensorflow",
    integrity = "sha256-Bo7a/OKqz8vHYud/dQbnEucByjNWbvTn0ynKLOrA1fk=",
    patches = [
        "//third_party/org_tensorflow:cython.patch",
        "//third_party/org_tensorflow:internal_visibility.patch",
        "//third_party/org_tensorflow:protobuf.patch",
        "//third_party/org_tensorflow:zlib.patch",
    ],
    strip_prefix = "tensorflow-2b8e118d2975975fad52c2a53bc30bcdb429ba49",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/2b8e118d2975975fad52c2a53bc30bcdb429ba49.tar.gz",
    ],
)

http_archive(
    name = "org_tensorflow_federated",
    integrity = "sha256-/lu0O2fwZGigUeasjTHC6PWYQhwn/Z7mBnFY3NA8eHE=",
    patches = [
        "//third_party/org_tensorflow_federated:eigen.patch",
        "//third_party/org_tensorflow_federated:tensorflow_2_18.patch",
        "@federated_language_jax//third_party/tensorflow_federated:cpp_to_python_executor_visibility.patch",
    ],
    strip_prefix = "tensorflow-federated-1bf676525dce346636d86c6917e7a76a84807fb8",
    url = "https://github.com/google-parfait/tensorflow-federated/archive/1bf676525dce346636d86c6917e7a76a84807fb8.tar.gz",
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
    patch_args = ["-p1"],
    # Patch rules_cc to be compatible with the older version of protobuf we use.
    patches = ["@rules_rust//rust/private/3rdparty:rules_cc.patch"],
    sha256 = "abc605dd850f813bb37004b77db20106a19311a96b2da1c92b789da529d28fe1",
    strip_prefix = "rules_cc-0.0.17",
    url = "https://github.com/bazelbuild/rules_cc/releases/download/0.0.17/rules_cc-0.0.17.tar.gz",
)

# Avoid Java dependencies.
stub_repo(
    name = "rules_java",
    rules = {"java:defs.bzl": [
        "java_binary",
        "java_library",
        "java_lite_proto_library",
        "java_proto_library",
        "java_test",
    ]},
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

# Initialize hermetic python prior to loading Tensorflow deps.
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

# Add a clang C++ toolchain for use with sanitizers, as the GCC toolchain does
# not easily enable sanitizers to be used with tests. The clang toolchain is not
# registered, so that the registered gcc toolchain is used by default, but can
# be specified on the command line with
# --extra_toolchains=@llvm_toolchain//:cc-toolchain-x86_64-linux (as is
# configured in the .bazelrc when asan, tsan, or ubsan are enabled.)
http_archive(
    name = "toolchains_llvm",
    sha256 = "fded02569617d24551a0ad09c0750dc53a3097237157b828a245681f0ae739f8",
    strip_prefix = "toolchains_llvm-v1.4.0",
    url = "https://github.com/bazel-contrib/toolchains_llvm/releases/download/v1.4.0/toolchains_llvm-v1.4.0.tar.gz",
)

http_archive(
    name = "trusted_computations_platform",
    integrity = "sha256-2pdWbHbrdbq8P4rpqJC2Y9hJT+ZL2h2LrnG5HL1Sbvw=",
    strip_prefix = "trusted-computations-platform-225919e62214f4ce80f6e141fca65a879765c5c5",
    url = "https://github.com/google-parfait/trusted-computations-platform/archive/225919e62214f4ce80f6e141fca65a879765c5c5.tar.gz",
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
    python_version = "3.10",  # Keep in sync with repo env set in .bazelrc
)

load("@python//:defs.bzl", "interpreter")
load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "pypi",
    python_interpreter_target = interpreter,
    requirements_lock = "//:requirements_lock.txt",
)

load("@pypi//:requirements.bzl", "install_deps")

install_deps()

# The following is copied from TensorFlow's own WORKSPACE, see
# https://github.com/tensorflow/tensorflow/blob/v2.19.0-rc0/WORKSPACE#L68
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

load("@local_xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    local_wheel_dist_folder = "dist",
    local_wheel_inclusion_list = [
        "tensorflow*",
        "tf_nightly*",
    ],
    local_wheel_workspaces = ["//:WORKSPACE"],
    requirements = {
        "3.10": "//:requirements_lock_3_10.txt",
    },
)

load(
    "@local_xla//third_party/py:python_repo.bzl",
    "python_repository",
)

python_repository(name = "python_version_repo")

load(
    "@local_xla//third_party/py:python_wheel.bzl",
    "python_wheel_version_suffix_repository",
)

python_wheel_version_suffix_repository(name = "tf_wheel_version_suffix")

load(
    "@local_xla//third_party/gpus/cuda/hermetic:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@local_xla//third_party/gpus/cuda/hermetic:cuda_redist_init_repositories.bzl",
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
    "@local_xla//third_party/gpus/cuda/hermetic:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@local_xla//third_party/nccl/hermetic:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@local_xla//third_party/nccl/hermetic:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies")

rules_proto_dependencies()

load("@rules_proto//proto:setup.bzl", "rules_proto_setup")

rules_proto_setup()

load("@rules_proto//proto:toolchains.bzl", "rules_proto_toolchains")

rules_proto_toolchains()

load("@oak//bazel:repositories.bzl", "oak_toolchain_repositories")

oak_toolchain_repositories()

load("@oak//bazel/rust:deps.bzl", "load_rust_repositories")

load_rust_repositories()

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
    # Using the PyPi tensorflow package requires using libstdc++ to ensure the
    # right C++ ABI.
    stdlib = {"linux-x86_64": "stdc++"},
    sysroot = {"linux-x86_64": "@sysroot"},
)

load("@llvm_toolchain//:toolchains.bzl", "llvm_register_toolchains")

llvm_register_toolchains()
