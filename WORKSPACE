# Copyright 2023 The Confidential Federated Compute Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Install a hermetic LLVM toolchain. LLVM was chosen somewhat arbitrarily; GCC
# should work as well.
http_archive(
    name = "com_grail_bazel_toolchain",
    sha256 = "d3d218287e76c0ad28bc579db711d1fa019fb0463634dfd944a1c2679ef9565b",
    strip_prefix = "bazel-toolchain-0.8",
    url = "https://github.com/grailbio/bazel-toolchain/archive/refs/tags/0.8.tar.gz",
)

load("@com_grail_bazel_toolchain//toolchain:deps.bzl", "bazel_toolchain_dependencies")

bazel_toolchain_dependencies()

load("@com_grail_bazel_toolchain//toolchain:rules.bzl", "llvm_toolchain")

llvm_toolchain(
    name = "llvm_toolchain",
    llvm_version = "14.0.0",
)

load("@llvm_toolchain//:toolchains.bzl", "llvm_register_toolchains")

llvm_register_toolchains()

http_archive(
    name = "com_github_grpc_grpc",
    sha256 = "931f07db9d48cff6a6007c1033ba6d691fe655bea2765444bc1ad974dfc840aa",
    strip_prefix = "grpc-1.56.2",
    url = "https://github.com/grpc/grpc/archive/refs/tags/v1.56.2.tar.gz",
)

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")

grpc_extra_deps()

load("@com_github_grpc_grpc//bazel:grpc_python_deps.bzl", "grpc_python_deps")

grpc_python_deps()

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")

rules_proto_dependencies()

rules_proto_toolchains()

load("@rules_python//python:repositories.bzl", "python_register_toolchains")

python_register_toolchains(
    name = "python_3_10",
    python_version = "3.10",
)

load("@python_3_10//:defs.bzl", "interpreter")
load("@rules_python//python:pip.bzl", "pip_parse")

# Create a central repo that knows about the dependencies needed from
# requirements.txt.
pip_parse(
   name = "pypi_deps",
   requirements_lock = "//tff_worker:requirements.txt",
   python_interpreter_target = interpreter,
)
# Load the starlark macro, which will define your dependencies.
load("@pypi_deps//:requirements.bzl", "install_deps")
# Call it to define repos for your requirements.
install_deps()

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
git_repository(
    name = "federated-compute",
    remote = "https://github.com/google/federated-compute.git",
    commit = "e7378f8c9e049702a3a55d086eec430c2762787b",
    patches = [
        "//third_party/federated_compute:pipeline_transform.patch",
    ],
)
