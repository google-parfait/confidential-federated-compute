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

load("@rules_pkg//pkg:install.bzl", "pkg_install")
load("@rules_pkg//pkg:mappings.bzl", "pkg_files")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")

package(features = ["layering_check"])

compile_pip_requirements(
    name = "requirements",
    timeout = "long",
    srcs = [
        "//:requirements.in",
        "@federated_language//:requirements.in",
        "@federated_language_jax//:requirements.in",
    ],
    extra_args = [
        "--no-annotate",
        "--strip-extras",
    ],
    requirements_txt = "//:requirements_lock.txt",
)

# All release (i.e. production) binaries, along with their names in the
# destination directory.
_RELEASE_BINARIES = {
    # keep-sorted start
    "//program_worker:oci_runtime_bundle.tar": "program_worker/container.tar",
    # keep-sorted end
}

pkg_files(
    name = "release_binaries",
    srcs = _RELEASE_BINARIES.keys(),
    renames = _RELEASE_BINARIES,
)

pkg_install(
    name = "install_release_binaries",
    srcs = [":release_binaries"],
)

# All release and testing binaries, along with their names in the destination
# directory.
_ALL_BINARIES = _RELEASE_BINARIES | {
    # keep-sorted start
    # keep-sorted end
}

pkg_files(
    name = "all_binaries",
    srcs = _ALL_BINARIES.keys(),
    renames = _ALL_BINARIES,
)

pkg_install(
    name = "install_all_binaries",
    srcs = [":all_binaries"],
)
