# Copyright 2025 Google LLC.
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

load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@rules_oci//oci:defs.bzl", "oci_image", "oci_load")

#load("@rules_oci//oci:defs.bzl", "oci_push")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [],
)

pkg_tar(
    name = "main_tar",
    srcs = [":main"],
    include_runfiles = True,
)

oci_image(
    name = "oci_image",
    base = "@distroless_cc_debian12_base",
    entrypoint = ["/main"],
    exposed_ports = ["8000/tcp"],
    tars = [
        # Install the main cc binary
        ":main_tar",
    ],
)

oci_load(
    name = "tarball",
    image = ":oci_image",
    repo_tags = [
        "gcp_prototype:latest",
    ],
)

# Fill in the correct repo setup
# oci_push(
#     name = "push",
#     image = ":oci_image",
#     repository = "us-docker.pkg.dev/${WORKER_PROJECT}/${WORKER_REPO}/${WORKER_IMAGE}",
#     remote_tags = ["latest"]
# )
