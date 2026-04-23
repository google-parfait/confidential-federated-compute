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

"""Macros for generating per-model GCP inference container targets."""

load("@rules_oci//oci:defs.bzl", "oci_image", "oci_load")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load(":build_defs.bzl", "define_load_runner")

def gcp_model_targets(model_name, weights_label):
    """Generates model_layer, oci_image, oci_load, and load_runner targets for a model.

    Creates targets for all 4 attestation variants (ita, ita_alts, gca, gca_alts).

    Args:
        model_name: Short model identifier (e.g., "gemma4_31b"). Used in target names.
        weights_label: Bazel label for the GGUF weights file (e.g., "@gemma4_31b_weights//:file.gguf").
    """

    model_layer_name = "model_layer_" + model_name

    pkg_tar(
        name = model_layer_name,
        srcs = [weights_label],
        mode = "0444",
        package_dir = "/saved_model",
    )

    _ATTESTATION_VARIANTS = [
        {
            "suffix": "ita",
            "cmd_extra": ["--attestation_provider=ita"],
        },
        {
            "suffix": "ita_alts",
            "cmd_extra": ["--attestation_provider=ita", "--use_alts=true"],
        },
        {
            "suffix": "gca",
            "cmd_extra": ["--attestation_provider=gca"],
        },
        {
            "suffix": "gca_alts",
            "cmd_extra": ["--attestation_provider=gca", "--use_alts=true"],
        },
    ]

    for variant in _ATTESTATION_VARIANTS:
        suffix = variant["suffix"]
        image_name = "batched_inference_gcp_{suffix}_{model}_oci_image".format(
            model = model_name,
            suffix = suffix,
        )
        tarball_name = "batched_inference_gcp_{suffix}_{model}_tarball".format(
            model = model_name,
            suffix = suffix,
        )
        repo_tag = "batched_inference_gcp_{suffix}_{model}:latest".format(
            model = model_name,
            suffix = suffix,
        )

        oci_image(
            name = image_name,
            base = "@distroless_cc_debian12_base",
            cmd = variant["cmd_extra"] + [
                "--gpu_layers=999",
                "--port=8000",
            ],
            entrypoint = ["/batched_inference_gcp_main"],
            env = {"LD_LIBRARY_PATH": "/usr/local/nvidia/lib64:/usr/local/nvidia/lib"},
            exposed_ports = ["8000/tcp"],
            tars = [
                ":batched_inference_gcp_tar",
                ":" + model_layer_name,
            ],
        )

        oci_load(
            name = tarball_name,
            image = ":" + image_name,
            repo_tags = [repo_tag],
        )

        define_load_runner(
            "batched_inference_gcp_{suffix}_{model}".format(
                model = model_name,
                suffix = suffix,
            ),
            repo_tag,
        )
