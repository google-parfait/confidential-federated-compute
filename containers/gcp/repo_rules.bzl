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

"""Custom repository rules for the GCP prototype."""

def _gcs_file_impl(ctx):
    """Downloads a file from a GCS bucket using gcloud or gsutil.

    The GCS URL can be overridden at build time by setting the GCS_MODEL_BUCKET
    environment variable via --repo_env. This replaces only the bucket portion
    of the URL, keeping the filename intact.

    Example:
        # Default (uses bucket from MODULE.bazel):
        bazel build :model_layer_gemma4_e4b

        # Override to a different bucket:
        bazel build :model_layer_gemma4_e4b \
            --repo_env=GCS_MODEL_BUCKET=gs://my-other-bucket
    """
    output = ctx.attr.downloaded_file_path
    gcs_url = ctx.attr.gcs_url

    # Allow overriding the GCS bucket via --repo_env=GCS_MODEL_BUCKET=gs://...
    bucket_override = ctx.os.environ.get("GCS_MODEL_BUCKET")
    if bucket_override:
        # Replace the bucket portion, keep the filename.
        filename = gcs_url.rsplit("/", 1)[-1]
        gcs_url = bucket_override.rstrip("/") + "/" + filename

        # buildifier: disable=print
        print("GCS_MODEL_BUCKET override: downloading from {}".format(gcs_url))

    # Build an environment that propagates GCP credentials (for CI) and
    # PATH (so ctx.execute can find the gcloud / gsutil binaries).
    exec_env = {}
    cred_file = ctx.os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if cred_file:
        exec_env["GOOGLE_APPLICATION_CREDENTIALS"] = cred_file
    path = ctx.os.environ.get("PATH", "")
    if path:
        exec_env["PATH"] = path

    # Try gcloud first (preferred), then fall back to gsutil.
    gcloud_result = ctx.execute(
        ["gcloud", "storage", "cp", gcs_url, output],
        timeout = 1800,  # 30 min for large model files
        environment = exec_env,
    )
    if gcloud_result.return_code != 0:
        gsutil_result = ctx.execute(
            ["gsutil", "cp", gcs_url, output],
            timeout = 1800,
            environment = exec_env,
        )
        if gsutil_result.return_code != 0:
            fail("Failed to download {} from GCS.\ngcloud error: {}\ngsutil error: {}".format(
                gcs_url,
                gcloud_result.stderr,
                gsutil_result.stderr,
            ))

    ctx.file("BUILD", 'exports_files(["' + output + '"])')

gcs_file = repository_rule(
    implementation = _gcs_file_impl,
    attrs = {
        "gcs_url": attr.string(mandatory = True),
        "downloaded_file_path": attr.string(mandatory = True),
    },
    # Re-fetch if GCS_MODEL_BUCKET or credentials change.
    environ = ["GCS_MODEL_BUCKET", "GOOGLE_APPLICATION_CREDENTIALS"],
)

def _curl_file_impl(ctx):
    """Downloads a file using curl."""
    url = ctx.attr.url

    # Download to file/file to match http_file behavior
    ctx.execute(["mkdir", "-p", "file"])
    result = ctx.execute(["curl", "-s", url, "-o", "file/file"])
    if result.return_code != 0:
        fail("Failed to download from {}: {}".format(url, result.stderr))

    ctx.file("file/BUILD", 'exports_files(["file"])')

curl_file = repository_rule(
    implementation = _curl_file_impl,
    attrs = {
        "url": attr.string(mandatory = True),
    },
)
