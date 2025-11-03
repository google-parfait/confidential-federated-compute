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

def _http_auth_file_impl(ctx):
    token = ctx.os.environ.get("HF_TOKEN", "")

    auth_config = {}
    if token:
        # FIX: Use the full URL prefix, including 'https://'
        auth_config["https://huggingface.co"] = {"Authorization": "Bearer " + token}

    ctx.download(
        url = ctx.attr.urls,
        output = ctx.attr.downloaded_file_path,
        integrity = ctx.attr.integrity,
        auth = auth_config,
    )
    ctx.file("BUILD", 'exports_files(["' + ctx.attr.downloaded_file_path + '"])')

http_auth_file = repository_rule(
    implementation = _http_auth_file_impl,
    attrs = {
        "urls": attr.string_list(mandatory = True),
        "integrity": attr.string(),
        "downloaded_file_path": attr.string(mandatory = True),
    },
    environ = ["HF_TOKEN"],
)
