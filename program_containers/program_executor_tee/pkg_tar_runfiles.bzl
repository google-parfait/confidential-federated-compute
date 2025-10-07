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

"""Define rule for packaging structured runfiles into a tar."""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@rules_pkg//:providers.bzl", "PackageFilesInfo")

def _pkg_runfiles_impl(ctx):
    """Implementation of pkg_tar_runfiles rule."""
    runfiles = ctx.runfiles().merge_all([
        s[DefaultInfo].default_runfiles
        for s in ctx.attr.srcs
    ])
    return [
        # Map each runfile to its path in the original runfiles tree.
        PackageFilesInfo(dest_src_map = {
            # The path needs to be normalized because the short_path may
            # contain parent directory segments. For example, a workspace_root
            # of A/foo and a short_path of ../B should become A/B.
            paths.normalize(
                paths.join(
                    ctx.attr.runfiles_prefix,
                    ctx.workspace_name,
                    file.owner.workspace_root,
                    file.short_path,
                ),
            ): file
            for file in runfiles.files.to_list()
        }),
        DefaultInfo(files = runfiles.files),
    ]

pkg_tar_runfiles = rule(
    implementation = _pkg_runfiles_impl,
    provides = [PackageFilesInfo],
    attrs = {
        "srcs": attr.label_list(
            mandatory = True,
            doc = "Targets with runfiles that need to be structured.",
        ),
        "runfiles_prefix": attr.string(
            doc = "Path prefix for the runfiles directory.",
            default = ".",
        ),
    },
    doc = """A rule to fix the runfiles structure for a tarball. The standard
             pkg_tar bazel rule collapses the runfiles directory for its srcs,
             meaning that when the srcs are executed, the runfiles (if
             included) are not found in the expected location. Providing a
             cc_binary C as a src for a target T of this type and then
             providing T as a src for a pkg_tar rule will result in a tarball
             where C can be executed with all its runfiles in the expected
             location.""",
)
