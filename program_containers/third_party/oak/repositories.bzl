# Copyright 2026 Google LLC.
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

"""Helpers for loading Oak toolchains and Rust crate repositories in WORKSPACE mode."""

load("@oak//bazel/crates:patched_crates.bzl", "load_patched_crates")
load("@oak//bazel/rust:defs.bzl", "RUST_EDITION", "RUST_SHA256S", "RUST_VERSIONS")
load("@oak//bazel/tools/umoci:umoci_toolchain.bzl", "umoci_toolchain_repo")
load("@rules_rust//crate_universe:repositories.bzl", "crate_universe_dependencies")
load("@rules_rust//rust:repositories.bzl", "rules_rust_dependencies", "rust_register_toolchains")

def _oak_std_crates_from_lockfile_impl(repository_ctx):
    lock_content = repository_ctx.read(repository_ctx.attr.lockfile)
    lock_data = json.decode(lock_content)

    crate_ext = lock_data["moduleExtensions"]["@@rules_rust+//crate_universe:extensions.bzl%crate"]["general"]["generatedRepoSpecs"]
    std_index_contents = crate_ext["oak_std_crates_index"]["attributes"]["contents"]

    repository_ctx.file("BUILD.bazel", std_index_contents["BUILD.bazel"], executable = False)
    repository_ctx.file("alias_rules.bzl", std_index_contents["alias_rules.bzl"], executable = False)

    defs_lines = [
        "load(\"@bazel_tools//tools/build_defs/repo:http.bzl\", \"http_archive\")",
        "load(\"@bazel_tools//tools/build_defs/repo:utils.bzl\", \"maybe\")",
        "",
        "def crate_repositories():",
    ]
    for repo_name, spec in sorted(crate_ext.items()):
        if not repo_name.startswith("oak_std_crates_index__"):
            continue
        attrs = spec["attributes"]
        patches = [
            p.replace("@@//", "@oak//") if p.startswith("@@//") else p
            for p in attrs.get("patches", [])
        ]
        defs_lines.append("    maybe(")
        defs_lines.append("        http_archive,")
        defs_lines.append("        name = {},".format(repr(repo_name)))
        defs_lines.append("        build_file_content = {},".format(repr(attrs["build_file_content"])))
        defs_lines.append("        patch_args = {},".format(repr(attrs.get("patch_args", []))))
        defs_lines.append("        patch_tool = {},".format(repr(attrs.get("patch_tool", ""))))
        defs_lines.append("        patches = {},".format(repr(patches)))
        defs_lines.append("        remote_patch_strip = {},".format(repr(attrs.get("remote_patch_strip", 0))))
        defs_lines.append("        sha256 = {},".format(repr(attrs["sha256"])))
        defs_lines.append("        strip_prefix = {},".format(repr(attrs["strip_prefix"])))
        defs_lines.append("        type = {},".format(repr(attrs["type"])))
        defs_lines.append("        urls = {},".format(repr(attrs["urls"])))
        defs_lines.append("    )")

    repository_ctx.file("defs.bzl", "\n".join(defs_lines) + "\n", executable = False)

_oak_std_crates_from_lockfile = repository_rule(
    implementation = _oak_std_crates_from_lockfile_impl,
    attrs = {
        "lockfile": attr.label(
            default = Label("@oak//:MODULE.bazel.lock"),
            allow_single_file = True,
        ),
    },
)

def _oak_aliased_crates_from_lockfile_impl(repository_ctx):
    lock_content = repository_ctx.read(repository_ctx.attr.lockfile)
    lock_data = json.decode(lock_content)
    aliased_ext = lock_data["moduleExtensions"]["//bazel/crates:extensions.bzl%aliased_crates_repository"]["general"]["generatedRepoSpecs"]
    raw_repos = aliased_ext["oak_crates_index"]["attributes"]["repositories"]

    repositories = []
    for raw in raw_repos:
        repo = json.decode(raw)
        if repo["name"] != "oak_std_crates_index":
            continue
        repositories.append(repo)

    packages = {}
    for repo in repositories:
        for pkg in repo["packages"]:
            packages[pkg] = True

    contents = "package(default_visibility = [\"//visibility:public\"])\n"
    for package in sorted(packages.keys()):
        contents += (
            "\n" +
            "alias(\n" +
            "    name = \"{}\",\n".format(package) +
            "    actual = \"@oak_std_crates_index//:{}\",\n".format(package) +
            ")\n"
        )
    repository_ctx.file("BUILD", contents, executable = False)

_oak_aliased_crates_from_lockfile = repository_rule(
    implementation = _oak_aliased_crates_from_lockfile_impl,
    attrs = {
        "lockfile": attr.label(
            default = Label("@oak//:MODULE.bazel.lock"),
            allow_single_file = True,
        ),
    },
)

# buildifier: disable=unnamed-macro
def oak_toolchain_repositories():
    """Registers the umoci toolchain used by oci_runtime_bundle."""
    umoci_toolchain_repo(name = "umoci")
    native.register_toolchains("@umoci//:umoci_toolchain", "@umoci//:umoci_toolchain_darwin")

# buildifier: disable=unnamed-macro
def setup_rust_dependencies():
    """Registers the Rust toolchains configured in @oak//bazel/rust:defs.bzl."""
    rules_rust_dependencies()
    crate_universe_dependencies()
    rust_register_toolchains(
        edition = RUST_EDITION,
        sha256s = RUST_SHA256S,
        versions = RUST_VERSIONS,
    )

# buildifier: disable=unnamed-macro
def create_oak_crate_repositories():
    """Creates @oak_std_crates_index and @oak_crates_index from @oak//:MODULE.bazel.lock."""
    load_patched_crates()
    _oak_std_crates_from_lockfile(name = "oak_std_crates_index")
    _oak_aliased_crates_from_lockfile(name = "oak_crates_index")
