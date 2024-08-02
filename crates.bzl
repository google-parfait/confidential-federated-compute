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

"""Rust crates required by this workspace."""

load("@rules_rust//crate_universe:defs.bzl", "crate")

# Crates used by both std and no_std builds.
_COMMON_PACKAGES = {
    "aes-gcm-siv": crate.spec(
        version = "0.11.1",
    ),
    "ahash": crate.spec(
        version = "0.8.3",
        default_features = False,
    ),
    "bitmask": crate.spec(
        default_features = False,
        version = "0.5.0",
    ),
    "byteorder": crate.spec(
        default_features = False,
        version = "1.4.3",
    ),
    "core2": crate.spec(
        default_features = False,
        version = "0.4.0",
    ),
    "libflate": crate.spec(
        default_features = False,
        version = "2.0.0",
    ),
    "rand": crate.spec(
        version = "0.8.0",
        default_features = False,
        features = ["alloc", "small_rng", "getrandom"],
    ),
    "slog": crate.spec(
        version = "2.2.0",
        default_features = False,
    ),
    "spin": crate.spec(
        version = "0.9.8",
    ),
}

# Crates used for std builds.
CFC_PACKAGES = _COMMON_PACKAGES | {
    "assert_cmd": crate.spec(
        version = "2.0.14",
    ),
    "getset": crate.spec(
        version = "0.1.1",
    ),
    "googletest": crate.spec(
        version = "0.11.0",
    ),
    "insta": crate.spec(
        version = "1.38.0",
    ),
    "mockall": crate.spec(
        version = "0.11.4",
    ),
    "slog-term": crate.spec(
        version = "2.9.0",
    ),
}

# Crates used for no_std builds.
CFC_NO_STD_PACKAGES = _COMMON_PACKAGES | {
    "base64": crate.spec(
        version = "0.22.1",
        default_features = False,
        features = ["alloc"],
    ),
    "libm": crate.spec(
        version = "0.2.8",
    ),
    "p384": crate.spec(
        version = "0.13.0",
        default_features = False,
        features = ["ecdsa", "pem"],
    ),
    "rlsf": crate.spec(
        version = "0.2.1",
    ),
    "rsa": crate.spec(
        version = "0.9.6",
        default_features = False,
    ),
    "serde": crate.spec(
        version = "1.0.195",
        default_features = False,
        features = ["derive"],
    ),
    "serde_json": crate.spec(
        version = "1.0.113",
        default_features = False,
        features = ["alloc"],
    ),
    "time": crate.spec(
        version = "0.3.28",
        default_features = False,
        features = ["serde", "parsing"],
    ),
    "x509-cert": crate.spec(
        version = "0.2.5",
        default_features = False,
        features = ["pem"],
    ),
    "zerocopy": crate.spec(
        version = "0.7.32",
    ),
}

def _alias_crates_repository(repository_ctx):
    repositories = repository_ctx.attr.repositories

    overrides = {}
    for key, target in repository_ctx.attr.overrides.items():
        crate, condition = key.split(",", 1)
        overrides.setdefault(crate, {})[condition] = target

    crates = {}
    for repo_name in repositories.values():
        path = repository_ctx.path(Label("@{}//:BUILD.bazel".format(repo_name)))
        cmd = repository_ctx.execute(["sed", "-ne", "s/\\s*name = \"\\(.*\\)\",/\\1/p", path])
        if cmd.return_code != 0:
            fail("Failed to parse {}:\n{}".format(path, cmd.stderr))
        crates.update({crate: True for crate in cmd.stdout.splitlines()})

    contents = "package(default_visibility = [\"//visibility:public\"])\n"
    for crate in crates.keys():
        contents += """
alias(
    name = "{}",
    actual = select({{
""".format(crate)
        for condition, repo_name in repositories.items():
            default_target = "@{}//:{}".format(repo_name, crate)
            target = overrides.get(crate, {}).get(condition, default_target)
            contents += "        \"{}\": \"{}\",\n".format(condition, target)
        contents += """    }),
    tags = ["manual"],
)
"""

    repository_ctx.file("BUILD", contents, executable = False)

alias_crates_repository = repository_rule(
    implementation = _alias_crates_repository,
    doc = """Creates a repository that selects between other crates_repository rules.

Example:

    alias_crates_repository(
        name = "crates_index",
        repositories = {
            "@platforms//os:none": "no_std_crates_index",
            "//conditions:default": "std_crates_index",
        },
    )
""",
    attrs = {
        "repositories": attr.string_dict(
            mandatory = True,
            doc = "Map from select condition to repo name.",
        ),
        "overrides": attr.string_dict(
            doc = "Map from \"crate,condition\" to a label to use instead of the default.",
        ),
    },
)
