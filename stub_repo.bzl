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

"""Repository rule for stubbing out unwanted rules in transitive deps."""

def _stub_repo_impl(rctx):
    packages = {"": True}
    for path, rules in rctx.attr.rules.items():
        package, file = path.split(":", 2)
        if package:
            package += "/"
        packages[package] = True
        content = "\n\n".join([
            "def {}(*args, **kwargs): pass\n".format(r) for r in rules
        ])
        rctx.file(package + file, content=content, executable=False)
    for p in packages.keys():
        rctx.file(p + "BUILD", executable=False)


stub_repo = repository_rule(
    _stub_repo_impl,
    doc = """Creates a repository containing no-op rules.

Example:
    stub_repo(
        name = "rules_java",
        rules = {"java:defs.bzl": ["java_proto_library"]},
    )
""",
    attrs = {
        "rules": attr.string_list_dict(
            doc = "List of rules to stub out per bzl file",
        ),
    },
)
