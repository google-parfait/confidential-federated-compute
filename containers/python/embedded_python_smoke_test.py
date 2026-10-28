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
"""Runs an embedded-Python binary extracted from a pkg_tar, as in a container.

Usage: embedded_python_smoke_test.py <path to embedded_python_smoke_tar.tar>
"""

import os
import subprocess
import sys
import tarfile
import tempfile
import unittest

_BINARY = "embedded_python_smoke_main"
_TAR_PATH = None


class EmbeddedPythonSmokeTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    # The binary finds its runfiles through /proc/self/exe, which has all
    # symlinks resolved.
    cls._root = os.path.realpath(
        tempfile.mkdtemp(dir=os.environ.get("TEST_TMPDIR"))
    )
    with tarfile.open(_TAR_PATH) as tar:
      cls._members = tar.getmembers()
      tar.extractall(cls._root, filter="tar")

  def assert_runs_from_extracted_tree(self, args, cwd, env):
    result = subprocess.run(
        args, cwd=cwd, env=env, capture_output=True, text=True, check=False
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    outputs = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    # Python comes from the extracted tree, not from the machine or from the
    # test's own runfiles.
    self.assertTrue(
        outputs["prefix"].startswith(self._root + "/"), result.stdout
    )
    self.assertTrue(
        outputs["os"].startswith(outputs["prefix"] + "/lib/python"),
        result.stdout,
    )
    self.assertEqual(outputs["greeting"], "hello from embedded python")
    # A package that, like pip packages, is found through its `imports`
    # directory.
    self.assertEqual(
        outputs["pip_style_package"], "imported from site-packages"
    )
    # The shipped interpreter binary runs from the extracted tree.
    self.assertEqual(outputs["subprocess"], "ok")

  def test_tar_contains_no_runfiles_manifest(self):
    names = [member.name for member in self._members]
    self.assertFalse(
        [n for n in names if n.endswith(("MANIFEST", "runfiles_manifest"))]
    )

  def test_tar_contains_no_symlinks_out_of_the_tree(self):
    for member in self._members:
      if member.issym():
        self.assertFalse(os.path.isabs(member.linkname), member.name)

  def test_runs_with_empty_environment(self):
    self.assert_runs_from_extracted_tree(
        [os.path.join(self._root, _BINARY)], cwd="/", env={}
    )

  def test_runs_through_path_lookup(self):
    self.assert_runs_from_extracted_tree(
        [_BINARY], cwd="/", env={"PATH": self._root}
    )

  def test_runs_through_symlink_in_other_directory(self):
    link_dir = tempfile.mkdtemp(dir=os.environ.get("TEST_TMPDIR"))
    link = os.path.join(link_dir, "linked_binary")
    os.symlink(os.path.join(self._root, _BINARY), link)
    self.assert_runs_from_extracted_tree([link], cwd=link_dir, env={})


if __name__ == "__main__":
  _TAR_PATH = os.path.abspath(sys.argv.pop(1))
  unittest.main()
