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

"""Offline verification test for server_image_registry.json."""

import json
import os
import unittest

from python.runfiles import runfiles
import provenance_lib


class VerifyRegistryTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        r = runfiles.Create()
        self.registry_path = r.Rlocation("_main/server_image_registry.json")
        self.repository = "google-parfait/confidential-federated-compute"

    def test_registry_file_exists_and_valid(self):
        self.assertTrue(
            self.registry_path and os.path.exists(self.registry_path),
            f"Registry file not found at: {self.registry_path}",
        )
        with open(self.registry_path, "r") as f:
            registry_data = json.load(f)
        images = registry_data.get("images", [])
        self.assertIsInstance(images, list)
        self.assertGreater(len(images), 0, "Registry contains no images.")

    def test_verify_all_entries_offline(self):
        with open(self.registry_path, "r") as f:
            registry_data = json.load(f)

        images = registry_data.get("images", [])
        self.assertGreater(len(images), 0)

        required_fields = ["model", "attestation", "digest", "tag", "created", "provenance"]
        for idx, entry in enumerate(images):
            model = entry.get("model", "<unknown>")
            digest = entry.get("digest", "<unknown>")
            with self.subTest(idx=idx, model=model, digest=digest):
                for field in required_fields:
                    self.assertIn(
                        field,
                        entry,
                        f"Registry entry {idx+1}/{len(images)} is missing required field '{field}'.",
                    )

                provenance = entry["provenance"]
                self.assertIsInstance(
                    provenance,
                    list,
                    f"Registry entry {idx+1}/{len(images)} ({model} - {digest}) has non-list 'provenance'.",
                )
                self.assertGreater(
                    len(provenance),
                    0,
                    f"Registry entry {idx+1}/{len(images)} ({model} - {digest}) has empty 'provenance' list.",
                )

                digest_clean = digest[7:] if digest.startswith("sha256:") else digest

                # 1. Cryptographically verify the embedded Sigstore bundle offline
                provenance_lib.verify_attestation_signatures(
                    provenance,
                    digest_clean,
                    repository=self.repository,
                    offline=True,
                )

                # 2. Extract and validate metadata (subject, gitCommit, workflow) from the bundle
                (
                    subject_name,
                    subject_digest,
                    unique_commits,
                    custom_metadata,
                    workflows,
                ) = provenance_lib.extract_provenance_metadata(
                    provenance, digest_clean
                )

                self.assertEqual(
                    subject_digest,
                    digest_clean,
                    f"Subject digest in attestation ({subject_digest}) does not match registry digest ({digest_clean}).",
                )


if __name__ == "__main__":
    # Map XDG_CACHE_HOME to TEST_TMPDIR when running inside a Bazel test sandbox
    # so that Sigstore's TUF client can write its cache to a writable directory.
    if "TEST_TMPDIR" in os.environ and "XDG_CACHE_HOME" not in os.environ:
        os.environ["XDG_CACHE_HOME"] = os.environ["TEST_TMPDIR"]

    unittest.main()
