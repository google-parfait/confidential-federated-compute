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

"""Tests for the registry pruning logic in update_server_registry.py."""

import copy
import datetime
import unittest

import update_server_registry

NOW = datetime.datetime(2026, 10, 6, 12, 0, 0, tzinfo=datetime.timezone.utc)


def _entry(digest, created):
    return {
        "model": "gemma4_e4b",
        "attestation": "ita_alts",
        "digest": f"sha256:{digest}",
        "tag": "us-docker.pkg.dev/private-inference/offloading/batched_inference",
        "created": created,
    }


def _digests(registry):
    return [e["digest"] for e in registry["images"]]


class PruneRegistryTest(unittest.TestCase):

    def test_removes_entries_older_than_max_age(self):
        registry = {
            "images": [
                _entry("old", "2026-07-09T04:05:17Z"),
                _entry("boundary", "2026-08-07T12:00:01Z"),  # 59d 23h 59m 59s old
                _entry("fresh", "2026-10-01T00:00:00Z"),
            ]
        }

        pruned = update_server_registry.prune_registry(registry, 60, now=NOW)

        self.assertEqual(_digests(pruned), ["sha256:boundary", "sha256:fresh"])

    def test_does_not_modify_input_and_keeps_other_keys(self):
        registry = {
            "comment": "kept as-is",
            "images": [
                _entry("old", "2026-07-09T04:05:17Z"),
                _entry("fresh", "2026-10-01T00:00:00Z"),
            ],
        }
        snapshot = copy.deepcopy(registry)

        pruned = update_server_registry.prune_registry(registry, 60, now=NOW)

        self.assertEqual(registry, snapshot)
        self.assertIsNot(pruned, registry)
        self.assertEqual(pruned["comment"], "kept as-is")
        self.assertEqual(_digests(pruned), ["sha256:fresh"])

    def test_cutoff_boundary(self):
        registry = {
            "images": [
                _entry("at_cutoff", "2026-08-07T12:00:00Z"),  # exactly 60 days old
                _entry("past_cutoff", "2026-08-07T11:59:59Z"),
            ]
        }

        pruned = update_server_registry.prune_registry(registry, 60, now=NOW)

        self.assertEqual(_digests(pruned), ["sha256:at_cutoff"])

    def test_keeps_entries_without_valid_created(self):
        registry = {
            "images": [
                {"digest": "sha256:nodate"},
                _entry("baddate", "yesterday"),
                _entry("old", "2026-01-01T00:00:00Z"),
            ]
        }

        pruned = update_server_registry.prune_registry(registry, 60, now=NOW)

        self.assertEqual(_digests(pruned), ["sha256:nodate", "sha256:baddate"])

    def test_naive_timestamps_are_treated_as_utc(self):
        registry = {
            "images": [
                _entry("old", "2026-07-09T04:05:17"),
                _entry("fresh", "2026-10-01T00:00:00"),
            ]
        }

        pruned = update_server_registry.prune_registry(registry, 60, now=NOW)

        self.assertEqual(_digests(pruned), ["sha256:fresh"])

    def test_rejects_non_positive_max_age(self):
        registry = {"images": [_entry("old", "2026-01-01T00:00:00Z")]}

        for bad_max_age in (0, -1):
            with self.assertRaises(ValueError):
                update_server_registry.prune_registry(registry, bad_max_age, now=NOW)

    def test_preserves_order_of_kept_entries(self):
        registry = {
            "images": [
                _entry("a", "2026-09-01T00:00:00Z"),
                _entry("old", "2026-01-01T00:00:00Z"),
                _entry("b", "2026-09-02T00:00:00Z"),
                _entry("c", "2026-09-03T00:00:00Z"),
            ]
        }

        pruned = update_server_registry.prune_registry(registry, 60, now=NOW)

        self.assertEqual(_digests(pruned), ["sha256:a", "sha256:b", "sha256:c"])

    def test_empty_registry(self):
        self.assertEqual(
            update_server_registry.prune_registry({"images": []}, 60, now=NOW),
            {"images": []},
        )
        self.assertEqual(
            update_server_registry.prune_registry({}, 60, now=NOW),
            {"images": []},
        )


if __name__ == "__main__":
    unittest.main()
