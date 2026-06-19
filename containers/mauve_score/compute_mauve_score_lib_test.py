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

"""Tests for compute_mauve_score_lib."""

import numpy as np
import unittest

from compute_mauve_score_lib import (
    compute_mauve,
    compute_mauve_and_serialize_result,
    compute_pca_embeddings,
    get_discretized_distribution,
    get_kmeans_assignments,
    get_mauve_curve,
    robust_kl,
)

from fcp.protos.confidentialcompute import mauve_score_config_pb2


class RobustKlTest(unittest.TestCase):

  def test_basic(self):
    p = np.array([0.5, 0.5])
    q = np.array([0.25, 0.75])
    expected = 0.5 * np.log(2.0) + 0.5 * np.log(2.0 / 3.0)
    self.assertAlmostEqual(robust_kl(p, q), expected)

  def test_disjoint_support_returns_inf(self):
    p = np.array([0.5, 0.5])
    q = np.array([0.0, 1.0])
    self.assertTrue(np.isinf(robust_kl(p, q)))


class GetDiscretizedDistributionTest(unittest.TestCase):

  def test_basic(self):
    assignments = np.array([0, 0, 1, 0])
    dist = get_discretized_distribution(assignments, n_clusters=2)
    np.testing.assert_allclose(dist, np.array([0.75, 0.25]))


class ComputeMauveTest(unittest.TestCase):

  def test_end_to_end(self):
    """Smoke test: runs without crashing on dummy data."""
    rng = np.random.RandomState(42)
    embeds_p = rng.randn(5, 10)
    embeds_q = rng.randn(5, 10)
    out = compute_mauve(embeds_p, embeds_q, n_clusters=2, use_pca=False)
    self.assertGreaterEqual(out.auc, 0.0)
    self.assertLessEqual(out.auc, 1.0 + 1e-9)
    self.assertAlmostEqual(out.discretized_p.sum(), 1.0)
    self.assertAlmostEqual(out.discretized_q.sum(), 1.0)


class ComputeMauveAndSerializeResultTest(unittest.TestCase):
  """Tests for the R6+R7 entry point used by C++."""

  def test_returns_valid_proto_bytes(self):
    rng = np.random.RandomState(123)
    dim = 10
    embeds_p = rng.randn(5, dim).astype(np.float32)
    embeds_q = rng.randn(5, dim).astype(np.float32)

    # Simulate what C++ does: flatten to bytes.
    real_bytes = embeds_p.tobytes()
    synth_bytes = embeds_q.tobytes()

    serialized = compute_mauve_and_serialize_result(
        real_bytes, dim, synth_bytes, dim
    )

    # Deserialize and validate.
    result = mauve_score_config_pb2.MauveScoreResult()
    result.ParseFromString(serialized)

    self.assertGreaterEqual(result.mauve_auc, 0.0)
    self.assertGreater(result.num_clusters, 0)
    self.assertGreater(len(result.kl_metrics), 0)

  def test_cluster_counts_sum_to_num_clusters(self):
    rng = np.random.RandomState(456)
    dim = 8
    embeds_p = rng.randn(10, dim).astype(np.float32)
    embeds_q = rng.randn(10, dim).astype(np.float32)

    serialized = compute_mauve_and_serialize_result(
        embeds_p.tobytes(), dim, embeds_q.tobytes(), dim
    )
    result = mauve_score_config_pb2.MauveScoreResult()
    result.ParseFromString(serialized)

    total = (
        result.clusters_with_real_and_synth
        + result.clusters_with_real_only
        + result.clusters_with_synth_only
        + result.clusters_empty
    )
    self.assertEqual(total, result.num_clusters)


if __name__ == '__main__':
  unittest.main()
