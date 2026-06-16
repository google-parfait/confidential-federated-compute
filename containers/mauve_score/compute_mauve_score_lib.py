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

"""Implementation of the MAUVE metric core math, extracted to avoid ML bloat.

The goal of the metric is to measure how similar the machine-generated text
distribution is to actual human language. This library extracts pure numpy,
scipy, and scikit-learn logic from bard/mauve and privacy/mauve to be run
inside lightweight TEE containers.

[1] Pillutla et al., 2022, https://arxiv.org/pdf/2102.01454.pdf/.
"""

import dataclasses
import math
from typing import Optional, Union

import numpy as np
from fcp.protos.confidentialcompute import mauve_score_config_pb2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import auc


_MAX_N_CLUSTERS = 500
_MIN_N_CLUSTERS = 2
_DEFAULT_MAX_KMEANS_ITERATIONS = 500
_DEFAULT_N_KMEANS_INITIALISATIONS = 5
_DEFAULT_SCALING_C = 5.0
_DEFAULT_TARGET_PCA_EXPLAINED_VARIANCE = 0.9
_DEFAULT_N_LAMBDA_VALUES_FOR_CURVE = 32

# Lambda values for soft KL divergence penalties.
_LAMBDA_SOFT_VALUES = [0.001, 0.00001]


@dataclasses.dataclass
class MauveOutputs:
  """A data structure gathering the different outputs produced by MAUVE."""

  auc: float
  mauve_curve: np.ndarray
  pca_embeds: np.ndarray
  n_clusters: int
  kmeans_assignments: np.ndarray
  discretized_p: np.ndarray
  discretized_q: np.ndarray


def robust_kl(p: np.ndarray, q: np.ndarray) -> float:
  """Computes KL(p|q) = sum_k p_k log(p_k / q_k)."""
  if not (p.ndim == 1 and p.shape == q.shape):
    raise ValueError((
        'The two distributions must have the same shapes;'
        f' received {p.shape} and {q.shape}.'
    ))
  zero_q = q == 0
  nonzero_p = p > 0
  if np.any(zero_q & nonzero_p):
    return np.inf
  return float(np.sum(p[nonzero_p] * np.log(p[nonzero_p] / q[nonzero_p])))


def compute_pca_embeddings(
    embeds: np.ndarray,
    target_explained_var: float = _DEFAULT_TARGET_PCA_EXPLAINED_VARIANCE,
    whiten: bool = True,
    random_state: int = 0,
) -> np.ndarray:
  """Computes PCA embeddings."""
  pca = PCA(whiten=whiten, random_state=random_state)
  pca_embeds = pca.fit_transform(embeds)

  # First dimension such that at least `target_explained_var` is accounted for.
  valid_dims = np.cumsum(pca.explained_variance_ratio_) >= target_explained_var
  pca_cutoff = int(np.min(valid_dims.nonzero()) + 1)
  pca_embeds = pca_embeds[:, :pca_cutoff]

  # The rows (i.e., each embedding) must have L2 unit norm.
  sq_norms = np.sum(np.square(pca_embeds), axis=1, keepdims=True)
  pca_embeds *= 1.0 / np.sqrt(sq_norms + 1e-9)

  return pca_embeds


def get_kmeans_assignments(
    embeds: np.ndarray,
    max_iter: int = _DEFAULT_MAX_KMEANS_ITERATIONS,
    n_clusters: int = _MAX_N_CLUSTERS,
    n_init: int = _DEFAULT_N_KMEANS_INITIALISATIONS,
    random_state: int = 1,
) -> np.ndarray:
  """Performs K-means on the embeddings and outputs their assignments."""
  kmeans = KMeans(
      n_clusters=n_clusters,
      max_iter=max_iter,
      n_init=n_init,
      random_state=random_state,
  )
  kmeans.fit(embeds)
  return kmeans.labels_


def get_discretized_distribution(
    kmeans_assignments: np.ndarray, n_clusters: int = _MAX_N_CLUSTERS
) -> np.ndarray:
  """Discretizes the distribution based on the K-means assignments."""
  h, _ = np.histogram(
      kmeans_assignments, bins=n_clusters, range=(0, n_clusters)
  )
  # Compute Eq. (3) of https://arxiv.org/pdf/2102.01454.pdf.
  return h * 1.0 / np.sum(h)


def get_mauve_curve(
    p: np.ndarray,
    q: np.ndarray,
    n_lambda_values: int = _DEFAULT_N_LAMBDA_VALUES_FOR_CURVE,
    scaling_c: float = _DEFAULT_SCALING_C,
    include_end_points: bool = True,
) -> np.ndarray:
  """Compute the MAUVE curve.

  See Eq. (1) of https://arxiv.org/pdf/2102.01454.pdf.
  """
  border = 1e-9  # To avoid hitting support issues with the KL.
  lambdas = np.linspace(border, 1 - border, n_lambda_values)
  mauve_curve = []
  for l in lambdas:
    # Compute Eq. (1) of https://arxiv.org/pdf/2102.01454.pdf.
    r_l = l * p + (1 - l) * q
    mauve_curve.append([robust_kl(q, r_l), robust_kl(p, r_l)])

  if include_end_points:
    mauve_curve.append([0.0, np.inf])
    mauve_curve.append([np.inf, 0.0])

  mauve_curve = np.asarray(mauve_curve)
  mauve_curve = np.exp(-scaling_c * mauve_curve)
  # The sorting is required for the computation of the AUC.
  xaxis_sorting_indices = np.argsort(mauve_curve[:, 0])
  mauve_curve = mauve_curve[xaxis_sorting_indices, :]
  return mauve_curve


def _check_number_sentences(
    p_and_q_inputs: Union[
        tuple[list[str], list[str]], tuple[np.ndarray, np.ndarray]
    ],
):
  """Checks if enough sentences have been provided to compute MAUVE."""
  p, q = p_and_q_inputs
  if len(p) < _MIN_N_CLUSTERS or len(q) < _MIN_N_CLUSTERS:
    raise ValueError(
        f'MAUVE requires at least {_MIN_N_CLUSTERS} sentences for p and q;'
        f' received {len(p)} and {len(q)} sentences for p and q.'
    )


def compute_mauve(
    embeds_p: np.ndarray,
    embeds_q: np.ndarray,
    n_clusters: Optional[int] = None,
    use_pca: bool = True,
) -> MauveOutputs:
  """Computes the MAUVE score directly from the embedded sentences."""
  _check_number_sentences((embeds_p, embeds_q))
  embeds = np.vstack((embeds_p, embeds_q))
  if use_pca:
    pca_embeds = compute_pca_embeddings(embeds)
  else:
    pca_embeds = embeds
  if n_clusters is None:
    # Heuristics of the paper to automatically set the number of clusters.
    n_clusters = min(_MAX_N_CLUSTERS, int(len(embeds) * 0.1))
    n_clusters = max(_MIN_N_CLUSTERS, n_clusters)
  kmeans_assignments = get_kmeans_assignments(pca_embeds, n_clusters=n_clusters)

  n_sentences_p = embeds_p.shape[0]
  assignments_p = kmeans_assignments[:n_sentences_p]
  assignments_q = kmeans_assignments[n_sentences_p:]

  p = get_discretized_distribution(assignments_p, n_clusters=n_clusters)
  q = get_discretized_distribution(assignments_q, n_clusters=n_clusters)

  mauve_curve = get_mauve_curve(p, q)
  return MauveOutputs(
      auc=float(auc(mauve_curve[:, 0], mauve_curve[:, 1])),
      mauve_curve=mauve_curve,
      pca_embeds=pca_embeds,
      n_clusters=n_clusters,
      kmeans_assignments=kmeans_assignments,
      discretized_p=p,
      discretized_q=q,
  )


def compute_mauve_and_serialize_result(
    real_bytes: bytes,
    real_dim: int,
    synth_bytes: bytes,
    synth_dim: int,
) -> bytes:
  """Computes MAUVE score and returns a serialized MauveScoreResult proto.

  This function receives raw float32 bytes from C++, constructs numpy arrays
  in Python, performs the full MAUVE computation, and builds the result proto
  entirely in Python. This minimizes the C++/Python boundary.

  Args:
    real_bytes: Raw float32 bytes for real data embeddings.
    real_dim: Embedding dimension for real data.
    synth_bytes: Raw float32 bytes for synthetic data embeddings.
    synth_dim: Embedding dimension for synthetic data.

  Returns:
    Serialized bytes of fcp.confidentialcompute.MauveScoreResult proto.
  """
  embeds_p = np.frombuffer(real_bytes, dtype=np.float32).reshape(-1, real_dim)
  embeds_q = np.frombuffer(synth_bytes, dtype=np.float32).reshape(-1, synth_dim)
  mauve_out = compute_mauve(embeds_p, embeds_q)

  result = _build_mauve_result(mauve_out)
  return result.SerializeToString()


def _build_mauve_result(
    mauve_out,
) -> mauve_score_config_pb2.MauveScoreResult:
  """Builds a MauveScoreResult proto from a MauveOutput."""
  result = mauve_score_config_pb2.MauveScoreResult()
  result.mauve_auc = mauve_out.auc
  result.num_clusters = mauve_out.n_clusters

  # Cluster counts from the 2x2 contingency table.
  p = mauve_out.discretized_p
  q = mauve_out.discretized_q
  real_and_synth = int(np.sum((p > 0) & (q > 0)))
  real_only = int(np.sum((p > 0) & (q == 0)))
  synth_only = int(np.sum((p == 0) & (q > 0)))
  empty = int(np.sum((p == 0) & (q == 0)))

  result.clusters_with_real_and_synth = real_and_synth
  result.clusters_with_real_only = real_only
  result.clusters_with_synth_only = synth_only
  result.clusters_empty = empty

  # Recall and precision.
  real_data_clusters = real_and_synth + real_only
  synth_data_clusters = real_and_synth + synth_only
  result.recall = (
      real_and_synth / real_data_clusters
      if real_data_clusters > 0
      else float('nan')
  )
  result.precision = (
      real_and_synth / synth_data_clusters
      if synth_data_clusters > 0
      else float('nan')
  )

  # KL coverage/fidelity metrics at each lambda value.
  for lam in _LAMBDA_SOFT_VALUES:
    mixed_q = (1 - lam) * q + lam * p
    kl_coverage = robust_kl(p, mixed_q)

    mixed_p = (1 - lam) * p + lam * q
    kl_fidelity = robust_kl(q, mixed_p)

    kl_metric = result.kl_metrics.add()
    kl_metric.lambda_soft = lam
    kl_metric.kl_coverage_penalty = kl_coverage
    kl_metric.kl_fidelity_penalty = kl_fidelity

  return result
