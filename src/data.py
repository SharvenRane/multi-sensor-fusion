"""Synthetic aligned multi-sensor data.

The point of this module is to manufacture a labelling rule that genuinely
needs two sensors at once. We model two streams:

  sensor A  ("camera features"): a dense feature vector
  sensor B  ("depth or radar vectors"): a second dense feature vector

Each stream on its own carries only a hidden binary "vote". The label is the
XOR of the two votes. XOR is the classic example of an interaction that is
not linearly recoverable from either input alone: if you marginalise out one
modality, the remaining one is exactly 50/50 on the label, so any model that
sees a single stream is stuck at chance no matter how powerful it is. A model
must combine both streams to do better than guessing.

Each stream's vote is itself a non trivial function of that stream's vector
(the sign of a fixed linear projection over an informative subspace), so a
single sensor model still has a real learning problem inside its own modality.
It just cannot see the other half of the answer.

Two design choices keep the per stream vote learnable in a handful of epochs
on CPU, which is what makes the behaviour tests fast and stable:

  - The signal lives in the first ``signal_dim`` coordinates of each stream.
    The remaining coordinates are pure distractor noise. So the vote is a
    real linear readout the encoder has to discover, but the search space is
    small enough to fit quickly.
  - Samples whose projection score falls inside a margin band around the
    decision boundary are dropped. Removing those ambiguous points means the
    learned boundary generalises cleanly instead of memorising edge cases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class FusionDataset(Dataset):
    """Holds aligned sensor A, sensor B tensors and their joint labels."""

    sensor_a: torch.Tensor  # (N, dim_a) float32
    sensor_b: torch.Tensor  # (N, dim_b) float32
    labels: torch.Tensor    # (N,) long, values in {0, 1}

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int):
        return self.sensor_a[idx], self.sensor_b[idx], self.labels[idx]


# Seed for the projection vectors. This is deliberately separate from the
# per call sample seed so the labelling rule is identical across every dataset
# you draw (train, validation, test), while the samples themselves vary.
_PROJECTION_SEED = 12345


def _fixed_projection(dim: int, signal_dim: int, stream: str) -> np.ndarray:
    """Return the constant projection vector for one stream.

    Only the first ``signal_dim`` coordinates carry weight. The rest are zero,
    making the trailing coordinates pure distractors. The vector is fully
    determined by (dim, signal_dim, stream), so it is shared across all
    datasets regardless of the per call sample seed.
    """
    offset = 0 if stream == "a" else 1
    rng = np.random.default_rng(_PROJECTION_SEED + offset)
    proj = np.zeros(dim, dtype=np.float32)
    k = min(signal_dim, dim)
    proj[:k] = rng.standard_normal(k).astype(np.float32)
    return proj


def _hidden_vote(vectors: np.ndarray, projection: np.ndarray):
    """Per row vote and standardised score.

    Returns a tuple (vote, score) where vote is in {0, 1} (sign of the score)
    and score is the projection standardised to unit variance. The score is
    used to apply the margin band that drops ambiguous samples.
    """
    score = vectors @ projection
    std = score.std()
    if std > 0:
        score = score / std
    vote = (score > 0.0).astype(np.int64)
    return vote, score


def make_fusion_dataset(
    n_samples: int = 2000,
    dim_a: int = 16,
    dim_b: int = 16,
    signal_dim: int = 3,
    margin: float = 0.3,
    noise: float = 0.05,
    seed: int = 0,
):
    """Build a FusionDataset whose label is XOR of two per stream votes.

    The returned dataset may contain fewer than ``n_samples`` rows because
    samples that land inside the margin band of either stream are dropped.
    Oversample ``n_samples`` if you need a guaranteed minimum count.

    Args:
        n_samples: number of candidate samples drawn before margin filtering.
        dim_a: feature width of sensor A.
        dim_b: feature width of sensor B.
        signal_dim: number of leading coordinates that carry the vote signal.
            The rest of each stream is distractor noise.
        margin: half width of the dead band around each stream's decision
            boundary. Samples inside the band are removed so the boundary
            generalises cleanly. Set to 0 to keep every sample.
        noise: probability of flipping the final label, simulating sensor
            imperfection. Keep below 0.5 so the signal stays learnable.
        seed: RNG seed for the sampled vectors and label flips.

    Returns:
        FusionDataset with float32 sensors and long labels.
    """
    if not 0.0 <= noise < 0.5:
        raise ValueError("noise must be in [0, 0.5)")
    if margin < 0.0:
        raise ValueError("margin must be non negative")
    if signal_dim < 1:
        raise ValueError("signal_dim must be at least 1")

    rng = np.random.default_rng(seed)

    a = rng.standard_normal((n_samples, dim_a)).astype(np.float32)
    b = rng.standard_normal((n_samples, dim_b)).astype(np.float32)

    proj_a = _fixed_projection(dim_a, signal_dim, "a")
    proj_b = _fixed_projection(dim_b, signal_dim, "b")

    vote_a, score_a = _hidden_vote(a, proj_a)
    vote_b, score_b = _hidden_vote(b, proj_b)

    label = np.bitwise_xor(vote_a, vote_b)

    if noise > 0.0:
        flip = rng.random(n_samples) < noise
        label = np.where(flip, 1 - label, label)

    if margin > 0.0:
        keep = (np.abs(score_a) > margin) & (np.abs(score_b) > margin)
        a, b, label = a[keep], b[keep], label[keep]

    return FusionDataset(
        sensor_a=torch.from_numpy(a),
        sensor_b=torch.from_numpy(b),
        labels=torch.from_numpy(label),
    )
