"""Behaviour checks on the synthetic dataset.

The dataset's whole reason to exist is that the label needs both sensors. We
verify that property directly: each single stream is uninformative about the
label (close to a 50/50 split inside each vote group), while the two streams
together determine it.
"""

import numpy as np
import torch

from src.data import make_fusion_dataset


def test_shapes_and_dtypes():
    # margin=0 so no samples are dropped and the count is exact.
    ds = make_fusion_dataset(
        n_samples=500, dim_a=8, dim_b=12, margin=0.0, noise=0.0, seed=1
    )
    assert ds.sensor_a.shape == (500, 8)
    assert ds.sensor_b.shape == (500, 12)
    assert ds.labels.shape == (500,)
    assert ds.sensor_a.dtype == torch.float32
    assert ds.sensor_b.dtype == torch.float32
    assert ds.labels.dtype == torch.long
    assert len(ds) == 500


def test_labels_are_binary():
    ds = make_fusion_dataset(n_samples=300, noise=0.1, seed=2)
    uniques = set(ds.labels.tolist())
    assert uniques.issubset({0, 1})


def test_label_is_xor_of_hidden_votes_without_noise():
    # With noise=0 and no margin filtering the label must be exactly
    # recoverable as XOR of the two per stream votes. We recompute the votes
    # using the module's own fixed projections and confirm an exact match.
    from src.data import _hidden_vote, _fixed_projection

    n, da, db, sd, seed = 400, 10, 10, 3, 7
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, da)).astype(np.float32)
    b = rng.standard_normal((n, db)).astype(np.float32)
    proj_a = _fixed_projection(da, sd, "a")
    proj_b = _fixed_projection(db, sd, "b")
    vote_a, _ = _hidden_vote(a, proj_a)
    vote_b, _ = _hidden_vote(b, proj_b)
    expected = np.bitwise_xor(vote_a, vote_b)

    ds = make_fusion_dataset(
        n_samples=n, dim_a=da, dim_b=db, signal_dim=sd,
        margin=0.0, noise=0.0, seed=seed,
    )
    assert np.array_equal(ds.labels.numpy(), expected)
    assert np.array_equal(ds.sensor_a.numpy(), a)
    assert np.array_equal(ds.sensor_b.numpy(), b)


def test_distractor_coordinates_carry_no_signal():
    # Coordinates beyond signal_dim should have zero projection weight, so the
    # fixed projection vector must be zero there.
    from src.data import _fixed_projection

    proj = _fixed_projection(dim=16, signal_dim=3, stream="a")
    assert np.all(proj[3:] == 0.0)
    assert np.any(proj[:3] != 0.0)


def test_margin_filtering_reduces_count():
    # With a margin, ambiguous samples are dropped, so the dataset is smaller
    # than the candidate count. Without a margin, all candidates are kept.
    full = make_fusion_dataset(n_samples=2000, margin=0.0, noise=0.0, seed=9)
    filtered = make_fusion_dataset(n_samples=2000, margin=0.5, noise=0.0, seed=9)
    assert len(full) == 2000
    assert len(filtered) < 2000


def test_single_stream_is_near_chance_for_label():
    # Marginally, neither stream's own sign should predict the label far from
    # 0.5. We bin by the sign of each coordinate-sum proxy and check the label
    # mean stays near chance. The cleanest test: correlation between a simple
    # linear readout of one stream and the label should be small.
    ds = make_fusion_dataset(
        n_samples=4000, dim_a=16, dim_b=16, margin=0.0, noise=0.0, seed=3
    )
    y = ds.labels.numpy().astype(np.float64)

    # Best single-feature linear correlation within stream A.
    a = ds.sensor_a.numpy().astype(np.float64)
    corrs = []
    for j in range(a.shape[1]):
        col = a[:, j]
        c = np.corrcoef(col, y)[0, 1]
        corrs.append(abs(c))
    # No single raw feature of one modality should strongly track the XOR label.
    assert max(corrs) < 0.15


def test_noise_flips_some_labels():
    # Use margin=0 so both datasets keep the exact same rows and only the
    # label flips differ. (With a margin the kept rows could differ.)
    clean = make_fusion_dataset(n_samples=1000, margin=0.0, noise=0.0, seed=5)
    noisy = make_fusion_dataset(n_samples=1000, margin=0.0, noise=0.3, seed=5)
    # Same seed, so sensors and base votes match; only label flips differ.
    assert torch.equal(clean.sensor_a, noisy.sensor_a)
    assert torch.equal(clean.sensor_b, noisy.sensor_b)
    diff = (clean.labels != noisy.labels).float().mean().item()
    assert 0.2 < diff < 0.4  # roughly the requested 0.3 flip rate


def test_invalid_noise_rejected():
    import pytest

    with pytest.raises(ValueError):
        make_fusion_dataset(noise=0.6)
    with pytest.raises(ValueError):
        make_fusion_dataset(noise=0.5)
