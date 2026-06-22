"""Shape and structure checks on the three model variants."""

import torch

from src.models import SingleSensorModel, EarlyFusionModel, LateFusionModel


def test_single_sensor_output_shape():
    model = SingleSensorModel(in_dim=16, which="a")
    x = torch.randn(8, 16)
    out = model(x)
    assert out.shape == (8, 2)


def test_early_fusion_output_shape():
    model = EarlyFusionModel(dim_a=16, dim_b=10)
    a = torch.randn(5, 16)
    b = torch.randn(5, 10)
    out = model(a, b)
    assert out.shape == (5, 2)


def test_late_fusion_both_modalities_shape():
    model = LateFusionModel(dim_a=16, dim_b=12, emb_dim=8)
    a = torch.randn(4, 16)
    b = torch.randn(4, 12)
    out = model(a, b)
    assert out.shape == (4, 2)


def test_late_fusion_missing_modality_runs():
    # The model must still produce a (batch, num_classes) output when one
    # stream is None, using the learned stand in embedding.
    model = LateFusionModel(dim_a=16, dim_b=12, emb_dim=8)
    a = torch.randn(3, 16)
    b = torch.randn(3, 12)

    out_drop_b = model(a, None)
    out_drop_a = model(None, b)
    assert out_drop_b.shape == (3, 2)
    assert out_drop_a.shape == (3, 2)


def test_late_fusion_missing_embedding_matches_parameter():
    # When a stream is dropped, its embedding must equal the learned missing
    # parameter broadcast over the batch.
    model = LateFusionModel(dim_a=6, dim_b=6, emb_dim=4)
    b = torch.randn(2, 6)
    emb_a, _ = model.encode(None, b)
    expected = model.missing_a.unsqueeze(0).expand(2, -1)
    assert torch.allclose(emb_a, expected)


def test_late_fusion_requires_at_least_one_modality():
    import pytest

    model = LateFusionModel(dim_a=6, dim_b=6, emb_dim=4)
    with pytest.raises(ValueError):
        model(None, None)


def test_single_sensor_rejects_bad_which():
    import pytest

    with pytest.raises(ValueError):
        SingleSensorModel(in_dim=4, which="c")
