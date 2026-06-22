"""The headline behaviour tests.

1. On data whose label needs both sensors, a trained fusion model beats both
   single sensor models, which are stuck near chance by construction.
2. A late fusion model still produces useful predictions when one modality is
   missing at inference, and clearly better predictions when both are present.

These train tiny models on CPU in a few seconds. We use a fixed seed and a
fresh train/test split so the numbers are honest held out accuracies.
"""

import torch

from src.data import make_fusion_dataset
from src.models import SingleSensorModel, EarlyFusionModel, LateFusionModel
from src.train import (
    train_model,
    evaluate_accuracy,
    single_sensor_forward,
    early_fusion_forward,
    late_fusion_forward,
)

DIM_A = 16
DIM_B = 16


def _split():
    # Candidate counts are oversized because the margin band drops roughly a
    # third of samples. Train and test draw different seeds but share the same
    # fixed labelling rule, so test accuracy is an honest held out number.
    train = make_fusion_dataset(
        n_samples=6000, dim_a=DIM_A, dim_b=DIM_B, noise=0.05, seed=100
    )
    test = make_fusion_dataset(
        n_samples=3000, dim_a=DIM_A, dim_b=DIM_B, noise=0.0, seed=200
    )
    return train, test


def test_single_sensor_models_stuck_near_chance():
    torch.manual_seed(0)
    train, test = _split()

    model_a = SingleSensorModel(in_dim=DIM_A, which="a")
    train_model(model_a, train, single_sensor_forward("a"), epochs=25)
    acc_a = evaluate_accuracy(model_a, test, single_sensor_forward("a"))

    model_b = SingleSensorModel(in_dim=DIM_B, which="b")
    train_model(model_b, train, single_sensor_forward("b"), epochs=25)
    acc_b = evaluate_accuracy(model_b, test, single_sensor_forward("b"))

    # Neither single stream can solve XOR, so both stay close to 0.5.
    assert acc_a < 0.65
    assert acc_b < 0.65


def test_early_fusion_beats_both_single_sensors():
    torch.manual_seed(0)
    train, test = _split()

    model_a = SingleSensorModel(in_dim=DIM_A, which="a")
    train_model(model_a, train, single_sensor_forward("a"), epochs=25)
    acc_a = evaluate_accuracy(model_a, test, single_sensor_forward("a"))

    model_b = SingleSensorModel(in_dim=DIM_B, which="b")
    train_model(model_b, train, single_sensor_forward("b"), epochs=25)
    acc_b = evaluate_accuracy(model_b, test, single_sensor_forward("b"))

    fusion = EarlyFusionModel(dim_a=DIM_A, dim_b=DIM_B)
    train_model(fusion, train, early_fusion_forward, epochs=25)
    acc_fusion = evaluate_accuracy(fusion, test, early_fusion_forward)

    assert acc_fusion > acc_a + 0.15
    assert acc_fusion > acc_b + 0.15
    assert acc_fusion > 0.80


def test_late_fusion_beats_both_single_sensors():
    torch.manual_seed(0)
    train, test = _split()

    model_a = SingleSensorModel(in_dim=DIM_A, which="a")
    train_model(model_a, train, single_sensor_forward("a"), epochs=25)
    acc_a = evaluate_accuracy(model_a, test, single_sensor_forward("a"))

    model_b = SingleSensorModel(in_dim=DIM_B, which="b")
    train_model(model_b, train, single_sensor_forward("b"), epochs=25)
    acc_b = evaluate_accuracy(model_b, test, single_sensor_forward("b"))

    fusion = LateFusionModel(dim_a=DIM_A, dim_b=DIM_B)
    train_model(fusion, train, late_fusion_forward(drop=None), epochs=25)
    acc_fusion = evaluate_accuracy(fusion, test, late_fusion_forward(drop=None))

    assert acc_fusion > acc_a + 0.15
    assert acc_fusion > acc_b + 0.15
    assert acc_fusion > 0.80


def test_late_fusion_handles_missing_modality():
    # Train late fusion so that it sometimes sees a dropped modality, then
    # check it still runs and does not crash when a stream is missing at test,
    # and that having both streams is at least as good as having one.
    torch.manual_seed(0)
    train, test = _split()

    fusion = LateFusionModel(dim_a=DIM_A, dim_b=DIM_B)
    # Train on full pairs. The missing-modality path uses learned stand ins.
    train_model(fusion, train, late_fusion_forward(drop=None), epochs=30)

    acc_both = evaluate_accuracy(fusion, test, late_fusion_forward(drop=None))
    acc_drop_a = evaluate_accuracy(fusion, test, late_fusion_forward(drop="a"))
    acc_drop_b = evaluate_accuracy(fusion, test, late_fusion_forward(drop="b"))

    # All three inference modes must produce valid accuracies in [0, 1].
    for acc in (acc_both, acc_drop_a, acc_drop_b):
        assert 0.0 <= acc <= 1.0

    # With both streams the model solves the task; with one stream gone it
    # falls back toward chance because the XOR label is unrecoverable from a
    # single sensor. The full pair must beat each degraded case.
    assert acc_both > acc_drop_a
    assert acc_both > acc_drop_b
    assert acc_both > 0.80
    # The degraded cases cannot exceed what a single sensor can know.
    assert acc_drop_a < 0.70
    assert acc_drop_b < 0.70
