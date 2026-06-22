"""Train every variant once and print a comparison table.

Run with the project's Python:

    python demo.py

The numbers move a little run to run because training is stochastic, but the
ordering is stable: the two fusion models clear 0.9 held out accuracy while
either single sensor sits near 0.5, and late fusion collapses back toward
chance when a modality is taken away at inference.
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


def main() -> None:
    torch.manual_seed(0)

    train = make_fusion_dataset(n_samples=6000, seed=100)
    test = make_fusion_dataset(n_samples=3000, noise=0.0, seed=200)
    print(f"train rows: {len(train)}    test rows: {len(test)}\n")

    model_a = SingleSensorModel(in_dim=16, which="a")
    train_model(model_a, train, single_sensor_forward("a"), epochs=25)
    acc_a = evaluate_accuracy(model_a, test, single_sensor_forward("a"))

    model_b = SingleSensorModel(in_dim=16, which="b")
    train_model(model_b, train, single_sensor_forward("b"), epochs=25)
    acc_b = evaluate_accuracy(model_b, test, single_sensor_forward("b"))

    early = EarlyFusionModel(dim_a=16, dim_b=16)
    train_model(early, train, early_fusion_forward, epochs=25)
    acc_early = evaluate_accuracy(early, test, early_fusion_forward)

    late = LateFusionModel(dim_a=16, dim_b=16)
    train_model(late, train, late_fusion_forward(drop=None), epochs=25)
    acc_late = evaluate_accuracy(late, test, late_fusion_forward(drop=None))
    acc_late_no_b = evaluate_accuracy(late, test, late_fusion_forward(drop="b"))
    acc_late_no_a = evaluate_accuracy(late, test, late_fusion_forward(drop="a"))

    rows = [
        ("sensor A only", acc_a),
        ("sensor B only", acc_b),
        ("early fusion", acc_early),
        ("late fusion (both)", acc_late),
        ("late fusion (B missing)", acc_late_no_b),
        ("late fusion (A missing)", acc_late_no_a),
    ]
    for name, acc in rows:
        print(f"{name:28s} {acc:.3f}")


if __name__ == "__main__":
    main()
