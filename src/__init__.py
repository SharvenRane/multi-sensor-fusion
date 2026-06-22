"""Multi-sensor fusion package."""

from .data import make_fusion_dataset, FusionDataset
from .models import (
    SingleSensorModel,
    EarlyFusionModel,
    LateFusionModel,
)
from .train import (
    train_model,
    evaluate_accuracy,
    single_sensor_forward,
    early_fusion_forward,
    late_fusion_forward,
)

__all__ = [
    "make_fusion_dataset",
    "FusionDataset",
    "SingleSensorModel",
    "EarlyFusionModel",
    "LateFusionModel",
    "train_model",
    "evaluate_accuracy",
    "single_sensor_forward",
    "early_fusion_forward",
    "late_fusion_forward",
]
