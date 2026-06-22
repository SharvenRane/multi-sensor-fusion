"""Training and evaluation helpers shared by every model variant.

The training loop is deliberately generic. Each model has a different forward
signature (one stream, two raw streams, or two streams with optional drop), so
the caller passes a small ``forward_fn`` that knows how to call its model on a
batch. Everything else (optimiser, loss, epochs, accuracy) is shared so the
comparison across models is controlled.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import FusionDataset

# A batch is (sensor_a, sensor_b, labels). A forward_fn turns a model plus a
# batch into class logits of shape (batch, num_classes).
Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
ForwardFn = Callable[[nn.Module, Batch], torch.Tensor]


def train_model(
    model: nn.Module,
    dataset: FusionDataset,
    forward_fn: ForwardFn,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-2,
    seed: int = 0,
) -> nn.Module:
    """Train ``model`` in place and return it.

    Args:
        model: the network to train.
        dataset: a FusionDataset of training samples.
        forward_fn: maps (model, batch) to logits, hiding each model's forward
            signature from the loop.
        epochs: number of passes over the data.
        batch_size: minibatch size.
        lr: Adam learning rate.
        seed: seed for the DataLoader shuffle, for reproducibility.
    """
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for batch in loader:
            optimiser.zero_grad()
            logits = forward_fn(model, batch)
            labels = batch[2]
            loss = loss_fn(logits, labels)
            loss.backward()
            optimiser.step()
    return model


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    dataset: FusionDataset,
    forward_fn: ForwardFn,
    batch_size: int = 256,
) -> float:
    """Return classification accuracy of ``model`` on ``dataset`` in [0, 1]."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        logits = forward_fn(model, batch)
        preds = logits.argmax(dim=-1)
        labels = batch[2]
        correct += int((preds == labels).sum().item())
        total += labels.shape[0]
    return correct / total


# --- forward_fn factories for each model variant ------------------------------


def single_sensor_forward(which: str) -> ForwardFn:
    """forward_fn for a SingleSensorModel reading stream 'a' or 'b'."""
    index = 0 if which == "a" else 1

    def fn(model: nn.Module, batch: Batch) -> torch.Tensor:
        return model(batch[index])

    return fn


def early_fusion_forward(model: nn.Module, batch: Batch) -> torch.Tensor:
    """forward_fn for EarlyFusionModel: feed both raw streams."""
    return model(batch[0], batch[1])


def late_fusion_forward(
    drop: str | None = None,
) -> ForwardFn:
    """forward_fn for LateFusionModel.

    Args:
        drop: if "a" or "b", that modality is replaced by None to simulate a
            missing sensor. If None, both modalities are passed.
    """

    def fn(model: nn.Module, batch: Batch) -> torch.Tensor:
        a = None if drop == "a" else batch[0]
        b = None if drop == "b" else batch[1]
        return model(a, b)

    return fn
