"""Single sensor and fusion models.

Three model families share the same encoder building block so the comparison
is fair. Each per stream encoder is a small multilayer perceptron that maps a
sensor vector to an embedding.

  SingleSensorModel: encodes one stream and classifies from it alone.
  EarlyFusionModel:  concatenates the two raw streams, then encodes the joint
                     vector and classifies. The interaction is available from
                     the very first layer.
  LateFusionModel:   encodes each stream independently, concatenates the two
                     embeddings, then classifies. Each branch can also run on
                     its own, which is what lets late fusion degrade gracefully
                     when one modality is missing.

All three output raw logits over two classes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    """A two layer perceptron with a ReLU nonlinearity in the middle."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class SingleSensorModel(nn.Module):
    """Classifier that sees exactly one sensor stream.

    Args:
        in_dim: width of the sensor vector it consumes.
        hidden: hidden width of the encoder.
        num_classes: number of output classes.
        which: "a" or "b", recorded so a caller knows which stream to feed.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        num_classes: int = 2,
        which: str = "a",
    ):
        super().__init__()
        if which not in ("a", "b"):
            raise ValueError("which must be 'a' or 'b'")
        self.which = which
        self.encoder = _mlp(in_dim, hidden, hidden)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.encoder(x)))


class EarlyFusionModel(nn.Module):
    """Fusion at the input: concatenate raw streams before encoding.

    Args:
        dim_a: width of sensor A.
        dim_b: width of sensor B.
        hidden: hidden width of the joint encoder.
        num_classes: number of output classes.
    """

    def __init__(
        self,
        dim_a: int,
        dim_b: int,
        hidden: int = 64,
        num_classes: int = 2,
    ):
        super().__init__()
        self.dim_a = dim_a
        self.dim_b = dim_b
        self.encoder = _mlp(dim_a + dim_b, hidden, hidden)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        joint = torch.cat([a, b], dim=-1)
        return self.head(torch.relu(self.encoder(joint)))


class LateFusionModel(nn.Module):
    """Fusion at the embedding: encode each stream, then combine.

    Each modality gets its own encoder. The two embeddings are concatenated
    and passed to a small fusion head. When a stream is absent at inference,
    its embedding is replaced by a learned bias so the head still receives a
    well defined vector. That is what lets the model produce a sensible
    prediction from a single sensor.

    Args:
        dim_a: width of sensor A.
        dim_b: width of sensor B.
        emb_dim: width of each per stream embedding.
        hidden: hidden width inside the fusion head.
        num_classes: number of output classes.
    """

    def __init__(
        self,
        dim_a: int,
        dim_b: int,
        emb_dim: int = 32,
        hidden: int = 64,
        num_classes: int = 2,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.encoder_a = _mlp(dim_a, hidden, emb_dim)
        self.encoder_b = _mlp(dim_b, hidden, emb_dim)
        # Learned stand in embeddings used when a modality is missing.
        self.missing_a = nn.Parameter(torch.zeros(emb_dim))
        self.missing_b = nn.Parameter(torch.zeros(emb_dim))
        self.fusion = _mlp(emb_dim * 2, hidden, num_classes)

    def encode(self, a: torch.Tensor | None, b: torch.Tensor | None):
        """Encode whichever streams are present.

        Either a or b may be None to signal a missing modality. The batch size
        is inferred from whichever stream is present.
        """
        if a is None and b is None:
            raise ValueError("at least one modality must be provided")

        if a is not None:
            emb_a = torch.relu(self.encoder_a(a))
            batch = emb_a.shape[0]
        else:
            batch = b.shape[0]
            emb_a = self.missing_a.unsqueeze(0).expand(batch, -1)

        if b is not None:
            emb_b = torch.relu(self.encoder_b(b))
        else:
            emb_b = self.missing_b.unsqueeze(0).expand(batch, -1)

        return emb_a, emb_b

    def forward(
        self,
        a: torch.Tensor | None,
        b: torch.Tensor | None,
    ) -> torch.Tensor:
        emb_a, emb_b = self.encode(a, b)
        joint = torch.cat([emb_a, emb_b], dim=-1)
        return self.fusion(joint)
