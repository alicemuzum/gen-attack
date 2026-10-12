"""Common interface for adversarial attacks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch

LossFn = Callable[[torch.Tensor], torch.Tensor]
"""Maps a batch of ``B`` images in [0, 1] to per-example losses of shape ``[B]``, which the attack maximises.

A scalar is also accepted when the batch holds a single image.
"""


class Attack(ABC):
    """An L-infinity bounded, untargeted evasion attack.

    Attacks are model-agnostic: instead of a model they receive a loss function
    that maps images to per-example losses. This lets the same attack run against an
    image classifier (cross-entropy) and an object detector (DETR's set loss).

    Args:
        epsilon: Maximum per-pixel perturbation, in [0, 1] pixel units.
        clip_min: Lower bound of the valid pixel range.
        clip_max: Upper bound of the valid pixel range.
    """

    name: str = ""

    def __init__(self, epsilon: float, clip_min: float = 0.0, clip_max: float = 1.0) -> None:
        if epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {epsilon}")
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max

    @abstractmethod
    def perturb(self, images: torch.Tensor, loss_fn: LossFn) -> torch.Tensor:
        """Return adversarial versions of ``images``.

        The input tensor is not modified and the result is detached from the graph.
        """

    def __call__(self, images: torch.Tensor, loss_fn: LossFn) -> torch.Tensor:
        return self.perturb(images, loss_fn)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(epsilon={self.epsilon})"

    @staticmethod
    def _loss_gradient(images: torch.Tensor, loss_fn: LossFn) -> torch.Tensor:
        """Gradient of ``loss_fn`` with respect to ``images``."""
        images = images.detach().clone().requires_grad_(True)
        loss = loss_fn(images).sum()
        (grad,) = torch.autograd.grad(loss, images)
        return grad
