"""Robustness benchmark for an ImageNet ResNet-18 classifier."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from gen_attack.attacks import Attack, LossFn, get_attack
from gen_attack.utils import resolve_device

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Normalize(nn.Module):
    """Per-channel normalisation as a layer, so attacks can work in [0, 1] pixel space."""

    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


@dataclass
class AttackOutcome:
    """Result of attacking one batch at one epsilon."""

    accuracy: float
    adversarial_images: torch.Tensor
    predictions: torch.Tensor


@dataclass
class RobustnessCurve:
    """Accuracy under attack for a sweep of epsilons."""

    attack: str
    epsilons: list[float]
    accuracies: list[float]


@dataclass
class QueryCurve:
    """Accuracy of a black-box attack as a function of the number of queries per image."""

    attack: str
    epsilon: float
    queries: list[int]
    accuracies: list[float]


class ResNetBenchmark:
    """Measures how the accuracy of a pretrained ResNet-18 degrades under attack.

    The model takes images in [0, 1] and normalises them internally, so all
    perturbations and epsilons are expressed in plain pixel units.
    """

    def __init__(self, device: str | None = None) -> None:
        self.device = resolve_device(device)
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        self.class_names: list[str] = list(weights.meta["categories"])
        self.model = nn.Sequential(
            Normalize(IMAGENET_MEAN, IMAGENET_STD),
            torchvision.models.resnet18(weights=weights),
        )
        self.model.eval().requires_grad_(False).to(self.device)

    def load_samples(self, num_samples: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
        """Load ImageNet validation images bundled with Foolbox (up to 20)."""
        import foolbox as fb

        fmodel = fb.PyTorchModel(self.model, bounds=(0, 1), device=self.device)
        images, labels = fb.utils.samples(fmodel, dataset="imagenet", batchsize=num_samples)
        return images.to(self.device), labels.to(self.device)

    def loss_fn(self, labels: torch.Tensor) -> LossFn:
        return lambda images: F.cross_entropy(self.model(images), labels, reduction="none")

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images).argmax(dim=1)

    def evaluate(self, attack: Attack, images: torch.Tensor, labels: torch.Tensor) -> AttackOutcome:
        """Attack ``images`` and report accuracy on the adversarial versions."""
        adversarial = attack(images, self.loss_fn(labels))
        predictions = self.predict(adversarial)
        accuracy = (predictions == labels).float().mean().item()
        return AttackOutcome(accuracy, adversarial, predictions)

    def robustness_curve(
        self,
        attack_name: str,
        epsilons: Iterable[float],
        images: torch.Tensor,
        labels: torch.Tensor,
        **attack_kwargs: Any,
    ) -> RobustnessCurve:
        """Run ``attack_name`` once per epsilon and collect the accuracies."""
        epsilons = [float(eps) for eps in epsilons]
        accuracies = [
            self.evaluate(get_attack(attack_name, epsilon=eps, **attack_kwargs), images, labels).accuracy
            for eps in epsilons
        ]
        return RobustnessCurve(attack_name, epsilons, accuracies)

    def query_curve(
        self,
        epsilon: float,
        images: torch.Tensor,
        labels: torch.Tensor,
        n_queries: int = 1000,
        num_points: int = 25,
        attack_name: str = "square",
        **attack_kwargs: Any,
    ) -> QueryCurve:
        """Run a query-based attack once and record accuracy at log-spaced query counts.

        The attack must accept a ``callback(queries_used, adversarial_images)`` argument.
        """
        checkpoints = {int(q) for q in np.unique(np.geomspace(1, n_queries, num_points).round())}
        queries: list[int] = []
        accuracies: list[float] = []

        def record(used: int, adversarial: torch.Tensor) -> None:
            if used in checkpoints:
                queries.append(used)
                accuracies.append((self.predict(adversarial) == labels).float().mean().item())

        attack = get_attack(attack_name, epsilon=epsilon, n_queries=n_queries, callback=record, **attack_kwargs)
        attack(images, self.loss_fn(labels))
        return QueryCurve(attack_name, epsilon, queries, accuracies)
