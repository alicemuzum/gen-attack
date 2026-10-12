"""Projected Gradient Descent (Madry et al., 2018)."""

from __future__ import annotations

import torch

from gen_attack.attacks.base import Attack, LossFn
from gen_attack.attacks.registry import register_attack


@register_attack("pgd")
class PGD(Attack):
    """Iterated FGSM, projected back onto the epsilon-ball after every step.

    ``x_{t+1} = Proj_{B(x, eps)}(x_t + alpha * sign(grad_x L(x_t)))``

    Reference: https://arxiv.org/abs/1706.06083

    Args:
        epsilon: Radius of the L-infinity ball around the clean image.
        steps: Number of gradient steps.
        alpha: Step size. Defaults to ``2.5 * epsilon / steps``, as in Madry et al.
        random_start: Start from a uniformly random point inside the ball.
        seed: Optional seed for the random start, for reproducible results.
    """

    def __init__(
        self,
        epsilon: float,
        steps: int = 10,
        alpha: float | None = None,
        random_start: bool = True,
        seed: int | None = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        super().__init__(epsilon, clip_min, clip_max)
        if steps < 1:
            raise ValueError(f"steps must be at least 1, got {steps}")
        self.steps = steps
        self.alpha = alpha if alpha is not None else 2.5 * epsilon / steps
        self.random_start = random_start
        self.seed = seed

    def perturb(self, images: torch.Tensor, loss_fn: LossFn) -> torch.Tensor:
        clean = images.detach()
        if self.epsilon == 0:
            return clean.clone()

        adversarial = clean.clone()
        if self.random_start:
            generator = None
            if self.seed is not None:
                generator = torch.Generator(device=clean.device).manual_seed(self.seed)
            noise = torch.rand(clean.shape, generator=generator, device=clean.device)
            adversarial = adversarial + (2 * noise - 1) * self.epsilon
            adversarial = adversarial.clamp(self.clip_min, self.clip_max)

        for _ in range(self.steps):
            grad = self._loss_gradient(adversarial, loss_fn)
            adversarial = adversarial + self.alpha * grad.sign()
            adversarial = clean + (adversarial - clean).clamp(-self.epsilon, self.epsilon)
            adversarial = adversarial.clamp(self.clip_min, self.clip_max)

        return adversarial

    def __repr__(self) -> str:
        return f"PGD(epsilon={self.epsilon}, steps={self.steps}, alpha={self.alpha:.2e})"
