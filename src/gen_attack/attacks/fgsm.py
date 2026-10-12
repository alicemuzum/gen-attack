"""Fast Gradient Sign Method (Goodfellow et al., 2015)."""

from __future__ import annotations

import torch

from gen_attack.attacks.base import Attack, LossFn
from gen_attack.attacks.registry import register_attack


@register_attack("fgsm")
class FGSM(Attack):
    """Single-step attack: ``x_adv = clip(x + epsilon * sign(grad_x L(x)))``.

    Reference: https://arxiv.org/abs/1412.6572
    """

    def perturb(self, images: torch.Tensor, loss_fn: LossFn) -> torch.Tensor:
        if self.epsilon == 0:
            return images.detach().clone()
        grad = self._loss_gradient(images, loss_fn)
        adversarial = images.detach() + self.epsilon * grad.sign()
        return adversarial.clamp(self.clip_min, self.clip_max)
