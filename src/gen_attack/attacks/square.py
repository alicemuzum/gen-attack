"""Square Attack (Andriushchenko et al., 2020): a score-based black-box attack."""

from __future__ import annotations

import math
from typing import Callable

import torch

from gen_attack.attacks.base import Attack, LossFn
from gen_attack.attacks.registry import register_attack

QueryCallback = Callable[[int, torch.Tensor], None]
"""Called as ``callback(queries_used, adversarial_images)`` after every query."""

# Iterations (on a 10,000-query scale) at which the square size is halved, from the reference implementation.
_SCHEDULE = (10, 50, 200, 500, 1000, 2000, 4000, 6000, 8000)


@register_attack("square")
class SquareAttack(Attack):
    """Random search over square-shaped perturbations, using only loss values.

    Unlike FGSM and PGD, this attack never computes a gradient. It only needs to
    *query* the model, so it also works against models whose internals are hidden
    behind an API. Each query changes the perturbation inside one random square
    to new values of ``±epsilon`` per channel. The change is kept if it increases
    the loss of that image and discarded otherwise. Squares start large and
    shrink as the query budget is used up.

    Reference: https://arxiv.org/abs/1912.00049

    Args:
        epsilon: Radius of the L-infinity ball around the clean image.
        n_queries: Number of loss evaluations per image, including the initial one.
        p_init: Initial fraction of the image covered by one square.
        seed: Optional seed for reproducible results.
        callback: Optional function called after every query, e.g. to track accuracy.
    """

    def __init__(
        self,
        epsilon: float,
        n_queries: int = 1000,
        p_init: float = 0.05,
        seed: int | None = None,
        callback: QueryCallback | None = None,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        super().__init__(epsilon, clip_min, clip_max)
        if n_queries < 1:
            raise ValueError(f"n_queries must be at least 1, got {n_queries}")
        if not 0 < p_init <= 1:
            raise ValueError(f"p_init must be in (0, 1], got {p_init}")
        self.n_queries = n_queries
        self.p_init = p_init
        self.seed = seed
        self.callback = callback

    @torch.no_grad()
    def perturb(self, images: torch.Tensor, loss_fn: LossFn) -> torch.Tensor:
        clean = images.detach()
        if self.epsilon == 0:
            return clean.clone()

        batch, channels, height, width = clean.shape
        generator = torch.Generator(device=clean.device)
        if self.seed is not None:
            generator.manual_seed(self.seed)
        else:
            generator.seed()

        # Start from vertical stripes of ±epsilon, which the paper found to be a strong initialisation.
        delta = self._random_signs((batch, channels, 1, width), generator, clean).expand_as(clean).contiguous()
        best_loss = self._query(clean, delta, loss_fn)
        self._notify(1, clean, delta)

        rows = torch.arange(height, device=clean.device)
        cols = torch.arange(width, device=clean.device)
        for query in range(2, self.n_queries + 1):
            size = self._square_size(query, height, width)
            top = torch.randint(0, height - size + 1, (batch, 1), generator=generator, device=clean.device)
            left = torch.randint(0, width - size + 1, (batch, 1), generator=generator, device=clean.device)
            in_rows = (rows >= top) & (rows < top + size)
            in_cols = (cols >= left) & (cols < left + size)
            square = in_rows[:, None, :, None] & in_cols[:, None, None, :]

            new_values = self._random_signs((batch, channels, 1, 1), generator, clean)
            candidate = torch.where(square, new_values, delta)
            loss = self._query(clean, candidate, loss_fn)

            improved = loss > best_loss
            delta = torch.where(improved.view(-1, 1, 1, 1), candidate, delta)
            best_loss = torch.where(improved, loss, best_loss)
            self._notify(query, clean, delta)

        return self._apply(clean, delta)

    def __repr__(self) -> str:
        return f"SquareAttack(epsilon={self.epsilon}, n_queries={self.n_queries})"

    def _square_size(self, query: int, height: int, width: int) -> int:
        """Side length of the square: shrinks as the query budget is used up."""
        progress = int(query / self.n_queries * 10_000)
        fraction = self.p_init / 2 ** sum(progress > step for step in _SCHEDULE)
        side = round(math.sqrt(fraction * height * width))
        return max(1, min(side, height - 1, width - 1))

    def _random_signs(self, shape: tuple[int, ...], generator: torch.Generator, like: torch.Tensor) -> torch.Tensor:
        signs = torch.randint(0, 2, shape, generator=generator, device=like.device).to(like.dtype) * 2 - 1
        return self.epsilon * signs

    def _apply(self, clean: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return (clean + delta).clamp(self.clip_min, self.clip_max)

    def _query(self, clean: torch.Tensor, delta: torch.Tensor, loss_fn: LossFn) -> torch.Tensor:
        loss = loss_fn(self._apply(clean, delta))
        if loss.numel() != len(clean):
            raise ValueError(
                "Square Attack needs one loss value per image; got a loss of shape "
                f"{tuple(loss.shape)} for a batch of {len(clean)}. Use reduction='none'."
            )
        return loss.reshape(len(clean))

    def _notify(self, query: int, clean: torch.Tensor, delta: torch.Tensor) -> None:
        if self.callback is not None:
            self.callback(query, self._apply(clean, delta))
