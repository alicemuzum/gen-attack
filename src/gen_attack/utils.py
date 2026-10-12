from __future__ import annotations

import numpy as np
import torch


def resolve_device(device: str | None = None) -> torch.device:
    """Use ``device`` if given, otherwise CUDA when available, otherwise CPU."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    """Seed Python-facing RNGs used across the project for reproducible runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_hwc_numpy(image: torch.Tensor) -> np.ndarray:
    """Convert a ``[C, H, W]`` (or ``[1, C, H, W]``) tensor in [0, 1] to an ``[H, W, C]`` array."""
    if image.dim() == 4:
        image = image.squeeze(0)
    return image.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
