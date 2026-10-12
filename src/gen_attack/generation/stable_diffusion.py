"""Text-to-image generation of test scenes with Stable Diffusion."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from gen_attack.utils import resolve_device


class StableDiffusionGenerator:
    """Thin wrapper around a ``diffusers`` Stable Diffusion pipeline.

    Requires the ``generation`` extra (``poetry install --extras generation``).
    Half precision is used on CUDA; CPU works but is slow.
    """

    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5", device: str | None = None) -> None:
        from diffusers import DiffusionPipeline

        self.device = resolve_device(device)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.pipeline = DiffusionPipeline.from_pretrained(model_name, torch_dtype=dtype, use_safetensors=True)
        self.pipeline.to(self.device)
        self.pipeline.enable_attention_slicing()

    def generate(self, prompt: str, seed: int | None = None) -> Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        return self.pipeline(prompt, generator=generator).images[0]

    def generate_to_file(self, prompt: str, path: str | Path, seed: int | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.generate(prompt, seed).save(path)
        return path
