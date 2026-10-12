"""Generative and open-vocabulary models for building test scenes.

The heavy dependencies are imported lazily, so importing this package is cheap.
"""

from gen_attack.generation.grounding_dino import GroundingDinoDetector
from gen_attack.generation.stable_diffusion import StableDiffusionGenerator

__all__ = ["GroundingDinoDetector", "StableDiffusionGenerator"]
