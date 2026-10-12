"""Open-vocabulary detection with Grounding DINO, used to label generated scenes."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from PIL import Image

from gen_attack.models.detr import Detections
from gen_attack.utils import resolve_device


class GroundingDinoDetector:
    """Detects objects described by free-text queries, e.g. ``["bench", "crosswalk"]``."""

    def __init__(self, model_name: str = "IDEA-Research/grounding-dino-tiny", device: str | None = None) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = resolve_device(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).eval().to(self.device)

    @torch.no_grad()
    def detect(
        self,
        image: Image.Image,
        queries: Sequence[str],
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
    ) -> Detections:
        # Grounding DINO expects lower-case queries, each terminated by a period.
        text = " ".join(f"{query.strip().lower()}." for query in queries)
        inputs = self.processor(images=image.convert("RGB"), text=text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # Thresholds are positional because transformers renamed `box_threshold` to `threshold` in 4.51.
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold,
            text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        labels = result.get("text_labels", result["labels"])
        return Detections(result["scores"].cpu(), list(labels), result["boxes"].cpu())
