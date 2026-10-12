"""Adversarial attacks on the DETR object detector."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image

from gen_attack.attacks import Attack, LossFn
from gen_attack.utils import resolve_device


@dataclass
class Detections:
    """Detections for one image. ``boxes`` are ``[x_min, y_min, x_max, y_max]`` in pixels."""

    scores: torch.Tensor
    labels: list[str]
    boxes: torch.Tensor

    def __len__(self) -> int:
        return len(self.labels)

    def summary(self) -> str:
        if not self.labels:
            return "no detections"
        return ", ".join(f"{label} ({score:.2f})" for label, score in zip(self.labels, self.scores.tolist()))


@dataclass
class DetectionAttackResult:
    clean_image: torch.Tensor
    adversarial_image: torch.Tensor
    clean_detections: Detections
    adversarial_detections: Detections


class DETRBenchmark:
    """Attacks DETR (ResNet-50 backbone) with any registered attack.

    No ground-truth annotations are needed: the detector's own confident
    predictions on the clean image are used as targets, and the attack
    maximises the DETR set-prediction loss against them. This pushes the model
    to drop, mislabel or misplace the objects it originally found.

    Args:
        model_name: Hugging Face model id.
        threshold: Minimum confidence for a prediction to count as a detection.
        device: Torch device. Defaults to CUDA when available.
    """

    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50",
        threshold: float = 0.9,
        device: str | None = None,
    ) -> None:
        from transformers import DetrForObjectDetection, DetrImageProcessor
        from transformers import logging as hf_logging

        hf_logging.set_verbosity_error()
        self.device = resolve_device(device)
        self.threshold = threshold
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name)
        self.model.eval().requires_grad_(False).to(self.device)
        self._mean = torch.tensor(self.processor.image_mean, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(self.processor.image_std, device=self.device).view(1, 3, 1, 1)

    def prepare(self, image: Image.Image) -> torch.Tensor:
        """Resize ``image`` as DETR expects and return it as a ``[1, 3, H, W]`` tensor in [0, 1].

        Normalisation is left to the forward pass so that attacks operate in pixel space.
        """
        inputs = self.processor(images=image.convert("RGB"), do_normalize=False, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)

    def _forward(self, images: torch.Tensor, labels: list[dict[str, torch.Tensor]] | None = None):
        return self.model(pixel_values=(images - self._mean) / self._std, labels=labels)

    @torch.no_grad()
    def detect(self, images: torch.Tensor) -> Detections:
        outputs = self._forward(images)
        height, width = images.shape[-2:]
        result = self.processor.post_process_object_detection(
            outputs, threshold=self.threshold, target_sizes=[(height, width)]
        )[0]
        labels = [self.model.config.id2label[i] for i in result["labels"].tolist()]
        return Detections(result["scores"].cpu(), labels, result["boxes"].cpu())

    @torch.no_grad()
    def pseudo_labels(self, images: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Turn the model's confident predictions into DETR training targets."""
        outputs = self._forward(images)
        probs = outputs.logits.softmax(-1)[0, :, :-1]  # drop the "no object" class
        scores, classes = probs.max(-1)
        keep = scores > self.threshold
        if not keep.any():
            raise ValueError(
                f"DETR found no objects above threshold {self.threshold}; nothing to attack. "
                "Try a lower threshold or a different image."
            )
        return [{"class_labels": classes[keep], "boxes": outputs.pred_boxes[0, keep]}]

    def loss_fn(self, labels: list[dict[str, torch.Tensor]]) -> LossFn:
        return lambda images: self._forward(images, labels).loss

    def attack(self, image: Image.Image, attack: Attack) -> DetectionAttackResult:
        clean = self.prepare(image)
        adversarial = attack(clean, self.loss_fn(self.pseudo_labels(clean)))
        return DetectionAttackResult(clean, adversarial, self.detect(clean), self.detect(adversarial))
