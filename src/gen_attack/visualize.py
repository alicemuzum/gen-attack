"""Figures for attack results. Every function saves to ``path`` and closes its figure."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from gen_attack.models.detr import DetectionAttackResult, Detections  # noqa: E402
from gen_attack.models.resnet import QueryCurve, RobustnessCurve  # noqa: E402
from gen_attack.utils import to_hwc_numpy  # noqa: E402

PathLike = Union[str, Path]

CORRECT = "#1a7f37"
WRONG = "#cf222e"
BOX = "#e85d04"


def _save(fig: plt.Figure, path: PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_robustness_curves(curves: Sequence[RobustnessCurve], path: PathLike, title: str | None = None) -> Path:
    """Plot accuracy against epsilon, one line per attack."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for curve in curves:
        ax.plot(
            [eps * 255 for eps in curve.epsilons],
            [acc * 100 for acc in curve.accuracies],
            marker="o",
            markersize=4,
            linewidth=2,
            label=curve.attack.upper(),
        )
    ax.set_xlabel(r"Perturbation budget $\epsilon$ (in 1/255 pixel steps)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(-3, 103)
    ax.set_title(title or "Accuracy under attack")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def plot_query_curves(
    curves: Sequence[QueryCurve],
    path: PathLike,
    clean_accuracy: float | None = None,
    white_box: dict[float, float] | None = None,
    title: str | None = None,
) -> Path:
    """Plot black-box accuracy against queries per image, one line per epsilon.

    ``white_box`` maps each epsilon to the accuracy a white-box attack reaches at that budget,
    drawn as a dashed reference line. Budgets with the same accuracy share one line.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for curve in curves:
        ax.plot(
            curve.queries,
            [acc * 100 for acc in curve.accuracies],
            marker="o",
            markersize=3,
            linewidth=2,
            label=f"{curve.attack.capitalize()}, ε = {curve.epsilon * 255:g}/255",
        )
    budgets_by_accuracy: dict[float, list[float]] = {}
    for epsilon, accuracy in (white_box or {}).items():
        budgets_by_accuracy.setdefault(accuracy, []).append(epsilon)
    for accuracy, epsilons in budgets_by_accuracy.items():
        budgets = ", ".join(f"{eps * 255:g}/255" for eps in sorted(epsilons))
        ax.axhline(
            accuracy * 100, color="black", linestyle="--", linewidth=1.2, label=f"PGD (white-box), ε = {budgets}"
        )
    if clean_accuracy is not None:
        ax.axhline(clean_accuracy * 100, color="gray", linestyle=":", linewidth=1.2, label="No attack")
    ax.set_xscale("log")
    ax.set_xlabel("Queries per image (log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(-3, 103)
    ax.set_title(title or "Black-box attack: accuracy vs. queries")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def plot_classification_examples(
    clean_images: torch.Tensor,
    adversarial_images: torch.Tensor,
    labels: torch.Tensor,
    clean_predictions: torch.Tensor,
    adversarial_predictions: torch.Tensor,
    class_names: Sequence[str],
    path: PathLike,
    num_examples: int = 6,
    title: str | None = None,
) -> Path:
    """Show clean images above their adversarial versions, with predictions coloured by correctness."""
    num_examples = min(num_examples, len(clean_images))
    fig, axes = plt.subplots(3, num_examples, figsize=(2.4 * num_examples, 7.6), squeeze=False)

    def short(index: int) -> str:
        return class_names[index].split(",")[0]

    for col in range(num_examples):
        label = labels[col].item()
        perturbation = adversarial_images[col] - clean_images[col]
        scale = perturbation.abs().max().clamp_min(1e-12)
        rows = [
            (clean_images[col], clean_predictions[col].item()),
            (adversarial_images[col], adversarial_predictions[col].item()),
        ]
        for row, (image, prediction) in enumerate(rows):
            ax = axes[row, col]
            ax.imshow(to_hwc_numpy(image))
            ax.set_title(short(prediction), color=CORRECT if prediction == label else WRONG, fontsize=10)
        axes[2, col].imshow(to_hwc_numpy(perturbation / (2 * scale) + 0.5))
        axes[0, col].text(
            0.5,
            1.25,
            f"true: {short(label)}",
            transform=axes[0, col].transAxes,
            ha="center",
            fontsize=9,
            color="dimgray",
        )

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    for row, name in enumerate(["Clean", "Adversarial", "Perturbation\n(amplified)"]):
        axes[row, 0].set_ylabel(name, fontsize=11)

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    return _save(fig, path)


def draw_detections(ax: Axes, image: torch.Tensor, detections: Detections, title: str = "") -> None:
    ax.imshow(to_hwc_numpy(image))
    for score, label, (x_min, y_min, x_max, y_max) in zip(
        detections.scores.tolist(), detections.labels, detections.boxes.tolist()
    ):
        ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, linewidth=2.5, color=BOX))
        ax.text(
            x_min,
            y_min,
            f"{label} {score:.2f}",
            fontsize=10,
            color="white",
            va="bottom",
            bbox={"facecolor": BOX, "edgecolor": "none", "pad": 2},
        )
    count = len(detections)
    ax.set_title(f"{title} ({count} detection{'' if count == 1 else 's'})" if title else "", fontsize=12)
    ax.axis("off")


def plot_detection_comparison(result: DetectionAttackResult, path: PathLike, title: str | None = None) -> Path:
    """Show detections on the clean image next to detections on the adversarial image."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    draw_detections(axes[0], result.clean_image, result.clean_detections, "Clean")
    draw_detections(axes[1], result.adversarial_image, result.adversarial_detections, "Adversarial")
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return _save(fig, path)
