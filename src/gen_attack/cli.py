"""Command-line interface: ``gen-attack {classify,detect,generate}``."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from gen_attack.attacks import available_attacks, get_attack
from gen_attack.utils import seed_everything


def _attack_kwargs(name: str, args: argparse.Namespace) -> dict[str, Any]:
    if name == "pgd":
        return {"steps": args.pgd_steps, "seed": args.seed}
    return {}


def run_classify(args: argparse.Namespace) -> None:
    from gen_attack.models import ResNetBenchmark
    from gen_attack.visualize import plot_classification_examples, plot_robustness_curves

    benchmark = ResNetBenchmark(device=args.device)
    images, labels = benchmark.load_samples(args.samples)
    clean_predictions = benchmark.predict(images)
    epsilons = np.linspace(0.0, args.eps_max, args.num_eps)
    print(f"ResNet-18 on {len(images)} ImageNet images, device={benchmark.device}")
    print(f"Clean accuracy: {(clean_predictions == labels).float().mean().item():.1%}\n")

    curves = []
    for name in args.attacks:
        curve = benchmark.robustness_curve(name, epsilons, images, labels, **_attack_kwargs(name, args))
        curves.append(curve)
        print(
            f"{name.upper():>5}  "
            + "  ".join(f"{eps * 255:4.1f}/255: {acc:4.0%}" for eps, acc in zip(curve.epsilons, curve.accuracies))
        )

    output = Path(args.output)
    print(
        f"\nSaved {plot_robustness_curves(curves, output / 'resnet_robustness.png', 'ResNet-18 accuracy under attack')}"
    )

    example_attack = get_attack(args.attacks[-1], epsilon=args.example_eps, **_attack_kwargs(args.attacks[-1], args))
    outcome = benchmark.evaluate(example_attack, images, labels)
    path = plot_classification_examples(
        images,
        outcome.adversarial_images,
        labels,
        clean_predictions,
        outcome.predictions,
        benchmark.class_names,
        output / "resnet_examples.png",
        title=f"{example_attack.name.upper()}, ε = {args.example_eps * 255:g}/255",
    )
    print(f"Saved {path}")


def run_detect(args: argparse.Namespace) -> None:
    from PIL import Image

    from gen_attack.models import DETRBenchmark
    from gen_attack.visualize import plot_detection_comparison

    benchmark = DETRBenchmark(threshold=args.threshold, device=args.device)
    attack = get_attack(args.attack, epsilon=args.epsilon, **_attack_kwargs(args.attack, args))
    result = benchmark.attack(Image.open(args.image), attack)

    print(f"Attack:      {attack}")
    print(f"Clean:       {result.clean_detections.summary()}")
    print(f"Adversarial: {result.adversarial_detections.summary()}")

    path = Path(args.output) / f"detr_{Path(args.image).stem}_{attack.name}.png"
    title = f"DETR under {attack.name.upper()}, ε = {args.epsilon * 255:g}/255"
    print(f"Saved {plot_detection_comparison(result, path, title)}")


def run_generate(args: argparse.Namespace) -> None:
    from gen_attack.generation import StableDiffusionGenerator

    generator = StableDiffusionGenerator(device=args.device)
    print(f"Saved {generator.generate_to_file(args.prompt, args.output, seed=args.seed)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gen-attack", description="Benchmark adversarial attacks on vision models.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default=None, help="torch device, e.g. cpu or cuda (default: auto)")
    common.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    attack_opts = argparse.ArgumentParser(add_help=False)
    attack_opts.add_argument("--pgd-steps", type=int, default=10, help="PGD iterations (default: 10)")
    attack_opts.add_argument("--output", default="results", help="output directory (default: results)")

    sub = parser.add_subparsers(dest="command", required=True)

    classify = sub.add_parser(
        "classify", parents=[common, attack_opts], help="robustness curve of ResNet-18 on ImageNet samples"
    )
    classify.add_argument("--attacks", nargs="+", choices=available_attacks(), default=available_attacks())
    classify.add_argument("--eps-max", type=float, default=1 / 255, help="largest epsilon, in [0, 1] units")
    classify.add_argument("--num-eps", type=int, default=11, help="number of epsilons in the sweep")
    classify.add_argument("--example-eps", type=float, default=1 / 255, help="epsilon for the example grid")
    classify.add_argument("--samples", type=int, default=16, help="number of ImageNet samples (max 20)")
    classify.set_defaults(func=run_classify)

    detect = sub.add_parser("detect", parents=[common, attack_opts], help="attack DETR on a single image")
    detect.add_argument("--image", default="assets/examples/bench.png")
    detect.add_argument("--attack", choices=available_attacks(), default="pgd")
    detect.add_argument("--epsilon", type=float, default=2 / 255, help="perturbation budget, in [0, 1] units")
    detect.add_argument("--threshold", type=float, default=0.9, help="detection confidence threshold")
    detect.set_defaults(func=run_detect)

    generate = sub.add_parser("generate", parents=[common], help="generate a test image with Stable Diffusion")
    generate.add_argument("prompt")
    generate.add_argument("--output", default="generated.png")
    generate.set_defaults(func=run_generate)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    seed_everything(args.seed)
    args.func(args)


if __name__ == "__main__":
    main()
