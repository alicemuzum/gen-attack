# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Gen-Attack benchmarks adversarial attacks (FGSM, PGD) against a ResNet-18 ImageNet classifier and the DETR object detector. Stable Diffusion and Grounding DINO wrappers exist for generating and labelling test scenes. This repo is a CV showcase, so the README, the figures in `assets/results/` and code quality all matter.

## Commands

```bash
poetry install                       # or: pip install -e . (needs Python 3.9–3.12)
poetry install --extras generation   # adds diffusers/accelerate for `gen-attack generate`

pytest                               # all tests: CPU only, no model downloads
pytest tests/test_attacks.py -k pgd  # a single test
ruff check . && ruff format --check .
pre-commit run --all-files

gen-attack classify                  # ResNet-18 robustness curve -> results/
gen-attack detect --image assets/examples/bench.png --attack pgd --epsilon 0.0078
docker compose run --rm cpu classify
```

`results/` is git-ignored. To refresh the README figures, run with `--output assets/results`. The committed photo figures are JPEG-compressed copies of the PNGs the CLI produces. CI (`.github/workflows/ci.yml`) runs ruff and runs pytest on Python 3.9 and 3.12 with CPU-only torch.

## Architecture

- **Attacks never see a model.** `Attack.perturb(images, loss_fn)` takes images in [0, 1] and a closure that returns a scalar loss to maximise. This is what lets the same FGSM/PGD code attack both a classifier (cross-entropy) and DETR (its Hungarian-matched set loss). New attacks subclass `Attack` in `src/gen_attack/attacks/`, use `@register_attack("name")`, and must be imported in `attacks/__init__.py`. They then show up in `get_attack()` and as CLI `choices` automatically.
- **Normalisation lives inside the model wrappers.** `ResNetBenchmark` prepends a `Normalize` layer, and `DETRBenchmark` normalises in `_forward` after calling the processor with `do_normalize=False`. Attacks, epsilons and clipping therefore all work in raw [0, 1] pixel space. Don't feed normalised tensors to an attack.
- **DETR has no ground truth.** `DETRBenchmark.pseudo_labels` takes the model's own clean detections above `threshold` as targets (normalised cxcywh boxes from `pred_boxes`). It raises if nothing is detected.
- Model parameters are frozen (`requires_grad_(False)`). Attacks take gradients only with respect to the input, through `torch.autograd.grad`.
- `generation/` and the DETR wrapper import `transformers`/`diffusers` lazily inside `__init__`, so importing the package and running the tests stay cheap.
- `visualize.py` forces the Agg backend and only saves figures. It never shows them.
- `notebooks/` contains the original exploratory work and is not part of the package. Ruff excludes it.

Code uses `from __future__ import annotations` with PEP 604 types, but runtime-evaluated aliases (e.g. `LossFn`, `PathLike`) must stay `typing`-style for Python 3.9.
