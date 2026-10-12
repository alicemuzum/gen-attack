"""CPU-only tests for the attack implementations, using a tiny random classifier."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from gen_attack.attacks import FGSM, PGD, Attack, available_attacks, get_attack, register_attack

EPSILON = 8 / 255


@pytest.fixture
def setup():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 10)
    ).eval()
    images = torch.rand(4, 3, 16, 16)
    labels = torch.randint(0, 10, (4,))

    def loss_fn(x):
        return F.cross_entropy(model(x), labels)

    return images, loss_fn


ATTACKS = [
    pytest.param(lambda eps: FGSM(epsilon=eps), id="fgsm"),
    pytest.param(lambda eps: PGD(epsilon=eps, steps=5, seed=0), id="pgd"),
]


@pytest.mark.parametrize("make_attack", ATTACKS)
def test_perturbation_within_budget_and_valid_range(setup, make_attack):
    images, loss_fn = setup
    adversarial = make_attack(EPSILON)(images, loss_fn)

    assert adversarial.shape == images.shape
    assert (adversarial - images).abs().max() <= EPSILON + 1e-6
    assert adversarial.min() >= 0 and adversarial.max() <= 1


@pytest.mark.parametrize("make_attack", ATTACKS)
def test_attack_increases_loss(setup, make_attack):
    images, loss_fn = setup
    adversarial = make_attack(EPSILON)(images, loss_fn)

    assert loss_fn(adversarial) > loss_fn(images)


@pytest.mark.parametrize("make_attack", ATTACKS)
def test_zero_epsilon_is_identity(setup, make_attack):
    images, loss_fn = setup
    assert torch.equal(make_attack(0.0)(images, loss_fn), images)


@pytest.mark.parametrize("make_attack", ATTACKS)
def test_input_is_not_modified(setup, make_attack):
    images, loss_fn = setup
    original = images.clone()
    adversarial = make_attack(EPSILON)(images, loss_fn)

    assert torch.equal(images, original)
    assert images.grad is None
    assert not adversarial.requires_grad


def test_pgd_is_at_least_as_strong_as_fgsm(setup):
    images, loss_fn = setup
    fgsm_loss = loss_fn(FGSM(epsilon=EPSILON)(images, loss_fn))
    pgd_loss = loss_fn(PGD(epsilon=EPSILON, steps=20, random_start=False)(images, loss_fn))

    assert pgd_loss >= fgsm_loss - 1e-4


def test_pgd_seed_is_reproducible(setup):
    images, loss_fn = setup
    first = PGD(epsilon=EPSILON, steps=3, seed=42)(images, loss_fn)
    second = PGD(epsilon=EPSILON, steps=3, seed=42)(images, loss_fn)

    assert torch.equal(first, second)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        FGSM(epsilon=-0.1)
    with pytest.raises(ValueError):
        PGD(epsilon=0.1, steps=0)


def test_registry():
    assert {"fgsm", "pgd"} <= set(available_attacks())
    attack = get_attack("PGD", epsilon=0.01, steps=3)
    assert isinstance(attack, PGD)
    assert attack.steps == 3

    with pytest.raises(KeyError, match="Unknown attack"):
        get_attack("does-not-exist", epsilon=0.1)
    with pytest.raises(ValueError, match="already registered"):
        register_attack("fgsm")(type("Duplicate", (Attack,), {"perturb": lambda self, x, f: x}))
