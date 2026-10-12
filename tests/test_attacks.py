"""CPU-only tests for the attack implementations, using a tiny random classifier."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from gen_attack.attacks import FGSM, PGD, Attack, SquareAttack, available_attacks, get_attack, register_attack

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
        return F.cross_entropy(model(x), labels, reduction="none")

    return images, loss_fn


ATTACKS = [
    pytest.param(lambda eps: FGSM(epsilon=eps), id="fgsm"),
    pytest.param(lambda eps: PGD(epsilon=eps, steps=5, seed=0), id="pgd"),
    pytest.param(lambda eps: SquareAttack(epsilon=eps, n_queries=100, seed=0), id="square"),
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

    assert loss_fn(adversarial).sum() > loss_fn(images).sum()


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
    fgsm_loss = loss_fn(FGSM(epsilon=EPSILON)(images, loss_fn)).sum()
    pgd_loss = loss_fn(PGD(epsilon=EPSILON, steps=20, random_start=False)(images, loss_fn)).sum()

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
    with pytest.raises(ValueError):
        SquareAttack(epsilon=0.1, n_queries=0)
    with pytest.raises(ValueError):
        SquareAttack(epsilon=0.1, p_init=0)


def test_registry():
    assert {"fgsm", "pgd", "square"} <= set(available_attacks())
    attack = get_attack("PGD", epsilon=0.01, steps=3)
    assert isinstance(attack, PGD)
    assert attack.steps == 3

    with pytest.raises(KeyError, match="Unknown attack"):
        get_attack("does-not-exist", epsilon=0.1)
    with pytest.raises(ValueError, match="already registered"):
        register_attack("fgsm")(type("Duplicate", (Attack,), {"perturb": lambda self, x, f: x}))


def test_square_seed_is_reproducible(setup):
    images, loss_fn = setup
    first = SquareAttack(epsilon=EPSILON, n_queries=30, seed=7)(images, loss_fn)
    second = SquareAttack(epsilon=EPSILON, n_queries=30, seed=7)(images, loss_fn)

    assert torch.equal(first, second)


def test_square_loss_never_decreases_between_queries(setup):
    images, loss_fn = setup
    losses = []
    SquareAttack(
        epsilon=EPSILON, n_queries=40, seed=0, callback=lambda _, adversarial: losses.append(loss_fn(adversarial))
    )(images, loss_fn)

    assert len(losses) == 40
    for before, after in zip(losses, losses[1:]):
        assert torch.all(after >= before - 1e-6)


def test_square_uses_no_gradients(setup):
    images, loss_fn = setup

    def gradient_free_loss(x):
        assert not torch.is_grad_enabled()
        return loss_fn(x)

    SquareAttack(epsilon=EPSILON, n_queries=5, seed=0)(images, gradient_free_loss)


def test_square_rejects_batch_reduced_loss(setup):
    images, loss_fn = setup
    with pytest.raises(ValueError, match="one loss value per image"):
        SquareAttack(epsilon=EPSILON, n_queries=5)(images, lambda x: loss_fn(x).mean())
