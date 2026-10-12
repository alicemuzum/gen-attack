"""Adversarial attacks. Importing this package registers all built-in attacks."""

from gen_attack.attacks.base import Attack, LossFn
from gen_attack.attacks.fgsm import FGSM
from gen_attack.attacks.pgd import PGD
from gen_attack.attacks.registry import available_attacks, get_attack, register_attack
from gen_attack.attacks.square import SquareAttack

__all__ = [
    "FGSM",
    "PGD",
    "Attack",
    "SquareAttack",
    "LossFn",
    "available_attacks",
    "get_attack",
    "register_attack",
]
