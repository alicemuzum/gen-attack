"""Name-based lookup of attack implementations."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from gen_attack.attacks.base import Attack

AttackT = TypeVar("AttackT", bound=type[Attack])

_REGISTRY: dict[str, type[Attack]] = {}


def register_attack(name: str) -> Callable[[AttackT], AttackT]:
    """Class decorator that makes an attack available under ``name``."""

    def decorator(cls: AttackT) -> AttackT:
        if name in _REGISTRY:
            raise ValueError(f"An attack named '{name}' is already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_attack(name: str, **kwargs: Any) -> Attack:
    """Instantiate the attack registered under ``name`` with ``kwargs``."""
    try:
        cls = _REGISTRY[name.lower()]
    except KeyError:
        raise KeyError(f"Unknown attack '{name}'. Available attacks: {', '.join(available_attacks())}") from None
    return cls(**kwargs)


def available_attacks() -> list[str]:
    return sorted(_REGISTRY)
