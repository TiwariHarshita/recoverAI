from random import Random
from typing import TypeVar


T = TypeVar("T")


def weighted_choice(
    rng: Random,
    weights: dict[T, float],
) -> T:
    """
    Pick one value using non-negative relative weights.

    The caller supplies its Random instance so generation remains
    deterministic and does not mutate Python's global random state.
    """

    if not weights:
        raise ValueError(
            "Weight map cannot be empty."
        )

    if any(
        weight < 0
        for weight in weights.values()
    ):
        raise ValueError(
            "Weights cannot be negative."
        )

    total = sum(
        weights.values()
    )

    if total <= 0:
        raise ValueError(
            "Weight map must have a positive total."
        )

    threshold = (
        rng.random()
        * total
    )

    cumulative = 0.0

    items = list(
        weights.items()
    )

    for item, weight in items:
        cumulative += weight

        if threshold <= cumulative:
            return item

    # Handles tiny floating-point rounding differences.
    return items[-1][0]