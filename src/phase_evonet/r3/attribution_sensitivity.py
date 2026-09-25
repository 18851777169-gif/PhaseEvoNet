"""Pure, independently testable four-channel cooperative-game summaries for R3.3."""

from __future__ import annotations

import itertools
import math
from typing import Mapping

import numpy as np
from scipy.stats import spearmanr


PLAYERS = (
    "competitor_inventory",
    "uncorrected_energy",
    "compatibility_correction",
    "candidate_identity",
)
FULL_MASK = (1 << len(PLAYERS)) - 1


def _complete(values: Mapping[int, float]) -> dict[int, float]:
    result = {int(key): float(value) for key, value in values.items()}
    if set(result) != set(range(FULL_MASK + 1)):
        raise ValueError("R3.3 requires exactly the 16 frozen coalition values")
    if not np.isfinite(list(result.values())).all():
        raise ValueError("coalition values must all be finite")
    return result


def exact_shapley(values: Mapping[int, float]) -> dict[str, float]:
    """Recompute exact Shapley values without calling the P3.3 implementation."""

    game = _complete(values)
    n = len(PLAYERS)
    denominator = math.factorial(n)
    output: dict[str, float] = {}
    for index, player in enumerate(PLAYERS):
        bit = 1 << index
        total = 0.0
        for coalition in range(FULL_MASK + 1):
            if coalition & bit:
                continue
            size = coalition.bit_count()
            weight = math.factorial(size) * math.factorial(n - size - 1) / denominator
            total += weight * (game[coalition | bit] - game[coalition])
        output[player] = float(total)
    return output


def harsanyi_dividends(values: Mapping[int, float]) -> dict[int, float]:
    """Return the Möbius/Harsanyi dividend for every non-empty coalition."""

    game = _complete(values)
    output: dict[int, float] = {}
    for coalition in range(1, FULL_MASK + 1):
        dividend = 0.0
        subset = coalition
        while True:
            dividend += (-1.0) ** (coalition.bit_count() - subset.bit_count()) * game[subset]
            if subset == 0:
                break
            subset = (subset - 1) & coalition
        output[coalition] = float(dividend)
    return output


def reconstruct_from_dividends(dividends: Mapping[int, float], baseline: float) -> dict[int, float]:
    output = {0: float(baseline)}
    for coalition in range(1, FULL_MASK + 1):
        output[coalition] = float(
            baseline
            + sum(value for subset, value in dividends.items() if subset & coalition == subset)
        )
    return output


def rank_channels(contributions: Mapping[str, float]) -> tuple[str, ...]:
    """Deterministic descending absolute rank with frozen player order as tie break."""

    return tuple(
        sorted(PLAYERS, key=lambda player: (-abs(float(contributions[player])), PLAYERS.index(player)))
    )


def rank_spearman(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    left_rank = {player: rank for rank, player in enumerate(rank_channels(left), start=1)}
    right_rank = {player: rank for rank, player in enumerate(rank_channels(right), start=1)}
    value = spearmanr(
        [left_rank[player] for player in PLAYERS],
        [right_rank[player] for player in PLAYERS],
    ).statistic
    return float(value)


def summarize_game(values: Mapping[int, float]) -> dict[str, object]:
    game = _complete(values)
    shapley = exact_shapley(game)
    single_switch = {
        player: float(game[1 << index] - game[0]) for index, player in enumerate(PLAYERS)
    }
    leave_one_out = {
        player: float(game[FULL_MASK] - game[FULL_MASK ^ (1 << index)])
        for index, player in enumerate(PLAYERS)
    }
    total_effect = {}
    for index, player in enumerate(PLAYERS):
        bit = 1 << index
        marginal = [
            game[coalition | bit] - game[coalition]
            for coalition in range(FULL_MASK + 1)
            if not coalition & bit
        ]
        total_effect[player] = float(np.mean(marginal))
    dividends = harsanyi_dividends(game)
    reconstructed = reconstruct_from_dividends(dividends, game[0])
    endpoint_delta = float(game[FULL_MASK] - game[0])
    pair_or_higher = float(
        sum(abs(value) for coalition, value in dividends.items() if coalition.bit_count() >= 2)
    )
    return {
        "shapley": shapley,
        "single_switch": single_switch,
        "leave_one_out": leave_one_out,
        "total_effect": total_effect,
        "harsanyi": dividends,
        "endpoint_delta": endpoint_delta,
        "shapley_residual": float(sum(shapley.values()) - endpoint_delta),
        "mobius_max_reconstruction_error": float(
            max(abs(reconstructed[mask] - game[mask]) for mask in game)
        ),
        "interaction_absolute_total": pair_or_higher,
        "rank_orders": {
            method: rank_channels(contributions)
            for method, contributions in (
                ("exact_shapley", shapley),
                ("single_switch", single_switch),
                ("leave_one_out", leave_one_out),
                ("total_effect", total_effect),
            )
        },
        "rank_spearman_vs_shapley": {
            method: rank_spearman(shapley, contributions)
            for method, contributions in (
                ("single_switch", single_switch),
                ("leave_one_out", leave_one_out),
                ("total_effect", total_effect),
            )
        },
    }


def coalition_label(mask: int) -> str:
    members = [player for index, player in enumerate(PLAYERS) if mask & (1 << index)]
    return "+".join(members) if members else "baseline"


def all_player_subsets() -> list[tuple[str, ...]]:
    return [
        subset
        for size in range(1, len(PLAYERS) + 1)
        for subset in itertools.combinations(PLAYERS, size)
    ]
