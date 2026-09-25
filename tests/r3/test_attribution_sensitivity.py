from __future__ import annotations

import pytest

from phase_evonet.r3.attribution_sensitivity import (
    FULL_MASK,
    PLAYERS,
    harsanyi_dividends,
    reconstruct_from_dividends,
    summarize_game,
)


def test_additive_game_exactly_reconstructs_all_summaries() -> None:
    weights = (0.011, -0.004, 0.002, 0.007)
    values = {
        mask: 0.003 + sum(weight for index, weight in enumerate(weights) if mask & (1 << index))
        for mask in range(FULL_MASK + 1)
    }
    summary = summarize_game(values)
    for method in ("shapley", "single_switch", "leave_one_out", "total_effect"):
        assert [summary[method][player] for player in PLAYERS] == pytest.approx(weights)
    assert summary["shapley_residual"] == pytest.approx(0.0, abs=1e-14)
    assert summary["mobius_max_reconstruction_error"] == pytest.approx(0.0, abs=1e-14)
    assert all(
        abs(value) < 1e-14
        for coalition, value in summary["harsanyi"].items()
        if coalition.bit_count() >= 2
    )


def test_interacting_game_discloses_pair_dividend() -> None:
    values = {mask: 0.0 for mask in range(FULL_MASK + 1)}
    for mask in values:
        if mask & 1 and mask & 2:
            values[mask] = 0.020
    summary = summarize_game(values)
    assert summary["harsanyi"][3] == pytest.approx(0.020)
    assert summary["shapley"][PLAYERS[0]] == pytest.approx(0.010)
    assert summary["shapley"][PLAYERS[1]] == pytest.approx(0.010)
    assert summary["single_switch"][PLAYERS[0]] == pytest.approx(0.0)
    assert summary["leave_one_out"][PLAYERS[0]] == pytest.approx(0.020)


def test_mobius_reconstruction_covers_every_coalition() -> None:
    values = {mask: (mask * mask - 3 * mask) / 1000 for mask in range(FULL_MASK + 1)}
    dividends = harsanyi_dividends(values)
    rebuilt = reconstruct_from_dividends(dividends, values[0])
    assert rebuilt == pytest.approx(values)


def test_incomplete_game_is_rejected() -> None:
    with pytest.raises(ValueError, match="16"):
        summarize_game({mask: 0.0 for mask in range(15)})
