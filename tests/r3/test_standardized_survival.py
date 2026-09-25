from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import statsmodels.api as sm

from phase_evonet.r3.standardized_survival import (
    CONTEXT_LABELS,
    MARGIN_LABELS,
    RELEASE_ORDER,
    STATE_ORDER,
    _context_density_bin,
    _design_formula,
    _direction_stability,
    _inverse_cloglog,
    _margin_bin,
    _support_cell,
    composition_entropy,
    endpoint_state,
    lineage_one_step_bootstrap,
)


def test_endpoint_states_are_exclusive_and_cover_boundaries() -> None:
    cases = [
        (-1e-10, "S0"),
        (1e-8, "S0"),
        (1.01e-8, "S1"),
        (0.010, "S1"),
        (0.0100001, "S2"),
        (0.025, "S2"),
        (0.025001, "S3"),
        (0.050, "S3"),
        (0.050001, "S4"),
    ]
    observed = [endpoint_state(value, present=True, usable=True, absent_state="M") for value, _ in cases]
    assert observed == [expected for _, expected in cases]
    assert endpoint_state(None, present=False, usable=False, absent_state="D") == "D"
    assert endpoint_state(None, present=False, usable=False, absent_state="M") == "M"
    assert endpoint_state(None, present=True, usable=False, absent_state="D") == "M"
    assert set(observed + ["D", "M"]) == set(STATE_ORDER)


def test_composition_entropy_is_deterministic_and_normalized() -> None:
    assert composition_entropy(json.dumps({"Li": 1})) == 0.0
    assert composition_entropy(json.dumps({"Li": 1, "O": 1})) == 1.0
    value = composition_entropy(json.dumps({"Li": 2, "O": 1}, sort_keys=True))
    assert 0.0 < value < 1.0
    assert value == composition_entropy('{"O": 1, "Li": 2}')


def test_frozen_physical_bins_do_not_depend_on_outcomes() -> None:
    margin = _margin_bin(pd.Series([0.0, 0.001, 0.005, 0.010, 0.025, 0.050, 1.0]))
    assert margin.tolist() == [
        "LE_1meV",
        "LE_1meV",
        "1_5meV",
        "5_10meV",
        "10_25meV",
        "25_50meV",
        "GT_50meV",
    ]
    assert tuple(MARGIN_LABELS) == (
        "LE_1meV",
        "1_5meV",
        "5_10meV",
        "10_25meV",
        "25_50meV",
        "GT_50meV",
    )
    density = _context_density_bin(pd.Series([1, 2, 5, 6, 20, 21, 100, 101]))
    assert density.tolist() == ["1", "2_5", "2_5", "6_20", "6_20", "21_100", "21_100", "GT_100"]
    assert len(CONTEXT_LABELS) == 5


def test_declared_exposure_offset_days_are_exact() -> None:
    exposure = {
        "2022-10-28->2023-11-01": 369,
        "2023-11-01->2024-12-18": 413,
        "2024-12-18->2025-09-25": 281,
    }
    assert tuple(exposure) == RELEASE_ORDER
    offsets = {key: math.log(value / 365.25) for key, value in exposure.items()}
    assert math.isclose(offsets[RELEASE_ORDER[0]], math.log(369 / 365.25), abs_tol=1e-15)
    assert math.isclose(offsets[RELEASE_ORDER[1]], math.log(413 / 365.25), abs_tol=1e-15)
    assert math.isclose(offsets[RELEASE_ORDER[2]], math.log(281 / 365.25), abs_tol=1e-15)


def test_lineage_bootstrap_is_deterministic_and_samples_clusters() -> None:
    x = np.column_stack([np.ones(12), np.linspace(-1, 1, 12)])
    y = np.asarray([0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1], dtype=float)
    groups = [f"L{index // 2}" for index in range(12)]
    result = sm.GLM(
        y, x, family=sm.families.Binomial(link=sm.families.links.CLogLog())
    ).fit(maxiter=100)
    first, first_diag = lineage_one_step_bootstrap(
        result, x, groups, replicates=40, seed=42, batch_size=7
    )
    second, second_diag = lineage_one_step_bootstrap(
        result, x, groups, replicates=40, seed=42, batch_size=7
    )
    assert np.array_equal(first, second)
    assert first.shape == (40, 2)
    assert first_diag == second_diag
    assert first_diag["unit"] == "canonical_lineage_id"
    assert first_diag["clusters"] == 6
    assert first_diag["successful_replicates"] == 40


def test_direct_standardization_inverse_link_toy_result() -> None:
    eta = np.asarray([-2.0, 0.0, 2.0])
    observed = _inverse_cloglog(eta)
    expected = 1 - np.exp(-np.exp(eta))
    assert np.allclose(observed, expected, atol=1e-14)
    assert np.all(np.diff(observed) > 0)


def test_design_contains_only_source_covariates_plus_release() -> None:
    spec = {"spline_knots": [0.01], "spline_upper_bound": 1.0}
    formula = _design_formula(spec)
    assert "release_pair" in formula
    assert "source_stability_margin_eV_per_atom" in formula
    assert "source_context_entry_count" in formula
    assert "target_" not in formula
    assert "outcome" not in formula


def test_support_cell_uses_registered_nonparametric_dimensions() -> None:
    frame = pd.DataFrame(
        {
            "thermo_type": ["GGA_GGA+U"],
            "source_chemsys_dimensionality": [3],
            "source_margin_bin": ["1_5meV"],
            "source_context_density_bin": ["6_20"],
        }
    )
    assert _support_cell(frame).iloc[0] == "GGA_GGA+U|d=3|m=1_5meV|c=6_20"


def test_direction_stability_requires_at_least_one_sensitivity_reference() -> None:
    rows = []
    for reference, risks in {
        "pooled_source_person_period": [0.010, 0.012, 0.011],
        "first_release_source_population": [0.011, 0.013, 0.012],
        "latest_release_source_population": [0.009, 0.011, 0.010],
    }.items():
        for release, risk in zip(RELEASE_ORDER, risks, strict=True):
            rows.append(
                {
                    "outcome": "exact",
                    "estimate_type": "model_direct_standardization",
                    "reference_population": reference,
                    "release_pair": release,
                    "risk": risk,
                }
            )
    result = _direction_stability(pd.DataFrame(rows), "exact")
    assert result["stable_under_at_least_two_references"] is True
    assert result["sensitivity_references_stable_vs_pooled"] == 2
