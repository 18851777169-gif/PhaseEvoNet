from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_transitions(n_materials: int = 1600, n_snapshots: int = 4, seed: int = 42) -> pd.DataFrame:
    if n_snapshots < 3:
        raise ValueError("n_snapshots must be at least 3")
    rng = np.random.default_rng(seed)
    material_id = np.arange(n_materials)
    latent_fragility = rng.normal(0, 1, n_materials)
    rows: list[dict[str, float | int | str]] = []
    for t in range(n_snapshots - 1):
        current_hull = np.maximum(0.0, rng.gamma(1.3, 18.0, n_materials) - 10.0)
        phase_coverage = np.clip(rng.beta(2 + t * 0.4, 2.5, n_materials), 0, 1)
        competitor_growth = np.clip(rng.normal(0.18 + 0.06 * t, 0.12, n_materials), 0, 0.8)
        structure_complexity = rng.normal(0, 1, n_materials)
        logit = (
            -3.0
            + 0.055 * np.maximum(0, 25 - current_hull)
            + 1.8 * (1 - phase_coverage)
            + 2.3 * competitor_growth
            + 0.7 * latent_fragility
            + 0.25 * structure_complexity
        )
        prob = 1 / (1 + np.exp(-logit))
        flip = rng.binomial(1, np.clip(prob, 0.005, 0.9))
        for i in range(n_materials):
            rows.append({
                "canonical_lineage_id": f"L{i:06d}",
                "transition": t,
                "current_hull_mev": float(current_hull[i]),
                "phase_coverage": float(phase_coverage[i]),
                "competitor_growth": float(competitor_growth[i]),
                "structure_complexity": float(structure_complexity[i]),
                "flip": int(flip[i]),
            })
    return pd.DataFrame(rows)
