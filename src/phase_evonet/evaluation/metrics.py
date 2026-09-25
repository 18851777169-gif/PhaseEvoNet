from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1 else y_prob <= hi)
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def top_fraction_enrichment(y_true, y_prob, fraction: float = 0.10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if not (0 < fraction <= 1):
        raise ValueError("fraction must be in (0, 1]")
    prevalence = y_true.mean()
    if prevalence == 0:
        return float("nan")
    k = max(1, int(np.ceil(len(y_true) * fraction)))
    idx = np.argsort(-y_prob)[:k]
    return float(y_true[idx].mean() / prevalence)


def classification_metrics(y_true, y_prob) -> dict[str, float]:
    return {
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece_10": expected_calibration_error(y_true, y_prob, n_bins=10),
        "top_decile_enrichment": top_fraction_enrichment(y_true, y_prob, 0.10),
    }
