from __future__ import annotations

from pathlib import Path

from .data.synthetic import make_synthetic_transitions
from .evaluation.metrics import classification_metrics
from .manifest import write_report
from .models.baseline import build_logistic_baseline


FEATURES = ["current_hull_mev", "phase_coverage", "competitor_growth", "structure_complexity"]


def run_smoke(output_dir: str | Path, seed: int = 42) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = make_synthetic_transitions(seed=seed)
    train = df[df["transition"] <= 1].copy()
    test = df[df["transition"] == 2].copy()
    model = build_logistic_baseline(seed=seed)
    model.fit(train[FEATURES], train["flip"])
    prob = model.predict_proba(test[FEATURES])[:, 1]
    metrics = classification_metrics(test["flip"].to_numpy(), prob)
    payload = {
        "task_id": "P0.1",
        "status": "PASS",
        "synthetic_only": True,
        "split": {"train_transitions": [0, 1], "test_transition": 2},
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "prevalence_test": float(test["flip"].mean()),
        "metrics": metrics,
        "gate_decision": "NOT_APPLICABLE",
    }
    write_report(out / "report.json", payload)
    test.assign(predicted_risk=prob).head(100).to_csv(out / "predictions_preview.csv", index=False)
    return payload
