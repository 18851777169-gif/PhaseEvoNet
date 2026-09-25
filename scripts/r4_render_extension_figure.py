from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


LABEL_NAMES = {
    "exact_zero": "Exact zero",
    "within_10meV": "Within 10 meV",
    "within_25meV": "Within 25 meV",
}
METRIC_NAMES = {
    "average_precision": "Average precision",
    "brier_score": "Brier score",
    "roc_auc": "ROC AUC",
    "expected_calibration_error": "ECE",
}
COLORS = {"exact_zero": "#1B4F72", "within_10meV": "#D35400", "within_25meV": "#148F77"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    args = parser.parse_args()
    repo = args.repo.resolve()
    report = repo / "reports/R4_0"
    output = report / "figures/current_release_extension"
    output.mkdir(parents=True, exist_ok=True)

    labels_path = repo / "data/processed/R4_0/versioned_labels_2024_2026.parquet"
    labels = pd.read_parquet(labels_path)
    metrics = pd.read_csv(report / "fixed_score_metrics_2024_2026.csv")
    deltas = pd.read_csv(report / "fixed_score_metric_deltas_2025_2026.csv")
    state = pd.read_csv(report / "state_label_evolution_2025_2026.csv")
    rankings = pd.read_csv(report / "model_rankings_2024_2026.csv")
    versions = ["2024-12-18", "2025-09-25", "2026-04-13"]
    representatives = ["M0", "M1", "M2", "M5"]
    core_metrics = list(METRIC_NAMES)

    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "text.color": "#111111",
            "axes.labelcolor": "#111111",
            "axes.edgecolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(14.2, 10.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    prevalence = (
        labels.groupby(["label_definition", "label_version"], as_index=False)
        .agg(events=("event", "sum"), panel_rows=("event", "size"))
    )
    prevalence["prevalence_percent"] = 100 * prevalence["events"] / prevalence["panel_rows"]
    for label in LABEL_NAMES:
        rows = prevalence[prevalence["label_definition"].eq(label)].set_index("label_version").reindex(versions)
        ax_a.plot(
            range(3), rows["prevalence_percent"], marker="o", markersize=7,
            linewidth=2.3, color=COLORS[label], label=LABEL_NAMES[label],
        )
        for x, (_, row) in enumerate(rows.iterrows()):
            offset = (0, 9)
            vertical_alignment = "bottom"
            horizontal_alignment = "center"
            if x == 0:
                offset = {
                    "exact_zero": (0, 10),
                    "within_10meV": (7, 7),
                    "within_25meV": (7, 9),
                }[label]
                horizontal_alignment = {
                    "exact_zero": "center",
                    "within_10meV": "left",
                    "within_25meV": "left",
                }[label]
            elif x == 2 and label == "exact_zero":
                offset = (0, -12)
                vertical_alignment = "top"
            ax_a.annotate(
                (
                    f"{int(row.events):,} ({row.prevalence_percent:.2f}%)"
                    if x == 0
                    else f"{int(row.events):,}\n({row.prevalence_percent:.2f}%)"
                ),
                (x, row.prevalence_percent), xytext=offset, textcoords="offset points",
                ha=horizontal_alignment, va=vertical_alignment, color="#111111", fontsize=9.5,
            )
    ax_a.set_xticks(range(3), ["2024", "2025", "2026"])
    ax_a.set_xlabel("Materials Project label version")
    ax_a.set_ylabel("Revision-event prevalence (%)")
    ax_a.set_title("a  Fixed candidates acquire version-dependent event labels", loc="left", fontweight="bold", pad=8)
    ax_a.legend(frameon=False, ncol=1, loc="upper left")
    ax_a.grid(axis="x", visible=False)

    x = np.arange(len(state))
    a = state["stable_to_unstable"].to_numpy()
    b = state["unstable_to_stable"].to_numpy()
    ax_b.bar(x, a, width=0.62, color="#C0392B", label="Stable → unstable")
    ax_b.bar(x, b, width=0.62, bottom=a, color="#2471A3", label="Unstable → stable")
    for index, total in enumerate(a + b):
        if index == 0:
            ax_b.text(index, total - max(a + b) * 0.018, f"{int(total):,}", ha="center", va="top", fontsize=10, color="#111111")
        else:
            ax_b.text(index, total + max(a + b) * 0.025, f"{int(total):,}", ha="center", va="bottom", fontsize=10, color="#111111")
    ax_b.set_xticks(x, [LABEL_NAMES[item] for item in state["label_definition"]], rotation=0)
    ax_b.set_ylabel("Panel units changing state label")
    ax_b.set_title("b  Direct 2025→2026 state-label evolution", loc="left", fontweight="bold")
    ax_b.legend(frameon=False, loc="upper right")
    ax_b.grid(axis="x", visible=False)

    delta_core = deltas[
        deltas["metric"].isin(core_metrics)
        & deltas["model_id"].isin(representatives)
        & deltas["budget"].fillna("").eq("")
    ].copy()
    delta_core["column"] = delta_core["label_definition"].map(LABEL_NAMES) + " · " + delta_core["model_id"]
    delta_core["row"] = delta_core["metric"].map(METRIC_NAMES)
    delta_matrix = delta_core.pivot(index="row", columns="column", values="delta_later_minus_earlier")
    row_order = [METRIC_NAMES[item] for item in core_metrics]
    column_order = [f"{LABEL_NAMES[label]} · {model}" for label in LABEL_NAMES for model in representatives]
    delta_matrix = delta_matrix.reindex(index=row_order, columns=column_order)
    vmax = float(np.nanmax(np.abs(delta_matrix.to_numpy()))) or 1.0
    sns.heatmap(
        delta_matrix,
        ax=ax_c,
        cmap=sns.diverging_palette(240, 15, as_cmap=True),
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".3f",
        annot_kws={"fontsize": 8.1, "color": "#111111"},
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Raw metric change (2026 − 2025)", "shrink": 0.75},
    )
    ax_c.set_xlabel("Label definition · fixed score vector")
    ax_c.set_ylabel("")
    ax_c.set_xticklabels(ax_c.get_xticklabels(), rotation=62, ha="right")
    ax_c.set_title("c  Metric values move while score bytes remain fixed", loc="left", fontweight="bold")

    rank_core = rankings[
        rankings["metric"].isin(core_metrics)
        & rankings["label_version"].isin(["2025-09-25", "2026-04-13"])
        & rankings["budget"].fillna("").eq("")
    ].copy()
    wide = rank_core.pivot_table(
        index=["metric", "label_definition", "model_id"],
        columns="label_version",
        values="rank",
        aggfunc="first",
    ).reset_index()
    wide["changed"] = wide["2025-09-25"].ne(wide["2026-04-13"])
    rank_counts = wide.groupby(["metric", "label_definition"])["changed"].sum().unstack()
    rank_counts = rank_counts.reindex(index=core_metrics, columns=list(LABEL_NAMES))
    rank_counts.index = [METRIC_NAMES[item] for item in rank_counts.index]
    rank_counts.columns = [LABEL_NAMES[item] for item in rank_counts.columns]
    sns.heatmap(
        rank_counts,
        ax=ax_d,
        cmap=sns.light_palette("#7D3C98", as_cmap=True),
        vmin=0,
        vmax=6,
        annot=True,
        fmt=".0f",
        annot_kws={"fontsize": 13, "fontweight": "bold", "color": "#111111"},
        linewidths=1.2,
        linecolor="white",
        cbar_kws={"label": "Changed rank records (of 6)", "ticks": range(7), "shrink": 0.75},
    )
    ax_d.set_xlabel("Registered label definition")
    ax_d.set_ylabel("")
    ax_d.set_title("d  Label updates reassign model ranks", loc="left", fontweight="bold")

    fig.suptitle(
        "The v2026.04.13 database release extends fixed-score benchmark nonstationarity",
        fontsize=16,
        fontweight="bold",
        color="#111111",
        y=1.015,
    )
    stem = output / "figure_r4_current_release_extension"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    source_rows = []
    for name, frame in (
        ("event_prevalence", prevalence),
        ("state_label_evolution", state),
        ("metric_deltas", delta_core),
        ("rank_change_counts", rank_counts.reset_index(names="metric")),
    ):
        item = frame.copy()
        item.insert(0, "panel", name)
        source_rows.append(item)
    source_path = output / "figure_r4_current_release_extension_source_data.csv"
    pd.concat(source_rows, ignore_index=True, sort=False).to_csv(source_path, index=False)
    exact = next(item for item in state.to_dict(orient="records") if item["label_definition"] == "exact_zero")
    caption = (
        "**Figure R4-1. The v2026.04.13 release provides an independent descriptive fifth-snapshot extension.** "
        "(a) Revision-event prevalence for the unchanged, identity-mapped comparable panel under three registered label definitions; the candidates and prediction-score bytes are fixed across label versions. "
        f"(b) Direct 2025→2026 state-label changes; under the exact-zero definition, {int(exact['state_label_changes']):,} panel units changed state label ({int(exact['stable_to_unstable']):,} stable-to-unstable and {int(exact['unstable_to_stable']):,} unstable-to-stable). "
        "(c) Raw 2026-minus-2025 changes in AP, Brier score, ROC AUC, and ECE for the four distinct frozen score vectors. "
        "(d) Counts of model-identifier rank records that change between 2025 and 2026. This post-amendment analysis measures database-version sensitivity; it is not an external or prospective model confirmation.\n"
    )
    caption_path = output / "figure_r4_current_release_extension_caption.md"
    caption_path.write_text(caption, encoding="utf-8")
    artifacts = []
    for path in [stem.with_suffix(ext) for ext in (".png", ".pdf", ".svg")] + [source_path, caption_path]:
        artifacts.append({"path": path.relative_to(repo).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "task_id": "R4.0",
        "status": "PASS",
        "figure_id": "R4-1",
        "source_files": [
            "data/processed/R4_0/versioned_labels_2024_2026.parquet",
            "reports/R4_0/fixed_score_metrics_2024_2026.csv",
            "reports/R4_0/fixed_score_metric_deltas_2025_2026.csv",
            "reports/R4_0/state_label_evolution_2025_2026.csv",
            "reports/R4_0/model_rankings_2024_2026.csv",
        ],
        "artifacts": artifacts,
        "external_confirmation_language": False,
    }
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
