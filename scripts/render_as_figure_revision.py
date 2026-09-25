from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "source_data" / "figures"
OUT_ROOT = ROOT / "outputs"
FIG_ROOT = OUT_ROOT / "figures"

BLACK = "#111111"
NAVY = "#6F9FC7"
BLUE = "#80B7E0"
TEAL = "#7CC9B5"
ORANGE = "#F3B36B"
RED = "#EE9585"
PLUM = "#B6A2D9"
GOLD = "#E8CA72"
PALE = "#E8F1F7"
LIGHT = "#F6FAFC"

LABELS = {
    "exact_zero": "Exact zero",
    "within_10meV": "Within 10 meV",
    "within_25meV": "Within 25 meV",
}
METRICS = {
    "average_precision": "Average precision",
    "brier_score": "Brier score",
    "roc_auc": "ROC AUC",
    "expected_calibration_error": "ECE",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.labelcolor": BLACK,
            "axes.edgecolor": BLACK,
            "axes.linewidth": 0.8,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.color": BLACK,
            "legend.fontsize": 7.6,
            "legend.frameon": False,
            "grid.color": PALE,
            "grid.linewidth": 0.65,
            "grid.alpha": 1.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def clean_axis(ax: plt.Axes, grid: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)
    if grid:
        ax.grid(True, axis=grid, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, letter: str, x: float = -0.13, y: float = 1.08) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def source_csv(number: int) -> Path:
    return SOURCE_ROOT / f"figure_{number}_source_data.csv"


def read(number: int) -> pd.DataFrame:
    return pd.read_csv(source_csv(number))


def export(fig: plt.Figure, number: int) -> dict[str, str]:
    out = FIG_ROOT / f"Figure_{number}"
    out.mkdir(parents=True, exist_ok=True)
    src = source_csv(number)
    copied = out / f"figure_{number}_source_data.csv"
    shutil.copy2(src, copied)
    paths = {
        "svg": out / f"figure_{number}.svg",
        "pdf": out / f"figure_{number}.pdf",
        "png": out / f"figure_{number}.png",
        "tiff": out / f"figure_{number}.tiff",
    }
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
    fig.savefig(paths["tiff"], dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return {
        "source_csv": str(copied.relative_to(ROOT)),
        "source_sha256": sha256(copied),
        **{k: str(v.relative_to(ROOT)) for k, v in paths.items()},
        **{f"{k}_sha256": sha256(v) for k, v in paths.items()},
    }


def figure_1() -> plt.Figure:
    d = read(1)
    fig = plt.figure(figsize=(7.2, 3.8), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.25], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    bx = fig.add_subplot(gs[0, 1])

    s = d[d.panel == "a"].copy()
    styles = {
        "all_workflows": ("Overall", NAVY, "-", "o"),
        "GGA_GGA+U": ("GGA/GGA+U", BLUE, "--", "s"),
        "GGA_GGA+U_R2SCAN": ("Mixed", TEAL, "-.", "^"),
        "R2SCAN": ("r²SCAN", ORANGE, ":", "D"),
    }
    for key, (label, color, ls, marker) in styles.items():
        g = s[s.stratum_value == key].sort_values("time_months")
        if g.empty:
            continue
        ax.fill_between(g.time_months, g.ci_lower, g.ci_upper, color=color, alpha=0.10, linewidth=0)
        ax.plot(g.time_months, g.survival_probability, color=color, ls=ls, marker=marker,
                ms=4.2, lw=1.8, label=label, zorder=3)
    ax.set_ylim(0.94, 1.001)
    ax.set_xlim(-0.5, 36)
    ax.set_xticks([0, 12, 24, 35])
    ax.set_xticklabels(["0", "12", "24", "35"])
    ax.set_xlabel("Months since first snapshot")
    ax.set_ylabel("Stable-label survival\n(truncated axis)")
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0.0, 0.02), columnspacing=0.9, handlelength=2.3)
    overall = s[s.stratum_value == "all_workflows"].sort_values("time_months").iloc[-1]
    ax.text(0.02, 0.72, f"Overall at 35 months\n{overall.survival_percent:.2f}%  (95% CI {overall.ci_lower_percent:.2f}–{overall.ci_upper_percent:.2f}%)",
            transform=ax.transAxes, fontsize=7.4, va="bottom", bbox=dict(boxstyle="round,pad=0.3", fc=LIGHT, ec="none"))
    risk = s[s.stratum_value == "all_workflows"].sort_values("time_months")
    ax.text(0.0, -0.19, "At risk", transform=ax.transAxes, fontsize=7.2, fontweight="bold", ha="left")
    for x, n in zip([0.02, 0.34, 0.67, 0.97], risk.n_at_risk.astype(int)):
        ax.text(x, -0.28, f"{n:,}", transform=ax.transAxes, fontsize=7.2, ha="center")
    clean_axis(ax, "y")
    panel_label(ax, "a")

    e = d[d.panel == "b"].copy()
    for direction, label, color, ls in [
        ("stable_to_unstable", "Stable → unstable", ORANGE, "-"),
        ("unstable_to_stable", "Unstable → stable", BLUE, "--"),
    ]:
        g = e[e.direction == direction].sort_values("amplitude_meV_per_atom")
        bx.plot(g.amplitude_meV_per_atom, g.cumulative_probability, color=color, lw=2.0, ls=ls, label=label)
    bx.set_xscale("log")
    bx.set_xlim(0.05, 1800)
    bx.set_ylim(0, 1.01)
    bx.set_xlabel("Absolute crossing amplitude (meV atom$^{-1}$)")
    bx.set_ylabel("Cumulative fraction")
    bx.legend(loc="upper left")
    t = d[d.panel == "b_thresholds"].set_index("threshold_meV_per_atom")
    for threshold, color in [(10.0, PLUM), (25.0, GOLD)]:
        row = t.loc[threshold]
        bx.axvline(threshold, color=color, lw=1.2, ls=(0, (3, 2)))
        bx.text(threshold * 1.08, 0.07 if threshold == 10 else 0.20,
                f">{int(threshold)} meV\n{row.surviving_flips:,.0f}/{row.exact_flip_rows:,.0f} ({row.surviving_fraction:.1%})",
                fontsize=7.3, color=BLACK)
    clean_axis(bx, "y")
    panel_label(bx, "b", -0.10)
    return fig


def figure_2() -> plt.Figure:
    d = read(2)
    fig = plt.figure(figsize=(7.2, 3.7), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[0.86, 1.34], wspace=0.10)
    ax = fig.add_subplot(gs[0, 0])
    bx = fig.add_subplot(gs[0, 1])

    g = d[d.panel == "a"].copy()
    order = sorted(g.release_pair.unique())
    y = np.arange(len(order))[::-1]
    for offset, etype, label, color, marker, fill in [
        (0.07, "raw_release_stratified", "Raw", PLUM, "o", "white"),
        (-0.07, "model_direct_standardization", "Standardized", NAVY, "s", NAVY),
    ]:
        h = g[g.estimate_type == etype].set_index("release_pair").loc[order]
        ax.errorbar(h.risk_percent, y + offset,
                    xerr=[h.risk_percent - h.ci_lower_percent, h.ci_upper_percent - h.risk_percent],
                    fmt=marker, ms=5, mfc=fill, mec=color, mew=1.2, color=color, ecolor=color,
                    capsize=2.5, lw=1.1, label=label, zorder=3)
    ax.set_yticks(y)
    pair_labels = []
    for pair in order:
        start, end = pair.split("->")
        pair_labels.append(f"{start[:4]}–{end[2:4]}")
    ax.set_yticklabels(pair_labels)
    ax.set_xlabel("Exact revision risk (%)")
    ax.legend(loc="upper right")
    ax.set_xlim(0.15, 1.28)
    for yy, pair in zip(y, order):
        val = g[(g.release_pair == pair) & (g.estimate_type == "model_direct_standardization")].iloc[0]
        ax.text(val.risk_percent + 0.04, yy - 0.07, f"{val.risk_percent:.2f}%", ha="left", va="center", fontsize=7.2)
    clean_axis(ax, "x")
    panel_label(ax, "a")

    m = d[d.panel == "b"].copy()
    for outcome, label, color, marker in [
        ("exact", "Exact", TEAL, "o"),
        ("10meV", "Within 10 meV", GOLD, "D"),
    ]:
        h = m[m.outcome == outcome].sort_values("median_margin_meV_per_atom")
        x = h.median_margin_meV_per_atom.to_numpy()
        yv = h.observed_risk.to_numpy() * 100
        lo = h.ci_lower.to_numpy() * 100
        hi = h.ci_upper.to_numpy() * 100
        bx.errorbar(x, yv, yerr=[yv - lo, hi - yv], color=color, marker=marker, ms=4.5,
                    lw=1.2, capsize=2.2, label=label)
    bx.set_xscale("log")
    bx.set_xlabel("Median source margin (meV atom$^{-1}$)")
    bx.set_ylabel("Observed revision risk (%)")
    bx.legend(loc="upper right")
    bx.text(0.02, 0.04, "2,000 whole-lineage bootstraps\n≈29,292 rows per exact-risk decile",
            transform=bx.transAxes, fontsize=7.2, va="bottom")
    clean_axis(bx, "y")
    panel_label(bx, "b", -0.09)
    return fig


def figure_3() -> plt.Figure:
    d = read(3)
    fig = plt.figure(figsize=(7.2, 3.55), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 0.95], wspace=0.10)
    ax = fig.add_subplot(gs[0, 0])
    bx = fig.add_subplot(gs[0, 1])
    names = {
        "competitor_inventory": "Competitor inventory",
        "uncorrected_energy": "Raw energy",
        "compatibility_correction": "Compatibility",
        "candidate_identity": "Identity",
    }
    colors = {
        "competitor_inventory": ORANGE,
        "uncorrected_energy": BLUE,
        "compatibility_correction": GOLD,
        "candidate_identity": PLUM,
    }
    order = ["competitor_inventory", "uncorrected_energy", "compatibility_correction", "candidate_identity"]
    h = d[d.panel == "b"].set_index("channel").loc[order]
    ypos = np.arange(4)[::-1]
    bars = ax.barh(ypos, h.largest_absolute_contribution_count, color=[colors[k] for k in order], height=0.56)
    ax.set_yticks(ypos, [names[k] for k in order])
    ax.set_xlabel("Audited flips assigned largest absolute contribution")
    for bar, (_, row) in zip(bars, h.iterrows()):
        ax.text(bar.get_width() + 45, bar.get_y() + bar.get_height() / 2,
                f"{int(row.largest_absolute_contribution_count):,}  ({row.share_percent_display})",
                va="center", fontsize=7.6, fontweight="bold" if row.name == "competitor_inventory" else "normal")
    ax.set_xlim(0, 3400)
    clean_axis(ax, "x")
    panel_label(ax, "a")

    a = d[(d.panel == "a") & (d.scale == "absolute")].set_index("channel").loc[order]
    for yy, key in zip(ypos, order):
        row = a.loc[key]
        bx.plot([row.q1, row.q3], [yy, yy], color=colors[key], lw=4, solid_capstyle="round", zorder=2)
        bx.plot(row["median"], yy, marker="|", color=BLACK, ms=12, mew=1.6, zorder=3)
        bx.scatter(row["mean"], yy, marker="D", s=32, facecolor="white", edgecolor=colors[key], lw=1.4, zorder=4)
    bx.set_xscale("symlog", linthresh=1, linscale=0.6)
    bx.set_xlim(-0.2, 1200)
    bx.set_yticks(ypos, [names[k] for k in order])
    bx.set_xlabel("Absolute contribution (meV atom$^{-1}$)")
    bx.text(0.02, -0.22, "thick line: IQR   |: median   open diamond: mean\nn = 3,113 flips per channel",
            transform=bx.transAxes, fontsize=7.1, va="top")
    clean_axis(bx, "x")
    panel_label(bx, "b", -0.11)
    fig.text(0.5, -0.025, "Declared accounting contribution; not a physical-cause estimate",
             ha="center", fontsize=7.5, fontweight="bold")
    return fig


def figure_4() -> plt.Figure:
    d = read(4)
    fig = plt.figure(figsize=(7.2, 4.1), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.58], width_ratios=[0.82, 1.18], hspace=0.12, wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    bx = fig.add_subplot(gs[0, 1])
    cx = fig.add_subplot(gs[1, :])

    f = d[d.panel == "a_counts"].sort_values("stage_order")
    stage_labels = ["Strict 10 meV\nrevisions", "With registered\nA omissions", "Reversed below\n10 meV"]
    vals = f["count"].astype(int).to_numpy()
    yp = np.arange(3)[::-1]
    bars = ax.barh(yp, vals, color=[NAVY, TEAL, ORANGE], height=0.52)
    ax.set_yticks(yp, stage_labels)
    ax.set_xlabel("Transitions")
    for b, v in zip(bars, vals):
        ax.text(v + 24, b.get_y() + b.get_height() / 2, f"{v:,}", va="center", fontsize=8, fontweight="bold")
    ax.set_xlim(0, 1320)
    clean_axis(ax, "x")
    panel_label(ax, "a")

    q = d[(d.panel == "a") & (d.scope == "A_ONLY")].sort_values("threshold_meV_per_atom")
    bx.plot(q.threshold_meV_per_atom, q.reversal_fraction_primary * 100, color=PLUM, marker="o", lw=1.8,
            label="All 1,190 primary transitions")
    bx.plot(q.threshold_meV_per_atom, q.reversal_fraction_threshold_eligible * 100, color=ORANGE, marker="D", lw=1.8,
            label="Threshold-eligible transitions")
    bx.set_xticks(q.threshold_meV_per_atom.astype(int))
    bx.set_ylim(25, 54)
    bx.set_xlabel("Registered threshold (meV atom$^{-1}$)")
    bx.set_ylabel("Reversal fraction (%)")
    bx.legend(loc="lower left")
    r25 = q[q.threshold_meV_per_atom == 25].iloc[0]
    bx.annotate(f"{int(r25.reversed_transitions)}/{int(r25.threshold_eligible_transitions)} = {r25.reversal_fraction_threshold_eligible:.1%}",
                xy=(25, r25.reversal_fraction_threshold_eligible * 100), xytext=(14.5, 48.0),
                arrowprops=dict(arrowstyle="-", color=BLACK, lw=0.8), fontsize=7.3)
    clean_axis(bx, "y")
    panel_label(bx, "b", -0.10)

    diag = d[d.panel == "b"].iloc[0]
    gate = float(diag.frozen_maximum_absolute_smd_gate)
    observed = float(diag.maximum_absolute_smd)
    cx.plot([0.1, observed], [0, 0], color=RED, lw=5, solid_capstyle="round")
    cx.scatter([observed], [0], s=55, color=RED, edgecolor="white", linewidth=0.8, zorder=3)
    cx.axvline(gate, color=BLACK, ls="--", lw=1.2)
    cx.text(observed * 1.04, 0, f"observed max |SMD| = {observed:.3f}", va="center", fontsize=8, fontweight="bold")
    cx.text(gate, -0.30, f"prespecified gate = {gate:.3f}", ha="center", fontsize=7.3)
    cx.set_xscale("log")
    cx.set_xlim(0.1, 4.0)
    cx.set_ylim(-0.6, 0.6)
    cx.set_yticks([])
    cx.set_xlabel("Matched-design balance diagnostic (absolute standardized mean difference)")
    clean_axis(cx, "x")
    panel_label(cx, "c", -0.06, 1.18)
    fig.text(0.5, -0.02, "Frozen database-state counterfactual; not a physical-causality estimate",
             ha="center", fontsize=7.5, fontweight="bold")
    return fig


def figure_5() -> plt.Figure:
    d = read(5)
    fig = plt.figure(figsize=(7.2, 4.9), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], width_ratios=[0.88, 1.12], hspace=0.12, wspace=0.11)
    ax = fig.add_subplot(gs[0, 0])
    bx = fig.add_subplot(gs[0, 1])
    cx = fig.add_subplot(gs[1, :])

    colors = {"exact_zero": NAVY, "within_10meV": ORANGE, "within_25meV": TEAL}
    markers = {"exact_zero": "o", "within_10meV": "D", "within_25meV": "s"}
    p = d[d.panel == "a"].copy()
    for definition in LABELS:
        g = p[p.label_definition == definition].sort_values("label_version")
        x = np.arange(2)
        ax.plot(x, g.prevalence_percent, color=colors[definition], marker=markers[definition], ms=5, lw=1.8,
                label=LABELS[definition])
        for xx, (_, row) in zip(x, g.iterrows()):
            ax.text(xx, row.prevalence_percent + 0.07, f"{int(row.events)}\n{row.prevalence_percent:.2f}%",
                    ha="center", fontsize=7.0)
    ax.set_xticks([0, 1], ["2024", "2025"])
    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(0, 2.75)
    ax.set_ylabel("Label-event prevalence (%)")
    ax.legend(loc="upper left")
    ax.text(0.02, 0.03, "n = 13,783; candidates and scores fixed", transform=ax.transAxes, fontsize=7.2)
    clean_axis(ax, "y")
    panel_label(ax, "a")

    registry = d[d.panel == "model_registry"]
    reps = registry[registry.plot_representative == True].sort_values("plot_order").model_id.tolist()  # noqa: E712
    m = d[(d.panel == "b") & d.model_id.isin(reps)].copy()
    metric_order = list(METRICS)
    def_order = list(LABELS)
    offsets = np.linspace(-0.24, 0.24, len(reps))
    for mi, metric in enumerate(metric_order):
        z = m[m.metric == metric]
        for di, definition in enumerate(def_order):
            zz = z[z.label_definition == definition]
            piv = zz.pivot(index="model_id", columns="label_version", values="value").reindex(reps)
            delta = (piv["2025-09-25"] - piv["2024-12-18"]) * 100
            x0 = mi + (di - 1) * 0.08
            bx.scatter(x0 + offsets * 0.18, delta, s=22, marker=markers[definition],
                       facecolor=colors[definition], edgecolor="white", linewidth=0.5, zorder=3,
                       label=LABELS[definition] if mi == 0 and len(bx.collections) < 3 else None)
    bx.axhline(0, color=BLACK, lw=0.8)
    bx.set_xticks(range(4), ["AP", "Brier", "ROC AUC", "ECE"])
    bx.set_ylabel("2025 − 2024 change\n(percentage points)")
    handles = [mpl.lines.Line2D([], [], marker=markers[k], color="none", markerfacecolor=colors[k],
                               markeredgecolor=colors[k], label=LABELS[k], markersize=5) for k in def_order]
    bx.legend(handles=handles, ncol=3, loc="upper left", handletextpad=0.3, columnspacing=0.7)
    clean_axis(bx, "y")
    panel_label(bx, "b", -0.11)

    r = d[d.panel == "c"].copy()
    counts = np.zeros((4, 3), dtype=int)
    totals = np.zeros((4, 3), dtype=int)
    for i, metric in enumerate(metric_order):
        for j, definition in enumerate(def_order):
            z = r[(r.metric == metric) & (r.label_definition == definition)]
            piv = z.pivot(index="model_id", columns="label_version", values="rank")
            counts[i, j] = int((piv["2025-09-25"] != piv["2024-12-18"]).sum())
            totals[i, j] = len(piv)
    row_totals = counts.sum(axis=1)
    row_denoms = totals.sum(axis=1)
    col_totals = counts.sum(axis=0)
    col_denoms = totals.sum(axis=0)
    cell_text = []
    for i, metric in enumerate(metric_order):
        cell_text.append(
            [METRICS[metric]]
            + [f"{counts[i, j]}/{totals[i, j]}" for j in range(3)]
            + [f"{row_totals[i]}/{row_denoms[i]}"]
        )
    cell_text.append(
        ["All metrics"]
        + [f"{col_totals[j]}/{col_denoms[j]}" for j in range(3)]
        + [f"{counts.sum()}/{totals.sum()}"]
    )
    cx.axis("off")
    table = cx.table(
        cellText=cell_text,
        colLabels=["Metric", "Exact zero", "Within 10 meV", "Within 25 meV", "Total"],
        colWidths=[0.27, 0.17, 0.19, 0.19, 0.14],
        cellLoc="center",
        loc="center",
        bbox=[0.02, 0.06, 0.96, 0.86],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.3)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(1.2)
        cell.set_text_props(color=BLACK)
        if row == 0:
            cell.set_facecolor(PALE)
            cell.set_text_props(fontweight="bold", color=BLACK)
        elif row == len(cell_text):
            cell.set_facecolor("#DDEEF4")
            cell.set_text_props(fontweight="bold", color=BLACK)
        elif col == 0:
            cell.set_facecolor(LIGHT)
            cell.set_text_props(ha="left", fontweight="bold", color=BLACK)
        else:
            cell.set_facecolor("#F9FCFD" if row % 2 else "#EEF7F8")
    table.scale(1.0, 1.35)
    panel_label(cx, "c", -0.01, 0.98)
    return fig


def figure_6() -> plt.Figure:
    d = read(6)
    fig = plt.figure(figsize=(7.2, 5.15), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.2], width_ratios=[1.05, 0.95], hspace=0.10, wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    bx = fig.add_subplot(gs[0, 1])
    cx = fig.add_subplot(gs[1, 0])
    dx = fig.add_subplot(gs[1, 1])

    colors = {"exact_zero": NAVY, "within_10meV": ORANGE, "within_25meV": TEAL}
    markers = {"exact_zero": "o", "within_10meV": "D", "within_25meV": "s"}
    p = d[d.panel == "event_prevalence"]
    versions = ["2024-12-18", "2025-09-25", "2026-04-13"]
    for definition in LABELS:
        g = p[p.label_definition == definition].set_index("label_version").loc[versions]
        ax.plot(range(3), g.prevalence_percent, color=colors[definition], marker=markers[definition], ms=4.5,
                lw=1.8, label=LABELS[definition])
        ax.text(2.05, g.prevalence_percent.iloc[-1], f"{int(g.events.iloc[-1])}  ({g.prevalence_percent.iloc[-1]:.2f}%)",
                va="center", fontsize=7.0, color=BLACK)
    ax.set_xticks(range(3), ["2024", "2025", "2026"])
    ax.set_xlim(-0.15, 2.55)
    ax.set_ylabel("Label-event prevalence (%)")
    ax.legend(loc="upper left")
    clean_axis(ax, "y")
    panel_label(ax, "a")

    e = d[d.panel == "state_label_evolution"].set_index("label_definition").loc[list(LABELS)]
    yy = np.arange(3)[::-1]
    left = -e.stable_to_unstable.to_numpy()
    right = e.unstable_to_stable.to_numpy()
    bx.barh(yy, left, color=RED, height=0.55, label="Stable → unstable")
    bx.barh(yy, right, color=BLUE, height=0.55, label="Unstable → stable")
    for yv, l, r in zip(yy, e.stable_to_unstable.astype(int), e.unstable_to_stable.astype(int)):
        bx.text(-l - 8, yv, f"{l}", ha="right", va="center", fontsize=7.5, fontweight="bold")
        bx.text(r + 8, yv, f"{r}", ha="left", va="center", fontsize=7.5, fontweight="bold")
    bx.axvline(0, color=BLACK, lw=0.8)
    bx.set_yticks(yy, [LABELS[k] for k in LABELS])
    bx.set_xlabel("")
    bx.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.10), columnspacing=0.8, handlelength=1.5)
    clean_axis(bx, "x")
    panel_label(bx, "b", -0.12)

    m = d[d.panel == "metric_deltas"].copy()
    reps = [x for x in ["M0", "M1", "M2", "M5"] if x in set(m.model_id.dropna())]
    metric_order = list(METRICS)
    def_order = list(LABELS)
    columns = [(definition, model) for definition in def_order for model in reps]
    mat = np.full((4, len(columns)), np.nan)
    for i, metric in enumerate(metric_order):
        for j, (definition, model) in enumerate(columns):
            z = m[(m.metric == metric) & (m.label_definition == definition) & (m.model_id == model)]
            if not z.empty:
                mat[i, j] = float(z.delta_later_minus_earlier.iloc[0]) * 100
    vmax = np.nanmax(np.abs(mat))
    cmap = LinearSegmentedColormap.from_list("delta", ["#B8DDF3", "#FFFFFF", "#F7C5BA"])
    im = cx.imshow(mat, cmap=cmap, norm=Normalize(vmin=-vmax, vmax=vmax), aspect="auto")
    cx.set_yticks(range(4), ["AP", "Brier", "ROC AUC", "ECE"])
    cx.set_xticks(range(len(columns)), [model for _, model in columns], rotation=0)
    for split in [len(reps) - 0.5, 2 * len(reps) - 0.5]:
        cx.axvline(split, color="white", lw=2.0)
    for j, definition in enumerate(def_order):
        center = j * len(reps) + (len(reps) - 1) / 2
        cx.text(center, 4.35, ["Exact", "≤10 meV", "≤25 meV"][j], ha="center", fontsize=7.4, fontweight="bold")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                cx.text(j, i, f"{mat[i, j]:+.1f}", ha="center", va="center", fontsize=5.9, color=BLACK)
    cx.set_ylim(4.65, -0.5)
    cx.set_xlabel("Score vectors grouped by registered label definition")
    cx.tick_params(length=0)
    for spine in cx.spines.values():
        spine.set_visible(False)
    panel_label(cx, "c", -0.10, 1.08)

    r = d[d.panel == "rank_change_counts"].copy()
    rank_metrics = ["Average precision", "Brier score", "ROC AUC", "ECE"]
    rank_defs = ["Exact zero", "Within 10 meV", "Within 25 meV"]
    vals = r.set_index("metric").loc[rank_metrics, rank_defs].to_numpy(float)
    for i in range(4):
        for j in range(3):
            size = 42 + vals[i, j] * 34
            dx.scatter(j, i, s=size, color=NAVY if vals[i, j] > 0 else "white", edgecolor=NAVY, lw=1.0)
            dx.text(j, i, f"{int(vals[i, j])}", ha="center", va="center",
                    color=BLACK, fontsize=7.2, fontweight="bold")
    dx.set_xlim(-0.55, 2.55)
    dx.set_ylim(3.55, -0.55)
    dx.set_xticks(range(3), ["Exact", "≤10 meV", "≤25 meV"])
    dx.set_yticks(range(4), ["AP", "Brier", "ROC AUC", "ECE"])
    dx.set_xlabel("Registered label definition")
    dx.grid(True, color=PALE, lw=0.7)
    dx.tick_params(length=0)
    for spine in dx.spines.values():
        spine.set_visible(False)
    panel_label(dx, "d", -0.12)
    return fig


def main() -> None:
    configure_style()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "status": "PASTEL_PALETTE_AND_PANEL_C_TABLE_REVISION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(SOURCE_ROOT.relative_to(ROOT)),
        "output_root": str(OUT_ROOT.relative_to(ROOT)),
        "backend": "Python/Matplotlib",
        "scientific_changes": False,
        "frozen_results_overwritten": False,
        "figures": {},
    }
    for number, builder in enumerate([figure_1, figure_2, figure_3, figure_4, figure_5, figure_6], start=1):
        manifest["figures"][str(number)] = export(builder(), number)
    manifest_path = OUT_ROOT / "figure_revision_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
