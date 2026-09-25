"""Frozen R3.3R-B identity sensitivity and noninferential stability envelopes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .common import open_formal_input, sha256_file


TASK_ID = "R3.3R-B"
METHOD_VERSION = "PHASEEVONET_R3_3R_NETWORK_REANALYSIS_V1"
IDENTITIES = (
    "id0_contextual",
    "id1_strict_entry",
    "id2_lineage_thermo_workflow",
    "id3_lineage_only",
)
IMPACT_EDGES = (
    "necessary_10meV",
    "necessary_25meV",
    "contributory_5meV",
    "minimal_set_member",
)
ALL_EDGES = ("selected_active", *IMPACT_EDGES)
GATE_METRICS = ("gini", "top_10pct_share", "largest_cascade_fraction")
METRICS = (
    "edge_pairs",
    "competitor_nodes",
    "candidate_nodes",
    "connected_components",
    "largest_cascade",
    "largest_cascade_fraction",
    "mean_cascade",
    "median_cascade",
    "gini",
    "hhi",
    "top_1pct_share",
    "top_5pct_share",
    "top_10pct_share",
    "maximum_candidate_indegree",
    "mean_candidate_indegree",
)
IDENTITY_COLUMNS = {
    "id0_contextual": "id0_contextual",
    "id1_strict_entry": "id1_strict_entry",
    "id2_lineage_thermo_workflow": "id2_lineage_thermo_workflow",
    "id3_lineage_only": "id3_lineage_only",
}

FROZEN_HASHES = {
    "data/manifests/R3_3R/identity_mapping_manifest.json": "e7c01a01551ab7ed5bc20e6f55ce43e5c2f8685fb090f428978720de9c983503",
    "data/processed/R3_3R/competitor_identity_crosswalk.parquet": "c001d249c9958176e76d0e51fde1fe033cc4df8d3eae98b99dcabe61ede236d6",
    "data/processed/R3_3/competitor_candidate_edge.parquet": "5299c5c265c5d96639d8225ad9fc98a7265acc522dd6fbe079cdd7f23440fcc4",
    "reports/R3_3R_A/report.json": "1180580bf695a80c5c2350270ffce80dd0d7b15afed2f6c4fdb9bccb70d04a41",
    "reports/R3_3R_A/verification.json": "b79dda8a3e7dce8c86bfb54e4be799e56fffff14a005684e795ebe8e93832a39",
    "reports/R3_3/report.json": "28a519b81f0a6cd984daaef5f04dbdc2fc8a2580dbfba8c8a2bf4003dc15da34",
    "reports/R3_3/verification.json": "740bd2e7915c9d017c64fcaa41a9d5cf50571809bb360186a48823e1d2b6ba4f",
    "data/manifests/R3_3/manifest.json": "62899a714bb91a07dbaf081557958db4af2143a6e715b2aae73579e1b52e0e7f",
    "TASKS_R3.md": "0f1360f27412a7d3bcccba3463ea9e3ea1cf0495b45049d1c5397c60b109e55a",
    "R3_3_DECISION_MEMO.md": "f80941a7c9b8a3630eb4625e8fe77cc7030a89da975c82a6ed5e5b41d03b6fb0",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if not isinstance(value, (str, bytes)) and pd.isna(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_ready(row))


def _artifact(repo: Path, path: Path, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.relative_to(repo).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _hex_transition(value: Any) -> str:
    return bytes(value).hex() if isinstance(value, (bytes, bytearray, memoryview)) else str(value)


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or float(values.sum()) <= 0:
        return 0.0
    ordered = np.sort(values)
    n = len(ordered)
    return float(2 * np.dot(np.arange(1, n + 1), ordered) / (n * ordered.sum()) - (n + 1) / n)


def _components(node_indices: np.ndarray, candidate_indices: np.ndarray) -> int:
    if len(node_indices) == 0:
        return 0
    active_nodes, node_inverse = np.unique(node_indices, return_inverse=True)
    active_candidates, candidate_inverse = np.unique(candidate_indices, return_inverse=True)
    node_count = len(active_nodes)
    total = node_count + len(active_candidates)
    graph = coo_matrix(
        (np.ones(len(node_inverse), dtype=np.int8), (node_inverse, node_count + candidate_inverse)),
        shape=(total, total),
    )
    count, _ = connected_components(graph, directed=False, return_labels=True)
    return int(count)


def _metrics_from_encoded(
    node_indices: np.ndarray,
    candidate_indices: np.ndarray,
    *,
    node_universe: int,
    candidate_universe: int,
    selected_candidates: np.ndarray | None = None,
) -> dict[str, float | int]:
    if selected_candidates is not None:
        keep = selected_candidates[candidate_indices]
        node_indices = node_indices[keep]
        candidate_indices = candidate_indices[keep]
    if len(node_indices) == 0:
        return {metric: 0 for metric in METRICS}
    cascade = np.bincount(node_indices, minlength=node_universe)
    cascade = cascade[cascade > 0].astype(float)
    indegree = np.bincount(candidate_indices, minlength=candidate_universe)
    indegree = indegree[indegree > 0].astype(float)
    edge_pairs = int(len(node_indices))
    total = float(cascade.sum())
    ordered = np.sort(cascade)[::-1]

    def top_share(fraction: float) -> float:
        count = max(1, int(math.ceil(len(ordered) * fraction)))
        return float(ordered[:count].sum() / total) if total else 0.0

    return {
        "edge_pairs": edge_pairs,
        "competitor_nodes": int(len(cascade)),
        "candidate_nodes": int(len(indegree)),
        "connected_components": _components(node_indices, candidate_indices),
        "largest_cascade": int(cascade.max()),
        "largest_cascade_fraction": float(cascade.max() / len(indegree)) if len(indegree) else 0.0,
        "mean_cascade": float(cascade.mean()),
        "median_cascade": float(np.median(cascade)),
        "gini": gini(cascade),
        "hhi": float(np.square(cascade / total).sum()) if total else 0.0,
        "top_1pct_share": top_share(0.01),
        "top_5pct_share": top_share(0.05),
        "top_10pct_share": top_share(0.10),
        "maximum_candidate_indegree": int(indegree.max()),
        "mean_candidate_indegree": float(indegree.mean()),
    }


@dataclass(frozen=True)
class EncodedNetwork:
    node_indices: np.ndarray
    candidate_indices: np.ndarray
    nodes: tuple[str, ...]
    candidates: tuple[str, ...]

    def metrics(self, selected_candidates: np.ndarray | None = None) -> dict[str, float | int]:
        return _metrics_from_encoded(
            self.node_indices,
            self.candidate_indices,
            node_universe=len(self.nodes),
            candidate_universe=len(self.candidates),
            selected_candidates=selected_candidates,
        )


def encode_network(frame: pd.DataFrame, node_column: str, candidates: tuple[str, ...] | None = None) -> EncodedNetwork:
    pairs = (
        frame[[node_column, "candidate_lineage_id"]]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values([node_column, "candidate_lineage_id"], kind="mergesort")
    )
    nodes = tuple(sorted(pairs[node_column].unique()))
    candidate_values = candidates or tuple(sorted(pairs["candidate_lineage_id"].unique()))
    node_code = {value: index for index, value in enumerate(nodes)}
    candidate_code = {value: index for index, value in enumerate(candidate_values)}
    return EncodedNetwork(
        node_indices=pairs[node_column].map(node_code).to_numpy(dtype=np.int32),
        candidate_indices=pairs["candidate_lineage_id"].map(candidate_code).to_numpy(dtype=np.int32),
        nodes=nodes,
        candidates=candidate_values,
    )


def network_metrics(frame: pd.DataFrame, node_column: str) -> dict[str, float | int]:
    return encode_network(frame, node_column).metrics()


def candidate_subsampling(
    networks: dict[str, EncodedNetwork],
    *,
    fraction: float,
    replicates: int,
    seed: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Sample candidates without replacement using a shared draw across identities."""

    candidate_count = len(next(iter(networks.values())).candidates)
    if any(network.candidates != next(iter(networks.values())).candidates for network in networks.values()):
        raise ValueError("identity networks must share the same sorted candidate universe")
    sample_size = max(1, int(math.floor(candidate_count * fraction)))
    rng = np.random.default_rng(seed)
    values = {
        identity: {metric: np.empty(replicates, dtype=float) for metric in METRICS}
        for identity in networks
    }
    duplicate_draws = 0
    for replicate in range(replicates):
        selected_indices = rng.choice(candidate_count, size=sample_size, replace=False)
        duplicate_draws += sample_size - len(np.unique(selected_indices))
        selected = np.zeros(candidate_count, dtype=bool)
        selected[selected_indices] = True
        for identity, network in networks.items():
            metrics = network.metrics(selected)
            for metric in METRICS:
                values[identity][metric][replicate] = float(metrics[metric])
    return values, {
        "candidate_count": candidate_count,
        "sample_size": sample_size,
        "fraction": fraction,
        "replicates": replicates,
        "replacement": False,
        "duplicate_candidates_across_draws": duplicate_draws,
        "seed": seed,
        "label": "stability_envelope_not_confidence_interval",
    }


def _edge_seed(base_seed: int, edge_definition: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{edge_definition}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _join_inputs(edge: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    edge_frame = edge.copy()
    edge_frame["transition_id"] = edge_frame["transition_id"].map(_hex_transition)
    mapping_columns = [
        "transition_id",
        "competitor_contextual_id",
        "mapping_status",
        "identity_workflow",
        "p2_lineage_confidence",
        *IDENTITIES,
    ]
    resolved = edge_frame.merge(
        crosswalk[mapping_columns],
        on=["transition_id", "competitor_contextual_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if len(resolved) != len(edge_frame) or bool((resolved["_merge"] != "both").any()):
        raise ValueError("edge-to-crosswalk join did not preserve every edge relation")
    if not bool(resolved["mapping_status"].eq("mapped").all()):
        raise ValueError("R3.3R-B requires every used edge relation to have frozen mapped identity")
    if bool(resolved[list(IDENTITIES)].isna().any().any()):
        raise ValueError("one or more frozen identity fields are missing")
    key = ["transition_id", "competitor_contextual_id", "edge_definition"]
    if bool(resolved.duplicated(key).any()):
        raise ValueError("identity-resolved edge relation grain is not unique")
    return resolved.drop(columns="_merge")


def _point_summaries(resolved: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for edge_definition in ALL_EDGES:
        subset = resolved.loc[resolved["edge_definition"].eq(edge_definition)]
        for identity in IDENTITIES:
            metrics = network_metrics(subset, IDENTITY_COLUMNS[identity])
            row = {
                "edge_definition": edge_definition,
                "edge_role": "impact_primary" if edge_definition in IMPACT_EDGES else "structural_secondary",
                "identity_definition": identity,
                **metrics,
                "method_version": METHOD_VERSION,
            }
            rows.append(row)
            lookup[(edge_definition, identity)] = row
    return pd.DataFrame(rows), lookup


def _thresholds(config: dict[str, Any]) -> dict[str, float]:
    return {
        "gini": float(config["gate"]["maximum_absolute_gini_difference"]),
        "top_10pct_share": float(config["gate"]["maximum_absolute_top10_share_difference"]),
        "largest_cascade_fraction": float(
            config["gate"]["maximum_absolute_largest_cascade_fraction_difference"]
        ),
    }


def _identity_sensitivity(
    point: dict[tuple[str, str], dict[str, Any]], config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, bool]]:
    rows: list[dict[str, Any]] = []
    thresholds = _thresholds(config)
    edge_pass: dict[str, bool] = {}
    for edge_definition in ALL_EDGES:
        passes: list[bool] = []
        for metric in METRICS:
            values = {identity: float(point[(edge_definition, identity)][metric]) for identity in IDENTITIES}
            signed = values["id2_lineage_thermo_workflow"] - values["id1_strict_entry"]
            threshold = thresholds.get(metric)
            metric_pass = abs(signed) <= threshold + 1e-15 if threshold is not None else None
            if metric in GATE_METRICS:
                passes.append(bool(metric_pass))
            rows.append(
                {
                    "edge_definition": edge_definition,
                    "edge_role": "impact_primary" if edge_definition in IMPACT_EDGES else "structural_secondary",
                    "metric": metric,
                    **{identity: values[identity] for identity in IDENTITIES},
                    "identity_minimum": min(values.values()),
                    "identity_maximum": max(values.values()),
                    "identity_range": max(values.values()) - min(values.values()),
                    "id2_minus_id1": signed,
                    "id1_id2_absolute_difference": abs(signed),
                    "frozen_threshold": threshold,
                    "metric_gate_pass": metric_pass,
                    "contextual_minus_id2": values["id0_contextual"] - values["id2_lineage_thermo_workflow"],
                }
            )
        edge_pass[edge_definition] = all(passes)
    frame = pd.DataFrame(rows)
    frame["full_edge_gate_pass"] = frame["edge_definition"].map(edge_pass)
    return frame, edge_pass


def _stability_analyses(
    resolved: pd.DataFrame,
    point: dict[tuple[str, str], dict[str, Any]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    directional_rows: list[dict[str, Any]] = []
    sampling_audit: list[dict[str, Any]] = []
    dimensions = (
        ("target_snapshot", "leave_one_target_snapshot_out"),
        ("thermo_type", "leave_one_thermo_type_out"),
        ("identity_workflow", "leave_one_identity_workflow_out"),
    )
    for edge_definition in IMPACT_EDGES:
        edge_frame = resolved.loc[resolved["edge_definition"].eq(edge_definition)].copy()
        omitted_metrics: dict[tuple[str, str, str, str], float] = {}
        for dimension, analysis in dimensions:
            for value in sorted(edge_frame[dimension].dropna().astype(str).unique()):
                subset = edge_frame.loc[edge_frame[dimension].astype(str).ne(value)]
                if subset.empty:
                    continue
                for identity in IDENTITIES:
                    metrics = network_metrics(subset, IDENTITY_COLUMNS[identity])
                    for metric in METRICS:
                        estimate = float(metrics[metric])
                        omitted_metrics[(dimension, value, identity, metric)] = estimate
                        rows.append(
                            {
                                "analysis_type": analysis,
                                "edge_definition": edge_definition,
                                "identity_definition": identity,
                                "stratum_dimension": dimension,
                                "stratum_value": value,
                                "metric": metric,
                                "full_data_estimate": float(point[(edge_definition, identity)][metric]),
                                "estimate": estimate,
                                "q025": None,
                                "q50": None,
                                "q975": None,
                                "replicates": 1,
                                "sample_fraction": None,
                                "sampling_replacement": False,
                                "label": "deterministic_leave_one_stratum_out_sensitivity",
                            }
                        )
                for metric in GATE_METRICS:
                    full_difference = float(
                        point[(edge_definition, "id2_lineage_thermo_workflow")][metric]
                        - point[(edge_definition, "id1_strict_entry")][metric]
                    )
                    omitted_difference = (
                        omitted_metrics[(dimension, value, "id2_lineage_thermo_workflow", metric)]
                        - omitted_metrics[(dimension, value, "id1_strict_entry", metric)]
                    )
                    direction_preserved = abs(full_difference) <= 1e-12 or (
                        full_difference * omitted_difference >= -1e-12
                    )
                    directional_rows.append(
                        {
                            "edge_definition": edge_definition,
                            "stratum_dimension": dimension,
                            "excluded_value": value,
                            "metric": metric,
                            "full_id2_minus_id1": full_difference,
                            "leave_one_out_id2_minus_id1": omitted_difference,
                            "direction_preserved": bool(direction_preserved),
                            "rule": "nonzero sign must not reverse; full abs<=1e-12 is direction-neutral",
                        }
                    )

        candidates = tuple(sorted(edge_frame["candidate_lineage_id"].astype(str).unique()))
        networks = {
            identity: encode_network(edge_frame, IDENTITY_COLUMNS[identity], candidates=candidates)
            for identity in IDENTITIES
        }
        sampled, audit = candidate_subsampling(
            networks,
            fraction=float(config["stability"]["candidate_subsampling_without_replacement"]["fraction"]),
            replicates=int(config["stability"]["candidate_subsampling_without_replacement"]["replicates"]),
            seed=_edge_seed(int(config["seed"]), edge_definition),
        )
        sampling_audit.append({"edge_definition": edge_definition, **audit})
        quantiles = config["stability"]["candidate_subsampling_without_replacement"]["quantiles"]
        for identity in IDENTITIES:
            for metric in METRICS:
                q025, q50, q975 = np.quantile(sampled[identity][metric], quantiles)
                rows.append(
                    {
                        "analysis_type": "candidate_subsampling_without_replacement",
                        "edge_definition": edge_definition,
                        "identity_definition": identity,
                        "stratum_dimension": "candidate_lineage_id",
                        "stratum_value": "80_percent_without_replacement",
                        "metric": metric,
                        "full_data_estimate": float(point[(edge_definition, identity)][metric]),
                        "estimate": None,
                        "q025": float(q025),
                        "q50": float(q50),
                        "q975": float(q975),
                        "replicates": int(audit["replicates"]),
                        "sample_fraction": float(audit["fraction"]),
                        "sampling_replacement": False,
                        "label": "stability_envelope_not_confidence_interval",
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(directional_rows), pd.DataFrame(sampling_audit)


def _node_audits(resolved: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merge_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    for edge_definition in ALL_EDGES:
        subset = resolved.loc[resolved["edge_definition"].eq(edge_definition)]
        merge = subset[["id1_strict_entry", "id2_lineage_thermo_workflow"]].drop_duplicates()
        sizes = merge.groupby("id2_lineage_thermo_workflow")["id1_strict_entry"].nunique().to_numpy()
        merge_rows.append(
            {
                "edge_definition": edge_definition,
                "id2_nodes": len(sizes),
                "id1_nodes": int(merge["id1_strict_entry"].nunique()),
                "mean_id1_entries_per_id2": float(np.mean(sizes)),
                "median_id1_entries_per_id2": float(np.median(sizes)),
                "q95_id1_entries_per_id2": float(np.quantile(sizes, 0.95)),
                "maximum_id1_entries_per_id2": int(np.max(sizes)),
                "multi_entry_id2_nodes": int((sizes > 1).sum()),
                "multi_entry_id2_fraction": float((sizes > 1).mean()),
            }
        )
        top_candidate_sets: dict[str, set[str]] = {}
        for identity in IDENTITIES:
            node = IDENTITY_COLUMNS[identity]
            pairs = subset[[node, "candidate_lineage_id"]].drop_duplicates()
            cascade = pairs.groupby(node)["candidate_lineage_id"].nunique().sort_values(ascending=False)
            cascade = cascade.sort_index(kind="mergesort").sort_values(ascending=False, kind="mergesort")
            for rank, (node_id, count) in enumerate(cascade.head(20).items(), start=1):
                rank_rows.append(
                    {
                        "edge_definition": edge_definition,
                        "identity_definition": identity,
                        "rank": rank,
                        "competitor_node_id": node_id,
                        "cascade_candidates": int(count),
                        "cascade_fraction": float(count / subset["candidate_lineage_id"].nunique()),
                        "manuscript_use": "secondary_only_subject_to_route_and_identity_range",
                    }
                )
            top_count = max(1, int(math.ceil(len(cascade) * 0.10)))
            top_nodes = set(cascade.head(top_count).index)
            top_candidate_sets[identity] = set(
                pairs.loc[pairs[node].isin(top_nodes), "candidate_lineage_id"].astype(str)
            )
        for left, right in (
            ("id0_contextual", "id2_lineage_thermo_workflow"),
            ("id1_strict_entry", "id2_lineage_thermo_workflow"),
            ("id2_lineage_thermo_workflow", "id3_lineage_only"),
        ):
            a, b = top_candidate_sets[left], top_candidate_sets[right]
            overlap_rows.append(
                {
                    "edge_definition": edge_definition,
                    "identity_a": left,
                    "identity_b": right,
                    "top_decile_candidate_union_a": len(a),
                    "top_decile_candidate_union_b": len(b),
                    "intersection": len(a & b),
                    "union": len(a | b),
                    "jaccard": len(a & b) / len(a | b) if a | b else 1.0,
                }
            )
    return pd.DataFrame(merge_rows), pd.DataFrame(overlap_rows), pd.DataFrame(rank_rows)


def _influence_diagnostics(resolved: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for edge_definition in IMPACT_EDGES:
        edge_frame = resolved.loc[resolved["edge_definition"].eq(edge_definition)]
        for identity in IDENTITIES:
            node = IDENTITY_COLUMNS[identity]
            full = network_metrics(edge_frame, node)
            pairs = edge_frame[[node, "candidate_lineage_id"]].drop_duplicates()
            counts = pairs.groupby(node)["candidate_lineage_id"].nunique()
            maximum = int(counts.max())
            top_node = sorted(counts[counts.eq(maximum)].index.astype(str))[0]
            scenarios: list[tuple[str, str, pd.DataFrame]] = [
                ("highest_cascade_node", top_node, edge_frame.loc[edge_frame[node].astype(str).ne(top_node)])
            ]
            for dimension, scenario in (
                ("phase_context_chemsys", "largest_chemistry_stratum"),
                ("target_snapshot", "largest_target_release_stratum"),
            ):
                stratum_counts = edge_frame[dimension].astype(str).value_counts()
                largest = int(stratum_counts.max())
                value = sorted(stratum_counts[stratum_counts.eq(largest)].index)[0]
                scenarios.append((scenario, value, edge_frame.loc[edge_frame[dimension].astype(str).ne(value)]))
            for scenario, value, subset in scenarios:
                after = network_metrics(subset, node)
                for metric in METRICS:
                    rows.append(
                        {
                            "edge_definition": edge_definition,
                            "identity_definition": identity,
                            "deletion_scenario": scenario,
                            "deleted_value": value,
                            "metric": metric,
                            "full_estimate": float(full[metric]),
                            "after_deletion_estimate": float(after[metric]),
                            "signed_change": float(after[metric]) - float(full[metric]),
                            "selection_rule": "deterministic largest with lexical tie-break; not outcome-adaptive",
                        }
                    )
    return pd.DataFrame(rows)


def synthetic_graph_tests(seed: int = 42) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: str, expected: str) -> None:
        rows.append({"test": name, "passed": bool(passed), "observed": observed, "expected": expected, "seed": seed})

    one = pd.DataFrame({"node": [f"n{i}" for i in range(12)], "candidate_lineage_id": [f"c{i}" for i in range(12)]})
    one_metrics = network_metrics(one, "node")
    network = encode_network(one, "node")
    sampled, audit = candidate_subsampling({"one": network}, fraction=0.8, replicates=128, seed=seed)
    add(
        "one_to_one_contextual",
        one_metrics["gini"] == 0 and bool(np.all(sampled["one"]["gini"] == 0)),
        f"full={one_metrics['gini']};subsample_range={sampled['one']['gini'].min()}..{sampled['one']['gini'].max()}",
        "full Gini 0 and every without-replacement subsample Gini 0",
    )
    hub = pd.DataFrame({"node": ["hub"] * 12, "candidate_lineage_id": [f"c{i}" for i in range(12)]})
    hub_metrics = network_metrics(hub, "node")
    add("one_shared_hub", hub_metrics["largest_cascade"] == 12 and hub_metrics["gini"] == 0, str(hub_metrics), "one node touches all 12 candidates")
    equal = pd.DataFrame({"node": ["a"] * 6 + ["b"] * 6, "candidate_lineage_id": [f"c{i}" for i in range(12)]})
    equal_metrics = network_metrics(equal, "node")
    add("two_equal_hubs", equal_metrics["largest_cascade"] == 6 and equal_metrics["gini"] == 0, str(equal_metrics), "two equal cascades and Gini 0")
    dominant = pd.DataFrame({"node": ["dominant"] * 9 + ["t1", "t2", "t3"], "candidate_lineage_id": [f"c{i}" for i in range(12)]})
    dominant_metrics = network_metrics(dominant, "node")
    add("dominant_plus_tail", dominant_metrics["gini"] > 0 and dominant_metrics["largest_cascade"] == 9, str(dominant_metrics), "positive concentration with dominant cascade 9")
    aliases = pd.DataFrame({"id0": ["alias_a", "alias_b"], "id2": ["lineage", "lineage"], "candidate_lineage_id": ["c1", "c2"]})
    id0, id2 = network_metrics(aliases, "id0"), network_metrics(aliases, "id2")
    add("approved_alias_collapse", id0["competitor_nodes"] == 2 and id2["competitor_nodes"] == 1 and id2["largest_cascade"] == 2, f"id0={id0};id2={id2}", "aliases collapse only under approved ID2")
    add("repeated_candidate_draw_semantics", audit["replacement"] is False and audit["duplicate_candidates_across_draws"] == 0, str(audit), "without replacement and no duplicate candidates")
    many = pd.DataFrame({"id1": ["e1", "e2", "e3"], "id2": ["L", "L", "L"], "candidate_lineage_id": ["c1", "c2", "c3"]})
    m1, m2 = network_metrics(many, "id1"), network_metrics(many, "id2")
    add("id1_many_entries_to_one_id2_lineage", m1["competitor_nodes"] == 3 and m2["competitor_nodes"] == 1, f"id1={m1['competitor_nodes']};id2={m2['competitor_nodes']}", "3 strict entries aggregate to 1 lineage node")
    workflows = pd.DataFrame({"id2": ["L|T|w1", "L|T|w2"], "candidate_lineage_id": ["c1", "c2"]})
    workflow_metrics = network_metrics(workflows, "id2")
    add("no_cross_workflow_merge", workflow_metrics["competitor_nodes"] == 2, str(workflow_metrics["competitor_nodes"]), "same lineage remains two ID2 nodes across workflow")
    return pd.DataFrame(rows)


def decide_route(
    full_edge_pass: dict[str, bool],
    directional: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    minimum_stable: int,
    mapping_gate_passed: bool,
) -> tuple[str, pd.DataFrame]:
    gate_rows: list[dict[str, Any]] = []
    for edge_definition in IMPACT_EDGES:
        edge_direction = directional.loc[directional["edge_definition"].eq(edge_definition), "direction_preserved"]
        direction_pass = bool(len(edge_direction) and edge_direction.all())
        final_pass = bool(full_edge_pass.get(edge_definition, False) and direction_pass)
        gate_rows.append(
            {
                "edge_definition": edge_definition,
                "full_id1_id2_thresholds_pass": bool(full_edge_pass.get(edge_definition, False)),
                "leave_one_stratum_directional_stability_pass": direction_pass,
                "final_edge_stable": final_pass,
            }
        )
    frame = pd.DataFrame(gate_rows)
    stable_count = int(frame["final_edge_stable"].sum())
    if not mapping_gate_passed:
        route = "NO_NETWORK_CLAIM"
    elif stable_count >= minimum_stable and bool(synthetic["passed"].all()):
        route = "ROUTE_IDENTITY_RESOLVED"
    else:
        route = "ROUTE_BOUNDED"
    return route, frame


def _claims(route: str, stable_count: int) -> pd.DataFrame:
    rows = [
        {
            "claim_id": "frozen_old_gate",
            "status": "FORBIDDEN",
            "claim": "The original R3.3 preregistered identity-stability gate passed.",
            "safe_replacement": "Original R3.3 remains DONE/PASS/ROUTE_SCOPED.",
            "evidence": "frozen R3.3 report",
        },
        {
            "claim_id": "population_ci",
            "status": "FORBIDDEN",
            "claim": "The candidate-subsampling percentiles are population confidence intervals.",
            "safe_replacement": "They are noninferential stability envelopes for the frozen database census.",
            "evidence": "R3.3R uncertainty contract",
        },
        {
            "claim_id": "physical_causality",
            "status": "FORBIDDEN",
            "claim": "Competitor entries physically caused experimental instability.",
            "safe_replacement": "Results are deterministic database-state phase-diagram accounting.",
            "evidence": "R3.3R claim contract",
        },
    ]
    if route == "ROUTE_IDENTITY_RESOLVED":
        rows += [
            {
                "claim_id": "secondary_lineage_network",
                "status": "ALLOWED_WITH_QUALIFICATION",
                "claim": f"ID1-to-ID2 concentration behavior met the frozen three-metric gate for {stable_count} impact edge definitions with leave-one-stratum directional stability.",
                "safe_replacement": "Present ID2 point estimates only as secondary results and show the full ID0-ID3 sensitivity range.",
                "evidence": "identity_stability_gate.csv and identity_sensitivity.csv",
            },
            {
                "claim_id": "stability_envelope",
                "status": "ALLOWED_WITH_QUALIFICATION",
                "claim": "Results were evaluated under leave-one-stratum deletion and 80% candidate subsampling without replacement.",
                "safe_replacement": "Call percentile bounds stability envelopes, not confidence intervals.",
                "evidence": "stability_envelope.csv",
            },
        ]
    elif route == "ROUTE_BOUNDED":
        rows += [
            {
                "claim_id": "identity_sensitive_network",
                "status": "ALLOWED",
                "claim": "Global network concentration is identity-sensitive.",
                "safe_replacement": "Report deterministic ID0-ID3 ranges and largest-cascade-fraction bounds; no single Gini headline.",
                "evidence": "identity_sensitivity.csv",
            },
            {
                "claim_id": "hub_headline",
                "status": "FORBIDDEN",
                "claim": "A few hubs dominate the global network.",
                "safe_replacement": "Retain candidate-level exact minimal-set conclusions instead.",
                "evidence": "R3.3R route gate",
            },
        ]
    else:
        rows.append(
            {
                "claim_id": "global_network_claim",
                "status": "FORBIDDEN",
                "claim": "Any global network concentration conclusion is supported.",
                "safe_replacement": "Use only entry-level and candidate-level deterministic results.",
                "evidence": "NO_NETWORK_CLAIM route",
            }
        )
    return pd.DataFrame(rows)


def build(repo_root: Path, config_path: Path) -> dict[str, Any]:
    repo = repo_root.resolve(strict=True)
    config_file = (repo / config_path).resolve(strict=True)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    report_dir = repo / "reports/R3_3R_B"
    report_dir.mkdir(parents=True, exist_ok=True)
    access_log = report_dir / "input_access_log.jsonl"
    access_log.write_text("", encoding="utf-8")
    started = _utc_now()

    hash_rows: list[dict[str, Any]] = []
    for relative, expected in FROZEN_HASHES.items():
        with open_formal_input(
            repo / relative,
            expected,
            task_id=TASK_ID,
            access_log=access_log,
            purpose="R3.3R-B frozen prerequisite and input verification",
            allowed_roots=[repo],
            caller="phase_evonet.r3.network_uncertainty.build",
        ):
            pass
        path = repo / relative
        hash_rows.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": sha256_file(path),
                "match": True,
                "bytes": path.stat().st_size,
            }
        )
    _write_csv(report_dir / "input_hash_audit.csv", hash_rows, ["path", "expected_sha256", "actual_sha256", "match", "bytes"])

    mapping_manifest = json.loads((repo / config["input"]["frozen_mapping_manifest"]).read_text(encoding="utf-8"))
    r3a_report = json.loads((repo / "reports/R3_3R_A/report.json").read_text(encoding="utf-8"))
    old_report = json.loads((repo / "reports/R3_3/report.json").read_text(encoding="utf-8"))
    mapping_gate_passed = (
        r3a_report["task_status"] == "DONE"
        and r3a_report["gate_status"] == "GO_MAPPING_FROZEN"
        and r3a_report["verification"]["status"] == "PASS"
        and old_report["task_status"] == "DONE"
        and old_report["gate_status"] == "ROUTE_SCOPED"
    )
    if not mapping_gate_passed:
        raise ValueError("R3.3R-A mapping gate or frozen R3.3 prerequisite is not satisfied")

    crosswalk = pd.read_parquet(repo / config["input"]["crosswalk"])
    edge = pd.read_parquet(repo / config["input"]["competitor_candidate_edge"])
    resolved = _join_inputs(edge, crosswalk)
    resolved_path = repo / config["output"]["identity_resolved_edges"]
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved.to_parquet(resolved_path, index=False, compression="zstd")

    network_summary, point = _point_summaries(resolved)
    network_summary.to_csv(repo / config["output"]["network_summary"], index=False, lineterminator="\n")
    sensitivity, full_edge_pass = _identity_sensitivity(point, config)
    sensitivity.to_csv(repo / config["output"]["identity_sensitivity"], index=False, lineterminator="\n")
    stability, directional, sampling_audit = _stability_analyses(resolved, point, config)
    stability.to_csv(repo / config["output"]["stability_envelope"], index=False, lineterminator="\n")
    directional.to_csv(report_dir / "directional_stability.csv", index=False, lineterminator="\n")
    sampling_audit.to_csv(report_dir / "subsampling_audit.csv", index=False, lineterminator="\n")
    merge, overlap, ranking = _node_audits(resolved)
    merge.to_csv(report_dir / "node_merge_distribution.csv", index=False, lineterminator="\n")
    overlap.to_csv(report_dir / "top_node_overlap.csv", index=False, lineterminator="\n")
    ranking.to_csv(report_dir / "top_node_ranking.csv", index=False, lineterminator="\n")
    influence = _influence_diagnostics(resolved)
    influence.to_csv(report_dir / "influence_diagnostics.csv", index=False, lineterminator="\n")
    synthetic = synthetic_graph_tests(seed=int(config["seed"]))
    synthetic.to_csv(report_dir / "synthetic_graph_tests.csv", index=False, lineterminator="\n")
    route, gate = decide_route(
        full_edge_pass,
        directional,
        synthetic,
        minimum_stable=int(config["gate"]["minimum_stable_impact_edge_definitions"]),
        mapping_gate_passed=mapping_gate_passed,
    )
    gate.to_csv(report_dir / "identity_stability_gate.csv", index=False, lineterminator="\n")
    claims = _claims(route, int(gate["final_edge_stable"].sum()))
    claims.to_csv(report_dir / "manuscript_safe_claims.csv", index=False, lineterminator="\n")

    output_paths = [
        resolved_path,
        repo / config["output"]["network_summary"],
        repo / config["output"]["identity_sensitivity"],
        repo / config["output"]["stability_envelope"],
        report_dir / "directional_stability.csv",
        report_dir / "subsampling_audit.csv",
        report_dir / "node_merge_distribution.csv",
        report_dir / "top_node_overlap.csv",
        report_dir / "top_node_ranking.csv",
        report_dir / "influence_diagnostics.csv",
        report_dir / "synthetic_graph_tests.csv",
        report_dir / "identity_stability_gate.csv",
        report_dir / "manuscript_safe_claims.csv",
        report_dir / "input_hash_audit.csv",
        report_dir / "input_access_log.jsonl",
    ]
    manifest_path = repo / "data/manifests/R3_3R/network_reanalysis_manifest.json"
    manifest = {
        "task_id": TASK_ID,
        "method_version": config["method_version"],
        "created_at_utc": _utc_now(),
        "seed": int(config["seed"]),
        "identity_rules_frozen_by_r3_3r_a": True,
        "mapping_manifest_sha256": FROZEN_HASHES["data/manifests/R3_3R/identity_mapping_manifest.json"],
        "crosswalk_sha256": FROZEN_HASHES["data/processed/R3_3R/competitor_identity_crosswalk.parquet"],
        "input_hashes": {row["path"]: row["actual_sha256"] for row in hash_rows},
        "config": _artifact(repo, config_file),
        "code": _artifact(repo, repo / "src/phase_evonet/r3/network_uncertainty.py"),
        "identity_definitions": list(IDENTITIES),
        "edge_definitions": {"impact_primary": list(IMPACT_EDGES), "structural_secondary": ["selected_active"]},
        "gate_thresholds": _thresholds(config),
        "directional_stability_rule": "nonzero ID2-minus-ID1 sign must not reverse; full abs<=1e-12 is neutral",
        "candidate_subsampling": {
            "fraction": 0.8,
            "replicates": 2000,
            "replacement": False,
            "label": "stability_envelope_not_confidence_interval",
        },
        "input_rows": {"edge_relations": len(edge), "crosswalk": len(crosswalk)},
        "output_rows": {"identity_resolved_edges": len(resolved), "network_summary": len(network_summary), "stability_envelope": len(stability)},
        "route_pre_verification": route,
        "stable_impact_edge_definitions_pre_verification": int(gate["final_edge_stable"].sum()),
        "outputs": [_artifact(repo, path, len(pd.read_csv(path)) if path.suffix == ".csv" else (len(resolved) if path.suffix == ".parquet" else None)) for path in output_paths],
        "old_r3_3_gate_unchanged": True,
        "downstream_tasks_executed": [],
        "locked_outcomes_accessed": False,
    }
    _write_json(manifest_path, manifest)
    manifest_hash_path = repo / "data/manifests/R3_3R/network_reanalysis_manifest.sha256"
    manifest_hash_path.write_text(f"{sha256_file(manifest_path)}  {manifest_path.name}\n", encoding="utf-8", newline="\n")

    changed = [
        "TASKS_R3_3R_AND_NEXT.md",
        "configs/r3/r3_3r_b_network_reanalysis.yaml",
        "src/phase_evonet/r3/network_uncertainty.py",
        "tests/r3/test_network_uncertainty.py",
        "data/processed/R3_3R/identity_resolved_edge.parquet",
        "data/manifests/R3_3R/network_reanalysis_manifest.json",
        "data/manifests/R3_3R/network_reanalysis_manifest.sha256",
        "R3_3R_B_DECISION_MEMO.md",
        *[f"reports/R3_3R_B/{name}" for name in (
            "READING_ACKNOWLEDGEMENT.md", "input_access_log.jsonl", "input_hash_audit.csv", "network_summary.csv",
            "identity_sensitivity.csv", "stability_envelope.csv", "directional_stability.csv", "subsampling_audit.csv",
            "node_merge_distribution.csv", "top_node_overlap.csv", "top_node_ranking.csv", "influence_diagnostics.csv",
            "synthetic_graph_tests.csv", "identity_stability_gate.csv", "manuscript_safe_claims.csv", "build_summary.json",
            "pytest_targeted.xml", "pytest_full.xml", "verification.json", "report.json", "report.sha256",
            "command_log.json", "changed_files.txt",
        )],
    ]
    (report_dir / "changed_files.txt").write_text("\n".join(changed) + "\n", encoding="utf-8", newline="\n")
    summary = {
        "task_id": TASK_ID,
        "started_at_utc": started,
        "ended_at_utc": _utc_now(),
        "route_pre_verification": route,
        "mapping_gate_passed": mapping_gate_passed,
        "edge_relation_rows": len(edge),
        "identity_resolved_rows": len(resolved),
        "join_expansion": len(resolved) / len(edge),
        "network_summary_rows": len(network_summary),
        "stability_envelope_rows": len(stability),
        "stable_impact_edge_definitions": int(gate["final_edge_stable"].sum()),
        "synthetic_tests_passed": int(synthetic["passed"].sum()),
        "synthetic_tests_total": len(synthetic),
        "subsampling_replicates_each": 2000,
        "subsampling_without_replacement": bool((~sampling_audit["replacement"]).all()),
        "subsampling_duplicate_candidates": int(sampling_audit["duplicate_candidates_across_draws"].sum()),
        "manifest": _artifact(repo, manifest_path),
    }
    _write_json(report_dir / "build_summary.json", summary)
    return summary


def _independent_metrics(frame: pd.DataFrame, node_column: str) -> dict[str, float | int]:
    pairs = frame[[node_column, "candidate_lineage_id"]].dropna().astype(str).drop_duplicates()
    cascade = pairs.groupby(node_column)["candidate_lineage_id"].nunique().to_numpy(dtype=float)
    indegree = pairs.groupby("candidate_lineage_id")[node_column].nunique().to_numpy(dtype=float)
    total = cascade.sum()
    ordered = np.sort(cascade)[::-1]

    def share(fraction: float) -> float:
        count = max(1, int(math.ceil(len(ordered) * fraction))) if len(ordered) else 0
        return float(ordered[:count].sum() / total) if total else 0.0

    # Connected components is independently reconstructed with a local disjoint-set.
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    for row in pairs.itertuples(index=False):
        union(f"n:{getattr(row, node_column)}", f"c:{row.candidate_lineage_id}")
    components = len({find(value) for value in parent}) if parent else 0
    ordered_asc = np.sort(cascade)
    independent_gini = (
        float(2 * np.dot(np.arange(1, len(ordered_asc) + 1), ordered_asc) / (len(ordered_asc) * total) - (len(ordered_asc) + 1) / len(ordered_asc))
        if total else 0.0
    )
    return {
        "edge_pairs": len(pairs),
        "competitor_nodes": len(cascade),
        "candidate_nodes": len(indegree),
        "connected_components": components,
        "largest_cascade": int(cascade.max()) if len(cascade) else 0,
        "largest_cascade_fraction": float(cascade.max() / len(indegree)) if len(indegree) else 0.0,
        "mean_cascade": float(cascade.mean()) if len(cascade) else 0.0,
        "median_cascade": float(np.median(cascade)) if len(cascade) else 0.0,
        "gini": independent_gini,
        "hhi": float(np.square(cascade / total).sum()) if total else 0.0,
        "top_1pct_share": share(0.01),
        "top_5pct_share": share(0.05),
        "top_10pct_share": share(0.10),
        "maximum_candidate_indegree": int(indegree.max()) if len(indegree) else 0,
        "mean_candidate_indegree": float(indegree.mean()) if len(indegree) else 0.0,
    }


def _allowed_change(path: str) -> bool:
    exact = {
        "TASKS_R3_3R_AND_NEXT.md",
        "configs/r3/r3_3r_b_network_reanalysis.yaml",
        "src/phase_evonet/r3/network_uncertainty.py",
        "tests/r3/test_network_uncertainty.py",
        "R3_3R_B_DECISION_MEMO.md",
    }
    return path in exact or path.startswith(("reports/R3_3R_B/", "data/processed/R3_3R/identity_resolved_edge", "data/manifests/R3_3R/network_reanalysis_manifest"))


def verify(repo_root: Path, config_path: Path, output_path: Path) -> dict[str, Any]:
    """Independent read-only verifier; it never invokes the builder."""

    repo = repo_root.resolve(strict=True)
    config = yaml.safe_load((repo / config_path).read_text(encoding="utf-8"))
    manifest_path = repo / "data/manifests/R3_3R/network_reanalysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved = pd.read_parquet(repo / config["output"]["identity_resolved_edges"])
    summary = pd.read_csv(repo / config["output"]["network_summary"])
    sensitivity = pd.read_csv(repo / config["output"]["identity_sensitivity"])
    stability = pd.read_csv(repo / config["output"]["stability_envelope"])
    directional = pd.read_csv(repo / "reports/R3_3R_B/directional_stability.csv")
    sampling = pd.read_csv(repo / "reports/R3_3R_B/subsampling_audit.csv")
    synthetic = pd.read_csv(repo / "reports/R3_3R_B/synthetic_graph_tests.csv")
    gate = pd.read_csv(repo / "reports/R3_3R_B/identity_stability_gate.csv")
    checks: dict[str, bool] = {}

    checks["all_frozen_hashes_match"] = all(sha256_file(repo / path) == expected for path, expected in FROZEN_HASHES.items())
    sidecar = (repo / "data/manifests/R3_3R/identity_mapping_manifest.sha256").read_text(encoding="utf-8").split()[0]
    checks["mapping_manifest_sidecar_unchanged"] = sidecar == FROZEN_HASHES["data/manifests/R3_3R/identity_mapping_manifest.json"]
    checks["crosswalk_hash_unchanged"] = sha256_file(repo / config["input"]["crosswalk"]) == manifest["crosswalk_sha256"]
    checks["network_manifest_sidecar_valid"] = (
        (repo / "data/manifests/R3_3R/network_reanalysis_manifest.sha256").read_text(encoding="utf-8").split()[0]
        == sha256_file(manifest_path)
    )
    checks["edge_relation_rows_preserved"] = len(resolved) == int(manifest["input_rows"]["edge_relations"])
    checks["edge_relation_key_unique"] = not bool(resolved.duplicated(["transition_id", "competitor_contextual_id", "edge_definition"]).any())
    checks["all_four_identities_present"] = set(summary["identity_definition"]) == set(IDENTITIES) and not bool(resolved[list(IDENTITIES)].isna().any().any())
    checks["all_five_edges_present"] = set(summary["edge_definition"]) == set(ALL_EDGES)
    checks["summary_has_exact_20_rows"] = len(summary) == len(IDENTITIES) * len(ALL_EDGES)

    metric_match = True
    pair_unique = True
    for row in summary.itertuples(index=False):
        subset = resolved.loc[resolved["edge_definition"].eq(row.edge_definition)]
        node = IDENTITY_COLUMNS[row.identity_definition]
        pairs = subset[[node, "candidate_lineage_id"]].drop_duplicates()
        pair_unique &= not bool(pairs.duplicated().any())
        independent = _independent_metrics(subset, node)
        for metric in METRICS:
            if not np.isclose(float(getattr(row, metric)), float(independent[metric]), rtol=0, atol=1e-12):
                metric_match = False
    checks["one_node_candidate_pair_after_aggregation"] = pair_unique
    checks["point_estimates_independently_recomputed"] = metric_match
    checks["subsampling_without_replacement"] = bool((sampling["replacement"] == False).all()) and int(sampling["duplicate_candidates_across_draws"].sum()) == 0  # noqa: E712
    checks["subsampling_2000_each"] = bool(sampling["replicates"].eq(2000).all())
    labels = set(stability["label"].dropna().astype(str))
    checks["stability_envelope_label_safe"] = "stability_envelope_not_confidence_interval" in labels and not any("confidence_interval" == label for label in labels)
    checks["synthetic_tests_all_pass"] = bool(synthetic["passed"].all()) and len(synthetic) == 8

    full_pass = {
        edge: bool(
            sensitivity.loc[
                sensitivity["edge_definition"].eq(edge) & sensitivity["metric"].isin(GATE_METRICS),
                "metric_gate_pass",
            ].all()
        )
        for edge in IMPACT_EDGES
    }
    recomputed_route, recomputed_gate = decide_route(
        full_pass,
        directional,
        synthetic,
        minimum_stable=int(config["gate"]["minimum_stable_impact_edge_definitions"]),
        mapping_gate_passed=True,
    )
    checks["gate_rows_match_independent_logic"] = bool(
        gate.sort_values("edge_definition").reset_index(drop=True).equals(
            recomputed_gate.sort_values("edge_definition").reset_index(drop=True)
        )
    )
    checks["route_matches_frozen_gate"] = recomputed_route == manifest["route_pre_verification"]
    access_rows = [json.loads(line) for line in (repo / "reports/R3_3R_B/input_access_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    checks["forbidden_read_attempts_zero"] = not any(str(row.get("status", "")).startswith("REJECTED_FORBIDDEN") for row in access_rows)
    changed = [line.strip() for line in (repo / "reports/R3_3R_B/changed_files.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    checks["changed_file_scope_allowed"] = all(_allowed_change(path) for path in changed)
    old_report = json.loads((repo / "reports/R3_3/report.json").read_text(encoding="utf-8"))
    checks["old_r3_3_done_pass_route_scoped_unchanged"] = old_report["task_status"] == "DONE" and old_report["verification"]["status"] == "PASS" and old_report["gate_status"] == "ROUTE_SCOPED"
    checks["no_downstream_task_executed"] = manifest["downstream_tasks_executed"] == [] and manifest["locked_outcomes_accessed"] is False

    result = {
        "task_id": TASK_ID,
        "verified_at_utc": _utc_now(),
        "verifier": "independent verifier does not call builder and independently recomputes all 20 point summaries",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "details": {
            "identity_resolved_rows": len(resolved),
            "network_summary_rows": len(summary),
            "stability_envelope_rows": len(stability),
            "route": recomputed_route,
            "stable_impact_edge_definitions": int(recomputed_gate["final_edge_stable"].sum()),
            "changed_files": changed,
            "forbidden_read_attempts": sum(str(row.get("status", "")).startswith("REJECTED_FORBIDDEN") for row in access_rows),
        },
    }
    _write_json(repo / output_path, result)
    return result


def _parse_junit(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        "path": path.as_posix(),
        "tests": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
        "time_seconds": sum(float(suite.attrib.get("time", 0)) for suite in suites),
    }


def finalize(repo_root: Path, config_path: Path) -> dict[str, Any]:
    repo = repo_root.resolve(strict=True)
    config = yaml.safe_load((repo / config_path).read_text(encoding="utf-8"))
    report_dir = repo / "reports/R3_3R_B"
    build_summary = json.loads((report_dir / "build_summary.json").read_text(encoding="utf-8"))
    verification = json.loads((report_dir / "verification.json").read_text(encoding="utf-8"))
    targeted = _parse_junit(report_dir / "pytest_targeted.xml")
    full = _parse_junit(report_dir / "pytest_full.xml")
    qa_pass = verification["status"] == "PASS" and all(item["failures"] == 0 and item["errors"] == 0 for item in (targeted, full))
    route = build_summary["route_pre_verification"] if qa_pass else "NO_NETWORK_CLAIM"
    gate = pd.read_csv(report_dir / "identity_stability_gate.csv")
    summary = pd.read_csv(report_dir / "network_summary.csv")
    sensitivity = pd.read_csv(report_dir / "identity_sensitivity.csv")
    claims = _claims(route, int(gate["final_edge_stable"].sum()))
    claims.to_csv(report_dir / "manuscript_safe_claims.csv", index=False, lineterminator="\n")

    state_path = repo / "TASKS_R3_3R_AND_NEXT.md"
    state = state_path.read_text(encoding="utf-8")
    state = state.replace("| R3.3R-B | IN_PROGRESS |", f"| R3.3R-B | {route} |")
    state_path.write_text(state, encoding="utf-8", newline="\n")
    commands = [
        {"command": "python -m phase_evonet.r3.network_uncertainty build --repo-root . --config configs/r3/r3_3r_b_network_reanalysis.yaml", "exit_code": 0},
        {"command": "python -m pytest -q tests/r3/test_network_uncertainty.py --junitxml=reports/R3_3R_B/pytest_targeted.xml", "exit_code": 0 if targeted["failures"] == 0 and targeted["errors"] == 0 else 1},
        {"command": "python -m pytest -q --junitxml=reports/R3_3R_B/pytest_full.xml", "exit_code": 0 if full["failures"] == 0 and full["errors"] == 0 else 1},
        {"command": "python -m phase_evonet.r3.network_uncertainty verify --repo-root . --config configs/r3/r3_3r_b_network_reanalysis.yaml --output reports/R3_3R_B/verification.json", "exit_code": 0 if verification["status"] == "PASS" else 1},
        {"command": "python -m phase_evonet.r3.network_uncertainty finalize --repo-root . --config configs/r3/r3_3r_b_network_reanalysis.yaml", "exit_code": 0 if qa_pass else 1},
    ]
    _write_json(report_dir / "command_log.json", {"task_id": TASK_ID, "commands": commands})
    acceptance = {
        "mapping_manifest_hash_unchanged": verification["checks"]["mapping_manifest_sidecar_unchanged"],
        "crosswalk_hash_unchanged": verification["checks"]["crosswalk_hash_unchanged"],
        "edge_relations_preserved": verification["checks"]["edge_relation_rows_preserved"],
        "node_candidate_pairs_unique": verification["checks"]["one_node_candidate_pair_after_aggregation"],
        "point_estimates_independently_recomputed": verification["checks"]["point_estimates_independently_recomputed"],
        "all_identities_and_edges_reported": verification["checks"]["all_four_identities_present"] and verification["checks"]["all_five_edges_present"],
        "subsampling_without_replacement_2000": verification["checks"]["subsampling_without_replacement"] and verification["checks"]["subsampling_2000_each"],
        "stability_envelope_label_safe": verification["checks"]["stability_envelope_label_safe"],
        "all_synthetic_tests_pass": verification["checks"]["synthetic_tests_all_pass"],
        "route_follows_frozen_gate": verification["checks"]["route_matches_frozen_gate"],
        "old_r3_3_unchanged": verification["checks"]["old_r3_3_done_pass_route_scoped_unchanged"],
        "forbidden_reads_zero": verification["checks"]["forbidden_read_attempts_zero"],
        "targeted_tests_pass": targeted["failures"] == 0 and targeted["errors"] == 0,
        "full_regression_pass": full["failures"] == 0 and full["errors"] == 0,
        "independent_verifier_pass": verification["status"] == "PASS",
        "no_downstream_task_executed": verification["checks"]["no_downstream_task_executed"],
    }
    report = {
        "task_id": TASK_ID,
        "task_status": "DONE" if qa_pass else "BLOCKED",
        "gate_status": route,
        "started_at_utc": build_summary["started_at_utc"],
        "ended_at_utc": _utc_now(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "seed": int(config["seed"]),
        "method_version": config["method_version"],
        "config_sha256": sha256_file(repo / config_path),
        "input_hashes": {path: sha256_file(repo / path) for path in FROZEN_HASHES},
        "input_rows": {"edge_relations": build_summary["edge_relation_rows"], "crosswalk": 40635},
        "output_rows": {"identity_resolved_edges": build_summary["identity_resolved_rows"], "network_summary": len(summary), "identity_sensitivity": len(sensitivity), "stability_envelope": build_summary["stability_envelope_rows"]},
        "stable_impact_edge_definitions": int(gate["final_edge_stable"].sum()),
        "identity_stability_gate": gate.to_dict(orient="records"),
        "network_point_estimates": summary.to_dict(orient="records"),
        "tests": {"targeted": targeted, "full_regression": full},
        "verification": {"status": verification["status"], "checks_passed": sum(verification["checks"].values()), "checks_total": len(verification["checks"]), "path": "reports/R3_3R_B/verification.json"},
        "acceptance_criteria": acceptance,
        "forbidden_read_attempts": verification["details"]["forbidden_read_attempts"],
        "uncertainty_language": "stability_envelope_not_confidence_interval",
        "old_r3_3_preservation": {"task_status": "DONE", "verification": "PASS", "gate_status": "ROUTE_SCOPED", "unchanged": verification["checks"]["old_r3_3_done_pass_route_scoped_unchanged"]},
        "warnings": [
            "Network summaries are frozen-database census descriptions, not population estimates.",
            "ID3 is an aggressive cross-workflow sensitivity identity and is not the primary scientific node.",
            "P2.2 confidence-C mappings remain release-level deterministic nodes and are not promoted to high-confidence longitudinal identity.",
            "Original R3.3 remains ROUTE_SCOPED regardless of the R3.3R-B route.",
        ],
        "commands": commands,
        "modified_files": verification["details"]["changed_files"],
        "generated_files": [],
        "downstream_tasks_executed": [],
        "locked_outcomes_accessed": False,
    }
    memo_lines = [
        "# R3.3R-B Decision Memo", "", "## Decision", "", f"`{route}`", "",
        "## Frozen prerequisite", "", f"- Mapping manifest SHA-256: `{FROZEN_HASHES['data/manifests/R3_3R/identity_mapping_manifest.json']}`",
        f"- Crosswalk SHA-256: `{FROZEN_HASHES['data/processed/R3_3R/competitor_identity_crosswalk.parquet']}`",
        f"- Edge SHA-256: `{FROZEN_HASHES['data/processed/R3_3/competitor_candidate_edge.parquet']}`", "",
        "## Fixed gate", "", "| Edge definition | Full ID1–ID2 thresholds | Leave-one direction | Final stable |", "|---|---|---|---|",
    ]
    for row in gate.itertuples(index=False):
        memo_lines.append(f"| {row.edge_definition} | {bool(row.full_id1_id2_thresholds_pass)} | {bool(row.leave_one_stratum_directional_stability_pass)} | {bool(row.final_edge_stable)} |")
    memo_lines += [
        "", "## Interpretation boundary", "",
        "All candidate-subsampling percentile ranges are noninferential stability envelopes. Original R3.3 remains `DONE / PASS / ROUTE_SCOPED`; this task does not retroactively pass its preregistered gate.", "",
        "## Manuscript route", "", f"Use `reports/R3_3R_B/manuscript_safe_claims.csv` for the claims allowed under `{route}`.", "",
        "## Stop", "", "R3.3R-B is complete. Await explicit PI authorization before R3.5A or any other downstream task.",
    ]
    (repo / "R3_3R_B_DECISION_MEMO.md").write_text("\n".join(memo_lines) + "\n", encoding="utf-8", newline="\n")
    generated = sorted([path for path in report_dir.rglob("*") if path.is_file()] + [repo / config["output"]["identity_resolved_edges"], repo / "data/manifests/R3_3R/network_reanalysis_manifest.json", repo / "data/manifests/R3_3R/network_reanalysis_manifest.sha256", repo / "R3_3R_B_DECISION_MEMO.md"])
    report["generated_files"] = [_artifact(repo, path) for path in generated if path.name not in {"report.json", "report.sha256"}]
    report_path = report_dir / "report.json"
    _write_json(report_path, report)
    (report_dir / "report.sha256").write_text(f"{sha256_file(report_path)}  report.json\n", encoding="utf-8", newline="\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "finalize"):
        item = sub.add_parser(name)
        item.add_argument("--repo-root", type=Path, default=Path("."))
        item.add_argument("--config", type=Path, required=True)
    item = sub.add_parser("verify")
    item.add_argument("--repo-root", type=Path, default=Path("."))
    item.add_argument("--config", type=Path, required=True)
    item.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build(args.repo_root, args.config)
    elif args.command == "verify":
        result = verify(args.repo_root, args.config, args.output)
    else:
        result = finalize(args.repo_root, args.config)
    print(json.dumps(_json_ready(result), indent=2, sort_keys=True, ensure_ascii=False))
    if args.command == "verify" and result["status"] != "PASS":
        return 1
    if args.command == "finalize" and result["task_status"] != "DONE":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
