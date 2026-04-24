from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .results import append_csv_row, ensure_csv_header


VARIANT_AUDIT_COLUMNS = [
    "experiment_name",
    "dataset_id",
    "split_id",
    "scenario_id",
    "severity",
    "graph_seed",
    "oracle_labels",
    "scenario_applied",
    "scenario_family",
    "scenario_method",
    "severity_param",
    "requested_severity",
    "graph_view_mode",
    "base_graph_path",
    "graph_path",
    "requested_change",
    "requested_change_unit",
    "realized_change",
    "realized_change_unit",
    "n_changed_nodes_requested",
    "n_changed_nodes_actual",
    "n_added_edges_requested",
    "n_added_edges_actual",
    "n_added_edge_pairs_requested",
    "n_added_edge_pairs_actual",
    "n_removed_edges_requested",
    "n_removed_edges_actual",
    "n_rewired_edges_selected",
    "n_rewired_edges_actual",
    "n_nodes_before",
    "n_nodes_after",
    "n_edges_before",
    "n_edges_after",
    "mean_in_degree_before",
    "mean_in_degree_after",
    "median_in_degree_before",
    "median_in_degree_after",
    "mean_out_degree_before",
    "mean_out_degree_after",
    "median_out_degree_before",
    "median_out_degree_after",
    "heterophily_ratio_before",
    "heterophily_ratio_after",
    "pos_rate_before",
    "pos_rate_after",
    "mean_cosine_to_sampled_normal_before",
    "mean_cosine_to_sampled_normal_after",
    "mean_l2_delta",
    "gamma",
    "fraud_to_normal_neighbor_ratio_before",
    "fraud_to_normal_neighbor_ratio_after",
    "fraud_to_fraud_neighbor_ratio_before",
    "fraud_to_fraud_neighbor_ratio_after",
    "mean_selected_out_degree_before",
    "mean_selected_out_degree_after",
    "pseudo_label_pos_rate",
    "relation_filter_applied_json",
    "sampling_policy_json",
    "n_rewired_edges_selected_by_relation_json",
    "n_rewired_edges_actual_by_relation_json",
    "n_camouflaged_edges_added_by_relation_json",
    "n_added_edge_pairs_actual_by_relation_json",
    "extra_info_json",
]


def ensure_variant_audit_csv(path: Path, *, overwrite: bool = False) -> None:
    ensure_csv_header(path, VARIANT_AUDIT_COLUMNS, overwrite=overwrite)


def append_variant_audit_row(path: Path, row: Mapping[str, Any]) -> None:
    append_csv_row(path, VARIANT_AUDIT_COLUMNS, row)


def _json_or_blank(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set, dict)):
        if not value:
            return ""
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    text = str(value)
    return text if text else ""


def _pick_first(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if value == "":
            continue
        return value
    return ""


def _safe_metric(stats: Mapping[str, Any], key: str) -> Any:
    value = stats.get(key, "")
    return "" if value is None else value


def _normalized_counts(scenario_info: Mapping[str, Any]) -> dict[str, Any]:
    changed_nodes_requested = _pick_first(
        scenario_info.get("n_camouflaged_nodes_requested"),
        scenario_info.get("n_camouflaged_nodes"),
    )
    changed_nodes_actual = _pick_first(
        scenario_info.get("n_camouflaged_nodes_actual"),
        scenario_info.get("n_camouflaged_nodes"),
    )
    added_edges_requested = _pick_first(
        scenario_info.get("n_added_edges_requested_total"),
        scenario_info.get("n_added_edges_requested"),
        scenario_info.get("n_camouflaged_edges_requested"),
    )
    added_edges_actual = _pick_first(
        scenario_info.get("n_added_edges_actual_total"),
        scenario_info.get("n_added_edges"),
        scenario_info.get("n_camouflaged_edges_added"),
    )
    added_pairs_requested = _pick_first(scenario_info.get("n_added_edge_pairs_requested"))
    added_pairs_actual = _pick_first(scenario_info.get("n_added_edge_pairs_actual"))
    removed_edges_requested = _pick_first(
        scenario_info.get("n_suspicious_edges_removed_requested"),
        scenario_info.get("n_removed_edges_requested"),
    )
    removed_edges_actual = _pick_first(
        scenario_info.get("n_suspicious_edges_removed"),
        scenario_info.get("n_removed_edges_actual"),
    )
    rewired_selected = _pick_first(
        scenario_info.get("n_rewired_edges_selected"),
        scenario_info.get("n_rewired_edges"),
    )
    rewired_actual = _pick_first(
        scenario_info.get("n_rewired_edges_actual"),
        scenario_info.get("n_rewired_edges"),
    )
    return {
        "n_changed_nodes_requested": changed_nodes_requested,
        "n_changed_nodes_actual": changed_nodes_actual,
        "n_added_edges_requested": added_edges_requested,
        "n_added_edges_actual": added_edges_actual,
        "n_added_edge_pairs_requested": added_pairs_requested,
        "n_added_edge_pairs_actual": added_pairs_actual,
        "n_removed_edges_requested": removed_edges_requested,
        "n_removed_edges_actual": removed_edges_actual,
        "n_rewired_edges_selected": rewired_selected,
        "n_rewired_edges_actual": rewired_actual,
    }


def _requested_realized_summary(counts: Mapping[str, Any], *, scenario_applied: bool) -> tuple[Any, str, Any, str]:
    if counts.get("n_rewired_edges_selected", "") != "":
        return (
            counts.get("n_rewired_edges_selected", ""),
            "edges_selected",
            counts.get("n_rewired_edges_actual", ""),
            "edges_changed",
        )
    if counts.get("n_changed_nodes_requested", "") != "":
        return (
            counts.get("n_changed_nodes_requested", ""),
            "nodes_selected",
            counts.get("n_changed_nodes_actual", ""),
            "nodes_changed",
        )
    if counts.get("n_added_edges_requested", "") != "":
        return (
            counts.get("n_added_edges_requested", ""),
            "edges_requested",
            counts.get("n_added_edges_actual", ""),
            "edges_added",
        )
    if not scenario_applied:
        return 0, "none", 0, "none"
    return "", "", "", ""


def build_variant_audit_row(
    *,
    experiment_name: str,
    dataset_id: str,
    split_id: str,
    scenario_id: str,
    severity: float,
    graph_seed: int,
    oracle_labels: bool,
    scenario_applied: bool,
    scenario_family: str,
    scenario_method: str,
    severity_param: str,
    graph_view_mode: str,
    base_graph_path: str,
    graph_path: str,
    base_stats: Mapping[str, Any],
    variant_stats: Mapping[str, Any],
    scenario_info: Mapping[str, Any] | None,
) -> dict[str, Any]:
    info = dict(scenario_info or {})
    counts = _normalized_counts(info)
    requested_change, requested_change_unit, realized_change, realized_change_unit = _requested_realized_summary(
        counts,
        scenario_applied=bool(scenario_applied),
    )
    if str(scenario_id) == "clean":
        requested_change = 0
        requested_change_unit = "none"
        realized_change = 0
        realized_change_unit = "none"

    row: dict[str, Any] = {
        "experiment_name": str(experiment_name),
        "dataset_id": str(dataset_id),
        "split_id": str(split_id),
        "scenario_id": str(scenario_id),
        "severity": float(severity),
        "graph_seed": int(graph_seed),
        "oracle_labels": bool(oracle_labels),
        "scenario_applied": bool(scenario_applied),
        "scenario_family": str(scenario_family),
        "scenario_method": str(scenario_method),
        "severity_param": str(severity_param),
        "requested_severity": float(severity),
        "graph_view_mode": str(graph_view_mode),
        "base_graph_path": str(base_graph_path),
        "graph_path": str(graph_path),
        "requested_change": requested_change,
        "requested_change_unit": requested_change_unit,
        "realized_change": realized_change,
        "realized_change_unit": realized_change_unit,
        "n_nodes_before": _safe_metric(base_stats, "n_nodes"),
        "n_nodes_after": _safe_metric(variant_stats, "n_nodes"),
        "n_edges_before": _safe_metric(base_stats, "n_edges"),
        "n_edges_after": _safe_metric(variant_stats, "n_edges"),
        "mean_in_degree_before": _safe_metric(base_stats, "mean_in_degree"),
        "mean_in_degree_after": _safe_metric(variant_stats, "mean_in_degree"),
        "median_in_degree_before": _safe_metric(base_stats, "median_in_degree"),
        "median_in_degree_after": _safe_metric(variant_stats, "median_in_degree"),
        "mean_out_degree_before": _safe_metric(base_stats, "mean_out_degree"),
        "mean_out_degree_after": _safe_metric(variant_stats, "mean_out_degree"),
        "median_out_degree_before": _safe_metric(base_stats, "median_out_degree"),
        "median_out_degree_after": _safe_metric(variant_stats, "median_out_degree"),
        "heterophily_ratio_before": _pick_first(info.get("heterophily_ratio_before"), _safe_metric(base_stats, "heterophily_ratio")),
        "heterophily_ratio_after": _pick_first(info.get("heterophily_ratio_after"), _safe_metric(variant_stats, "heterophily_ratio")),
        "pos_rate_before": _safe_metric(base_stats, "pos_rate"),
        "pos_rate_after": _safe_metric(variant_stats, "pos_rate"),
        "mean_cosine_to_sampled_normal_before": _pick_first(info.get("mean_cosine_to_sampled_normal_before")),
        "mean_cosine_to_sampled_normal_after": _pick_first(info.get("mean_cosine_to_sampled_normal_after")),
        "mean_l2_delta": _pick_first(info.get("mean_l2_delta")),
        "gamma": _pick_first(info.get("gamma")),
        "fraud_to_normal_neighbor_ratio_before": _pick_first(info.get("fraud_to_normal_neighbor_ratio_before")),
        "fraud_to_normal_neighbor_ratio_after": _pick_first(info.get("fraud_to_normal_neighbor_ratio_after")),
        "fraud_to_fraud_neighbor_ratio_before": _pick_first(info.get("fraud_to_fraud_neighbor_ratio_before")),
        "fraud_to_fraud_neighbor_ratio_after": _pick_first(info.get("fraud_to_fraud_neighbor_ratio_after")),
        "mean_selected_out_degree_before": _pick_first(info.get("mean_selected_out_degree_before")),
        "mean_selected_out_degree_after": _pick_first(info.get("mean_selected_out_degree_after")),
        "pseudo_label_pos_rate": _pick_first(info.get("pseudo_label_pos_rate")),
        "relation_filter_applied_json": _json_or_blank(info.get("relation_filter_applied")),
        "sampling_policy_json": _json_or_blank(info.get("sampling_policy")),
        "n_rewired_edges_selected_by_relation_json": _json_or_blank(info.get("n_rewired_edges_selected_by_relation")),
        "n_rewired_edges_actual_by_relation_json": _json_or_blank(info.get("n_rewired_edges_actual_by_relation")),
        "n_camouflaged_edges_added_by_relation_json": _json_or_blank(info.get("n_camouflaged_edges_added_by_relation")),
        "n_added_edge_pairs_actual_by_relation_json": _json_or_blank(info.get("n_added_edge_pairs_actual_by_relation")),
    }
    row.update(counts)

    consumed_keys = {
        "heterophily_ratio_before",
        "heterophily_ratio_after",
        "mean_cosine_to_sampled_normal_before",
        "mean_cosine_to_sampled_normal_after",
        "mean_l2_delta",
        "gamma",
        "fraud_to_normal_neighbor_ratio_before",
        "fraud_to_normal_neighbor_ratio_after",
        "fraud_to_fraud_neighbor_ratio_before",
        "fraud_to_fraud_neighbor_ratio_after",
        "mean_selected_out_degree_before",
        "mean_selected_out_degree_after",
        "pseudo_label_pos_rate",
        "relation_filter_applied",
        "sampling_policy",
        "n_rewired_edges_selected_by_relation",
        "n_rewired_edges_actual_by_relation",
        "n_camouflaged_edges_added_by_relation",
        "n_added_edge_pairs_actual_by_relation",
        "n_camouflaged_nodes_requested",
        "n_camouflaged_nodes_actual",
        "n_camouflaged_nodes",
        "n_added_edges_requested_total",
        "n_added_edges_requested",
        "n_added_edges_actual_total",
        "n_added_edges",
        "n_added_edge_pairs_requested",
        "n_added_edge_pairs_actual",
        "n_camouflaged_edges_requested",
        "n_camouflaged_edges_added",
        "n_suspicious_edges_removed_requested",
        "n_suspicious_edges_removed",
        "n_removed_edges_requested",
        "n_removed_edges_actual",
        "n_rewired_edges_selected",
        "n_rewired_edges_actual",
        "n_rewired_edges",
    }
    extra_info = {key: value for key, value in info.items() if key not in consumed_keys}
    row["extra_info_json"] = _json_or_blank(extra_info)
    return row
