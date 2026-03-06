from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .config import get_training_seeds, scenario_oracle_labels
from .results import PROTOCOL_TRAIN_ON_VARIANT, normalize_protocol
from .variants import filter_variants, read_variants_csv


BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class ResultRow:
    experiment_name: str
    dataset_id: str
    split_id: str
    graph_seed: int
    training_seed: int
    scenario_id: str
    severity: float
    model_id: str
    protocol: str
    roc_auc: float
    average_precision: float
    f1_macro: float
    status: str


@dataclass(frozen=True)
class MetricStats:
    mean: float
    std: float
    n: int
    ci_lower: float
    ci_upper: float


@dataclass(frozen=True)
class AuditMetricSpec:
    metric_id: str
    display_name: str
    extractor: Callable[[Mapping[str, Any]], float | None]
    unit: str = ""
    unit_field: str = ""


@dataclass(frozen=True)
class CompletenessReport:
    total_expected: int
    present_ok: int
    missing_count: int
    error_count: int
    missing_keys: list[tuple[str, str, str, float, int, int, str, str]]
    error_keys: list[tuple[str, str, str, float, int, int, str, str]]


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return int(default)


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _parse_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return bool(x)
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}


def _read_results_csv(path: Path) -> list[ResultRow]:
    if not path.exists():
        raise FileNotFoundError(f"results.csv not found: {path}")

    rows: list[ResultRow] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                ResultRow(
                    experiment_name=str(r.get("experiment_name", "")),
                    dataset_id=str(r.get("dataset_id", "")),
                    split_id=str(r.get("split_id", "")),
                    graph_seed=_safe_int(r.get("graph_seed", 0)),
                    training_seed=_safe_int(r.get("training_seed", 0)),
                    scenario_id=str(r.get("scenario_id", "")),
                    severity=float(r.get("severity", 0.0) or 0.0),
                    model_id=str(r.get("model_id", "")),
                    protocol=normalize_protocol(r.get("protocol", "")),
                    roc_auc=_safe_float(r.get("roc_auc", "")),
                    average_precision=_safe_float(r.get("average_precision", "")),
                    f1_macro=_safe_float(r.get("f1_macro", "")),
                    status=str(r.get("status", "")),
                )
            )
    return rows


def _read_variant_audit_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"variant_audit.csv not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        required = {"dataset_id", "split_id", "scenario_id", "severity", "graph_seed", "oracle_labels", "graph_view_mode"}
        missing = sorted(required - set(fieldnames))
        if missing:
            raise RuntimeError(f"variant_audit.csv is missing required columns: {missing}")
        for raw in reader:
            row = dict(raw)
            row["severity"] = float(raw.get("severity", 0.0) or 0.0)
            row["graph_seed"] = _safe_int(raw.get("graph_seed", 0))
            row["oracle_labels"] = _parse_bool(raw.get("oracle_labels", False))
            row["scenario_applied"] = _parse_bool(raw.get("scenario_applied", True))
            rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No rows found in {path}. Re-run `py -m benchmark.run --stage graphs ...`; "
            "a previous graphs run may have been interrupted."
        )
    return rows


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


def _severity_grid_from_config(cfg: dict[str, Any]) -> dict[str, list[float]]:
    grid: dict[str, list[float]] = {}
    for s in cfg.get("scenarios", []):
        sid = str(s.get("scenario_id", ""))
        sevs = [float(x) for x in s.get("severity_values", [])]
        grid[sid] = sorted(set(sevs))
    return grid


def _model_ids_from_config(cfg: dict[str, Any]) -> list[str]:
    mids = []
    for m in cfg.get("models", []):
        mid = str(m.get("model_id", ""))
        if mid:
            mids.append(mid)
    seen = set()
    out = []
    for mid in mids:
        if mid in seen:
            continue
        seen.add(mid)
        out.append(mid)
    return out


def _scenario_meta_by_id(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {"clean": {"oracle_labels": False, "display_name": "clean"}}
    for scenario_cfg in cfg.get("scenarios", []):
        scenario_id = str(scenario_cfg.get("scenario_id", ""))
        oracle_labels = scenario_oracle_labels(scenario_cfg)
        out[scenario_id] = {
            "oracle_labels": oracle_labels,
            "display_name": scenario_id + (" (oracle)" if oracle_labels else ""),
        }
    return out


def _protocols_present_in_results(results_csv: Path, *, model_ids: list[str]) -> set[str]:
    if not results_csv.exists():
        return set()

    allowed_model_ids = set(model_ids)
    protocols: set[str] = set()
    with results_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            mid = str(r.get("model_id", ""))
            if allowed_model_ids and mid not in allowed_model_ids:
                continue
            status = str(r.get("status", "")).strip().lower()
            if status not in {"ok", "error"}:
                continue
            protocols.add(normalize_protocol(r.get("protocol", "")))
    return protocols


def _require_numpy() -> Any:
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover
        raise RuntimeError("numpy is required for plot aggregation and confidence intervals.") from e
    return np


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    num_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    vv = [float(v) for v in values if math.isfinite(float(v))]
    if not vv:
        return float("nan"), float("nan")
    if len(vv) == 1:
        return float(vv[0]), float(vv[0])

    np = _require_numpy()
    arr = np.asarray(vv, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    sample_idx = rng.choice(arr.shape[0], size=(int(num_resamples), arr.shape[0]), replace=True)
    sample_means = arr[sample_idx].mean(axis=1)
    lo, hi = np.percentile(sample_means, [2.5, 97.5])
    return float(lo), float(hi)


def _aggregate_metric_stats(values: Sequence[float], *, ci: bool) -> MetricStats:
    vv = [float(v) for v in values if math.isfinite(float(v))]
    if not vv:
        return MetricStats(mean=float("nan"), std=float("nan"), n=0, ci_lower=float("nan"), ci_upper=float("nan"))
    if len(vv) == 1:
        return MetricStats(mean=float(vv[0]), std=0.0, n=1, ci_lower=float(vv[0]), ci_upper=float(vv[0]))

    mean_v = float(sum(vv) / len(vv))
    var = float(sum((v - mean_v) ** 2 for v in vv) / (len(vv) - 1))
    std_v = float(math.sqrt(max(0.0, var)))
    if ci:
        ci_lower, ci_upper = _bootstrap_mean_ci(vv)
    else:
        ci_lower = float("nan")
        ci_upper = float("nan")
    return MetricStats(mean=mean_v, std=std_v, n=int(len(vv)), ci_lower=ci_lower, ci_upper=ci_upper)


def _bootstrap_difference_stats(
    left_values: Sequence[float],
    right_values: Sequence[float],
    *,
    ci: bool,
    num_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> MetricStats:
    left = [float(v) for v in left_values if math.isfinite(float(v))]
    right = [float(v) for v in right_values if math.isfinite(float(v))]
    if not left or not right:
        return MetricStats(mean=float("nan"), std=float("nan"), n=0, ci_lower=float("nan"), ci_upper=float("nan"))

    mean_v = float(sum(left) / len(left) - sum(right) / len(right))
    if len(left) == 1 and len(right) == 1:
        return MetricStats(mean=mean_v, std=0.0, n=1, ci_lower=mean_v, ci_upper=mean_v)

    np = _require_numpy()
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    left_idx = rng.choice(left_arr.shape[0], size=(int(num_resamples), left_arr.shape[0]), replace=True)
    right_idx = rng.choice(right_arr.shape[0], size=(int(num_resamples), right_arr.shape[0]), replace=True)
    diffs = left_arr[left_idx].mean(axis=1) - right_arr[right_idx].mean(axis=1)
    std_v = float(np.std(diffs, ddof=1)) if diffs.shape[0] > 1 else 0.0
    if ci:
        ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])
    else:
        ci_lower = float("nan")
        ci_upper = float("nan")
    return MetricStats(
        mean=mean_v,
        std=std_v,
        n=int(min(len(left), len(right))),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
    )


def _bootstrap_robustness_stats(
    x_values: Sequence[float],
    values_by_severity: Sequence[Sequence[float]],
    *,
    ci: bool,
    num_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[MetricStats, MetricStats]:
    np = _require_numpy()

    x = np.asarray([float(x) for x in x_values], dtype=np.float64)
    if x.shape[0] < 2:
        empty = MetricStats(mean=float("nan"), std=float("nan"), n=0, ci_lower=float("nan"), ci_upper=float("nan"))
        return empty, empty

    x_range = float(x.max() - x.min())
    if x_range <= 0.0:
        empty = MetricStats(mean=float("nan"), std=float("nan"), n=0, ci_lower=float("nan"), ci_upper=float("nan"))
        return empty, empty

    series: list[list[float]] = []
    for values in values_by_severity:
        vv = [float(v) for v in values if math.isfinite(float(v))]
        if not vv:
            empty = MetricStats(mean=float("nan"), std=float("nan"), n=0, ci_lower=float("nan"), ci_upper=float("nan"))
            return empty, empty
        series.append(vv)

    mean_curve = np.asarray([sum(vv) / len(vv) for vv in series], dtype=np.float64)
    auc_mean = float(np.sum((x[1:] - x[:-1]) * (mean_curve[1:] + mean_curve[:-1]) * 0.5))
    avg_mean = float(auc_mean / x_range)

    if all(len(vv) == 1 for vv in series):
        auc_stats = MetricStats(mean=auc_mean, std=0.0, n=1, ci_lower=auc_mean, ci_upper=auc_mean)
        avg_stats = MetricStats(mean=avg_mean, std=0.0, n=1, ci_lower=avg_mean, ci_upper=avg_mean)
        return auc_stats, avg_stats

    rng = np.random.default_rng(int(seed))
    auc_samples = []
    avg_samples = []
    for _ in range(int(num_resamples)):
        sampled_curve = []
        for vv in series:
            arr = np.asarray(vv, dtype=np.float64)
            idx = rng.choice(arr.shape[0], size=arr.shape[0], replace=True)
            sampled_curve.append(float(arr[idx].mean()))
        y = np.asarray(sampled_curve, dtype=np.float64)
        auc = float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5))
        auc_samples.append(auc)
        avg_samples.append(float(auc / x_range))

    auc_std = float(np.std(np.asarray(auc_samples, dtype=np.float64), ddof=1)) if len(auc_samples) > 1 else 0.0
    avg_std = float(np.std(np.asarray(avg_samples, dtype=np.float64), ddof=1)) if len(avg_samples) > 1 else 0.0
    if ci:
        auc_lo, auc_hi = np.percentile(np.asarray(auc_samples, dtype=np.float64), [2.5, 97.5])
        avg_lo, avg_hi = np.percentile(np.asarray(avg_samples, dtype=np.float64), [2.5, 97.5])
    else:
        auc_lo = auc_hi = avg_lo = avg_hi = float("nan")

    return (
        MetricStats(
            mean=auc_mean,
            std=auc_std,
            n=int(min(len(vv) for vv in series)),
            ci_lower=float(auc_lo),
            ci_upper=float(auc_hi),
        ),
        MetricStats(
            mean=avg_mean,
            std=avg_std,
            n=int(min(len(vv) for vv in series)),
            ci_lower=float(avg_lo),
            ci_upper=float(avg_hi),
        ),
    )


def _scenario_oracle_labels(scenario_meta: dict[str, dict[str, Any]], scenario_id: str) -> bool:
    meta = scenario_meta.get(str(scenario_id), {})
    return bool(meta.get("oracle_labels", False))


def _scenario_display_name(scenario_meta: dict[str, dict[str, Any]], scenario_id: str) -> str:
    meta = scenario_meta.get(str(scenario_id), {})
    return str(meta.get("display_name", str(scenario_id)))


def _variant_join_key(
    dataset_id: str,
    split_id: str,
    scenario_id: str,
    severity: float,
    graph_seed: int,
) -> tuple[str, str, str, float, int]:
    return (
        str(dataset_id),
        str(split_id),
        str(scenario_id),
        float(severity),
        int(graph_seed),
    )


def _variant_join_key_from_variant_row(variant: Any) -> tuple[str, str, str, float, int]:
    return _variant_join_key(
        str(getattr(variant, "dataset_id")),
        str(getattr(variant, "split_id")),
        str(getattr(variant, "scenario_id")),
        float(getattr(variant, "severity")),
        int(getattr(variant, "graph_seed")),
    )


def _variant_join_key_from_result_row(result: ResultRow) -> tuple[str, str, str, float, int]:
    return _variant_join_key(
        result.dataset_id,
        result.split_id,
        result.scenario_id,
        result.severity,
        result.graph_seed,
    )


def _variant_join_key_from_audit_row(audit_row: Mapping[str, Any]) -> tuple[str, str, str, float, int]:
    return _variant_join_key(
        str(audit_row.get("dataset_id", "")),
        str(audit_row.get("split_id", "")),
        str(audit_row.get("scenario_id", "")),
        float(audit_row.get("severity", 0.0) or 0.0),
        _safe_int(audit_row.get("graph_seed", 0)),
    )


def _index_variants_by_key(variants: Sequence[Any]) -> dict[tuple[str, str, str, float, int], Any]:
    out: dict[tuple[str, str, str, float, int], Any] = {}
    for variant in variants:
        key = _variant_join_key_from_variant_row(variant)
        if key in out:
            raise RuntimeError(f"Duplicate graph variant key found in graph_variants.csv: {key}")
        out[key] = variant
    return out


def _index_audit_rows_by_key(audit_rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str, float, int], dict[str, Any]]:
    out: dict[tuple[str, str, str, float, int], dict[str, Any]] = {}
    for audit_row in audit_rows:
        key = _variant_join_key_from_audit_row(audit_row)
        if key in out:
            raise RuntimeError(f"Duplicate audit key found in variant_audit.csv: {key}")
        out[key] = dict(audit_row)
    return out


def _sample_variant_key_text(keys: Sequence[tuple[str, str, str, float, int]], *, limit: int = 5) -> str:
    sample = ", ".join(
        f"{scenario_id}/sev={float(severity):g}/gs={int(graph_seed)}"
        for _dataset_id, _split_id, scenario_id, severity, graph_seed in keys[: int(limit)]
    )
    if len(keys) > int(limit):
        sample += ", ..."
    return sample


def _require_audit_keys_for_variants(
    variant_by_key: Mapping[tuple[str, str, str, float, int], Any],
    audit_by_key: Mapping[tuple[str, str, str, float, int], Mapping[str, Any]],
) -> None:
    missing = sorted(key for key in variant_by_key if key not in audit_by_key)
    if missing:
        sample = _sample_variant_key_text(missing)
        raise RuntimeError(f"variant_audit.csv is missing rows for graph variants: {sample}")


def _protocols_by_dataset_split(rows: Sequence[ResultRow], *, model_ids: Sequence[str]) -> dict[tuple[str, str], set[str]]:
    allowed_model_ids = set(model_ids)
    out: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        if allowed_model_ids and row.model_id not in allowed_model_ids:
            continue
        if row.status not in {"ok", "error"}:
            continue
        key = (str(row.dataset_id), str(row.split_id))
        out.setdefault(key, set()).add(normalize_protocol(row.protocol))
    return out


def _finite_float_or_none(value: Any) -> float | None:
    out = _safe_float(value)
    if not math.isfinite(out):
        return None
    return float(out)


def _audit_field_metric(metric_id: str, display_name: str, *, unit: str = "") -> AuditMetricSpec:
    return AuditMetricSpec(
        metric_id=metric_id,
        display_name=display_name,
        extractor=lambda row, field=metric_id: _finite_float_or_none(row.get(field, "")),
        unit=unit,
    )


def _audit_delta_metric(
    metric_id: str,
    display_name: str,
    *,
    before_key: str,
    after_key: str,
    unit: str = "",
) -> AuditMetricSpec:
    return AuditMetricSpec(
        metric_id=metric_id,
        display_name=display_name,
        extractor=lambda row, before_key=before_key, after_key=after_key: (
            None
            if _finite_float_or_none(row.get(before_key, "")) is None
            or _finite_float_or_none(row.get(after_key, "")) is None
            else float(_finite_float_or_none(row.get(after_key, "")) - _finite_float_or_none(row.get(before_key, "")))
        ),
        unit=unit,
    )


def _selected_audit_metric_specs(cfg: dict[str, Any]) -> list[AuditMetricSpec]:
    raw_metric_specs: dict[str, AuditMetricSpec] = {
        "requested_change": AuditMetricSpec(
            metric_id="requested_change",
            display_name="Requested change",
            extractor=lambda row: _finite_float_or_none(row.get("requested_change", "")),
            unit_field="requested_change_unit",
        ),
        "realized_change": AuditMetricSpec(
            metric_id="realized_change",
            display_name="Realized change",
            extractor=lambda row: _finite_float_or_none(row.get("realized_change", "")),
            unit_field="realized_change_unit",
        ),
        "heterophily_ratio_before": _audit_field_metric("heterophily_ratio_before", "Heterophily ratio before"),
        "heterophily_ratio_after": _audit_field_metric("heterophily_ratio_after", "Heterophily ratio after"),
        "mean_cosine_to_sampled_normal_before": _audit_field_metric(
            "mean_cosine_to_sampled_normal_before",
            "Cosine to sampled normal before",
        ),
        "mean_cosine_to_sampled_normal_after": _audit_field_metric(
            "mean_cosine_to_sampled_normal_after",
            "Cosine to sampled normal after",
        ),
        "fraud_to_normal_neighbor_ratio_before": _audit_field_metric(
            "fraud_to_normal_neighbor_ratio_before",
            "Fraud-to-normal neighbor ratio before",
        ),
        "fraud_to_normal_neighbor_ratio_after": _audit_field_metric(
            "fraud_to_normal_neighbor_ratio_after",
            "Fraud-to-normal neighbor ratio after",
        ),
        "n_added_edge_pairs_actual": _audit_field_metric(
            "n_added_edge_pairs_actual",
            "Added edge pairs",
            unit="edges",
        ),
        "n_rewired_edges_actual": _audit_field_metric(
            "n_rewired_edges_actual",
            "Rewired edges",
            unit="edges",
        ),
    }
    derived_metric_specs: list[AuditMetricSpec] = [
        _audit_delta_metric(
            "heterophily_ratio_shift",
            "Heterophily ratio shift",
            before_key="heterophily_ratio_before",
            after_key="heterophily_ratio_after",
        ),
        _audit_delta_metric(
            "mean_cosine_to_sampled_normal_shift",
            "Cosine to sampled normal shift",
            before_key="mean_cosine_to_sampled_normal_before",
            after_key="mean_cosine_to_sampled_normal_after",
        ),
        _audit_delta_metric(
            "fraud_to_normal_neighbor_ratio_shift",
            "Fraud-to-normal neighbor ratio shift",
            before_key="fraud_to_normal_neighbor_ratio_before",
            after_key="fraud_to_normal_neighbor_ratio_after",
        ),
    ]

    configured_metrics = cfg.get("evaluation", {}).get("audit_metrics", [])
    configured_metric_ids = {str(metric_id) for metric_id in configured_metrics if str(metric_id)}

    ordered_specs: list[AuditMetricSpec] = [
        raw_metric_specs["requested_change"],
        raw_metric_specs["realized_change"],
        raw_metric_specs["heterophily_ratio_after"],
        raw_metric_specs["n_rewired_edges_actual"],
        raw_metric_specs["mean_cosine_to_sampled_normal_after"],
        raw_metric_specs["fraud_to_normal_neighbor_ratio_after"],
        raw_metric_specs["n_added_edge_pairs_actual"],
    ]

    for metric_id in sorted(configured_metric_ids):
        spec = raw_metric_specs.get(metric_id)
        if spec is not None:
            ordered_specs.append(spec)

    if {"heterophily_ratio_before", "heterophily_ratio_after"} & configured_metric_ids:
        ordered_specs.append(derived_metric_specs[0])
    if {
        "mean_cosine_to_sampled_normal_before",
        "mean_cosine_to_sampled_normal_after",
    } & configured_metric_ids:
        ordered_specs.append(derived_metric_specs[1])
    if {
        "fraud_to_normal_neighbor_ratio_before",
        "fraud_to_normal_neighbor_ratio_after",
    } & configured_metric_ids:
        ordered_specs.append(derived_metric_specs[2])

    if not configured_metric_ids:
        ordered_specs.extend(derived_metric_specs)

    seen: set[str] = set()
    out: list[AuditMetricSpec] = []
    for spec in ordered_specs:
        if spec.metric_id in seen:
            continue
        seen.add(spec.metric_id)
        out.append(spec)
    return out


def _build_performance_audit_join_rows(
    rows: Sequence[ResultRow],
    *,
    model_ids: Sequence[str],
    variant_by_key: Mapping[tuple[str, str, str, float, int], Any],
    audit_by_key: Mapping[tuple[str, str, str, float, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    allowed_model_ids = set(model_ids)
    joined_rows: list[dict[str, Any]] = []
    for row in rows:
        if allowed_model_ids and row.model_id not in allowed_model_ids:
            continue
        key = _variant_join_key_from_result_row(row)
        if key not in variant_by_key:
            continue
        audit_row = audit_by_key.get(key)
        if audit_row is None:
            raise RuntimeError(f"variant_audit.csv is missing a join row for results key: {key}")
        variant = variant_by_key[key]
        joined_rows.append(
            {
                "experiment_name": row.experiment_name,
                "dataset_id": row.dataset_id,
                "split_id": row.split_id,
                "scenario_id": row.scenario_id,
                "severity": float(row.severity),
                "graph_seed": int(row.graph_seed),
                "training_seed": int(row.training_seed),
                "model_id": row.model_id,
                "protocol": normalize_protocol(row.protocol),
                "status": row.status,
                "oracle_labels": bool(audit_row.get("oracle_labels", False)),
                "scenario_applied": bool(getattr(variant, "scenario_applied")),
                "graph_view_mode": str(audit_row.get("graph_view_mode", "")),
                "scenario_family": str(audit_row.get("scenario_family", "")),
                "scenario_method": str(audit_row.get("scenario_method", "")),
                "severity_param": str(audit_row.get("severity_param", "")),
                "requested_severity": _finite_float_or_none(audit_row.get("requested_severity", "")),
                "requested_change": _finite_float_or_none(audit_row.get("requested_change", "")),
                "requested_change_unit": str(audit_row.get("requested_change_unit", "")),
                "realized_change": _finite_float_or_none(audit_row.get("realized_change", "")),
                "realized_change_unit": str(audit_row.get("realized_change_unit", "")),
                "roc_auc": row.roc_auc,
                "average_precision": row.average_precision,
                "f1_macro": row.f1_macro,
                "base_graph_path": str(getattr(variant, "base_graph_path")),
                "graph_path": str(getattr(variant, "graph_path")),
                "variant_n_nodes": int(getattr(variant, "n_nodes")),
                "variant_n_edges": int(getattr(variant, "n_edges")),
                "variant_heterophily_ratio": getattr(variant, "heterophily_ratio"),
                "variant_pos_rate": getattr(variant, "pos_rate"),
                "heterophily_ratio_before": _finite_float_or_none(audit_row.get("heterophily_ratio_before", "")),
                "heterophily_ratio_after": _finite_float_or_none(audit_row.get("heterophily_ratio_after", "")),
                "mean_cosine_to_sampled_normal_before": _finite_float_or_none(
                    audit_row.get("mean_cosine_to_sampled_normal_before", "")
                ),
                "mean_cosine_to_sampled_normal_after": _finite_float_or_none(
                    audit_row.get("mean_cosine_to_sampled_normal_after", "")
                ),
                "fraud_to_normal_neighbor_ratio_before": _finite_float_or_none(
                    audit_row.get("fraud_to_normal_neighbor_ratio_before", "")
                ),
                "fraud_to_normal_neighbor_ratio_after": _finite_float_or_none(
                    audit_row.get("fraud_to_normal_neighbor_ratio_after", "")
                ),
                "n_added_edge_pairs_actual": _finite_float_or_none(audit_row.get("n_added_edge_pairs_actual", "")),
                "n_rewired_edges_actual": _finite_float_or_none(audit_row.get("n_rewired_edges_actual", "")),
            }
        )
    return joined_rows


def _build_audit_summary_rows(
    audit_rows: Sequence[Mapping[str, Any]],
    *,
    protocols_by_dataset_split: Mapping[tuple[str, str], set[str]],
    metric_specs: Sequence[AuditMetricSpec],
    ci: bool,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, float, str, str, str, str, bool, str, str], list[float]] = {}
    for audit_row in audit_rows:
        dataset_id = str(audit_row.get("dataset_id", ""))
        split_id = str(audit_row.get("split_id", ""))
        scenario_id = str(audit_row.get("scenario_id", ""))
        severity = float(audit_row.get("severity", 0.0) or 0.0)
        oracle_labels = bool(audit_row.get("oracle_labels", False))
        graph_view_mode = str(audit_row.get("graph_view_mode", ""))
        source = "clean" if scenario_id == "clean" else "scenario"
        protocols = sorted(protocols_by_dataset_split.get((dataset_id, split_id), {PROTOCOL_TRAIN_ON_VARIANT}))
        for protocol in protocols:
            for spec in metric_specs:
                value = spec.extractor(audit_row)
                if value is None or not math.isfinite(float(value)):
                    continue
                metric_unit = str(audit_row.get(spec.unit_field, "")) if spec.unit_field else str(spec.unit)
                key = (
                    dataset_id,
                    split_id,
                    scenario_id,
                    severity,
                    protocol,
                    spec.metric_id,
                    spec.display_name,
                    metric_unit,
                    oracle_labels,
                    graph_view_mode,
                    source,
                )
                groups.setdefault(key, []).append(float(value))

    out: list[dict[str, Any]] = []
    for key in sorted(groups):
        (
            dataset_id,
            split_id,
            scenario_id,
            severity,
            protocol,
            metric_id,
            metric_label,
            metric_unit,
            oracle_labels,
            graph_view_mode,
            source,
        ) = key
        stats = _aggregate_metric_stats(groups[key], ci=ci)
        out.append(
            {
                "dataset_id": dataset_id,
                "split_id": split_id,
                "scenario_id": scenario_id,
                "severity": float(severity),
                "protocol": protocol,
                "audit_metric": metric_id,
                "audit_metric_label": metric_label,
                "audit_metric_unit": metric_unit,
                "oracle_labels": bool(oracle_labels),
                "graph_view_mode": graph_view_mode,
                "mean": stats.mean,
                "std": stats.std,
                "ci_lower": stats.ci_lower,
                "ci_upper": stats.ci_upper,
                "n_graphs": stats.n,
                "source": source,
            }
        )
    return out


def _build_cross_split_audit_rows(audit_rows: Sequence[dict[str, Any]], *, ci: bool) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float, str, str, str, str, bool, str, str], list[dict[str, Any]]] = {}
    for row in audit_rows:
        key = (
            str(row.get("dataset_id", "")),
            str(row.get("scenario_id", "")),
            float(row.get("severity", 0.0) or 0.0),
            str(row.get("protocol", "")),
            str(row.get("audit_metric", "")),
            str(row.get("audit_metric_label", "")),
            str(row.get("audit_metric_unit", "")),
            bool(row.get("oracle_labels", False)),
            str(row.get("graph_view_mode", "")),
            str(row.get("source", "")),
        )
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for key in sorted(groups):
        (
            dataset_id,
            scenario_id,
            severity,
            protocol,
            metric_id,
            metric_label,
            metric_unit,
            oracle_labels,
            graph_view_mode,
            source,
        ) = key
        split_means = [_safe_float(row.get("mean", "")) for row in groups[key]]
        stats = _aggregate_metric_stats(split_means, ci=ci)
        out.append(
            {
                "dataset_id": dataset_id,
                "scenario_id": scenario_id,
                "severity": float(severity),
                "protocol": protocol,
                "audit_metric": metric_id,
                "audit_metric_label": metric_label,
                "audit_metric_unit": metric_unit,
                "oracle_labels": bool(oracle_labels),
                "graph_view_mode": graph_view_mode,
                "mean": stats.mean,
                "std": stats.std,
                "ci_lower": stats.ci_lower,
                "ci_upper": stats.ci_upper,
                "n_splits": stats.n,
                "source": source,
            }
        )
    return out


def _format_run_key(key: tuple[str, str, str, float, int, int, str, str]) -> str:
    _dataset_id, _split_id, scenario_id, severity, graph_seed, training_seed, model_id, protocol = key
    suffix = ""
    if normalize_protocol(protocol) != PROTOCOL_TRAIN_ON_VARIANT:
        suffix = f"/protocol={normalize_protocol(protocol)}"
    return f"{model_id}/{scenario_id}/sev={float(severity):g}/gs={int(graph_seed)}/ts={int(training_seed)}{suffix}"


def _completeness_report(
    cfg: dict[str, Any],
    *,
    out_dir: Path,
    include_noop: bool,
    only_clean: bool,
    max_variants: int | None,
    max_training_seeds: int | None,
) -> CompletenessReport:
    variants_csv = out_dir / "graph_variants.csv"
    results_csv = out_dir / "results.csv"

    if not variants_csv.exists() or not results_csv.exists():
        return CompletenessReport(0, 0, 0, 0, [], [])

    variants = filter_variants(
        read_variants_csv(variants_csv),
        include_noop=bool(include_noop),
        only_clean=bool(only_clean),
        max_variants=max_variants,
    )
    training_seeds = get_training_seeds(cfg)
    if max_training_seeds is not None:
        training_seeds = training_seeds[: int(max_training_seeds)]

    model_ids = [m for m in _model_ids_from_config(cfg) if m in {"mlp", "sage", "pmp", "secgfd"}]
    protocols = _protocols_present_in_results(results_csv, model_ids=model_ids)
    if not protocols:
        protocols = {PROTOCOL_TRAIN_ON_VARIANT}

    expected_keys: set[tuple[str, str, str, float, int, int, str, str]] = set()
    for variant in variants:
        for training_seed in training_seeds:
            for model_id in model_ids:
                for protocol in protocols:
                    expected_keys.add(
                        (
                            str(variant.dataset_id),
                            str(variant.split_id),
                            str(variant.scenario_id),
                            float(variant.severity),
                            int(variant.graph_seed),
                            int(training_seed),
                            str(model_id),
                            normalize_protocol(protocol),
                        )
                    )

    found_ok: set[tuple[str, str, str, float, int, int, str, str]] = set()
    found_err: set[tuple[str, str, str, float, int, int, str, str]] = set()
    with results_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (
                str(r.get("dataset_id", "")),
                str(r.get("split_id", "")),
                str(r.get("scenario_id", "")),
                float(r.get("severity", 0.0) or 0.0),
                _safe_int(r.get("graph_seed", 0)),
                _safe_int(r.get("training_seed", 0)),
                str(r.get("model_id", "")),
                normalize_protocol(r.get("protocol", "")),
            )
            status = str(r.get("status", "")).strip().lower()
            if status == "ok":
                found_ok.add(key)
            elif status == "error":
                found_err.add(key)

    unresolved_errors = sorted((expected_keys & found_err) - found_ok)
    missing = sorted(expected_keys - found_ok - set(unresolved_errors))
    present_ok = len(expected_keys) - len(missing) - len(unresolved_errors)

    report_rows = []
    for ds, split, scenario_id, severity, graph_seed, tr, mid, protocol in missing:
        report_rows.append(
            {
                "dataset_id": ds,
                "split_id": split,
                "scenario_id": scenario_id,
                "severity": severity,
                "graph_seed": graph_seed,
                "training_seed": tr,
                "model_id": mid,
                "protocol": protocol,
                "status": "missing",
            }
        )
    for ds, split, scenario_id, severity, graph_seed, tr, mid, protocol in unresolved_errors:
        report_rows.append(
            {
                "dataset_id": ds,
                "split_id": split,
                "scenario_id": scenario_id,
                "severity": severity,
                "graph_seed": graph_seed,
                "training_seed": tr,
                "model_id": mid,
                "protocol": protocol,
                "status": "error",
            }
        )

    plots_dir = out_dir / "plots"
    _write_csv(
        plots_dir / "missing_or_error_runs.csv",
        [
            "dataset_id",
            "split_id",
            "scenario_id",
            "severity",
            "graph_seed",
            "training_seed",
            "model_id",
            "protocol",
            "status",
        ],
        report_rows,
    )

    return CompletenessReport(
        total_expected=len(expected_keys),
        present_ok=present_ok,
        missing_count=len(missing),
        error_count=len(unresolved_errors),
        missing_keys=missing,
        error_keys=unresolved_errors,
    )


def _print_completeness_report(report: CompletenessReport) -> None:
    if report.total_expected <= 0:
        print("[plots] Completeness: 0/0 expected runs present (0%), 0 errors, 0 missing")
        return

    pct = (100.0 * float(report.present_ok) / float(report.total_expected)) if report.total_expected > 0 else 0.0
    print(
        f"[plots] Completeness: {report.present_ok}/{report.total_expected} expected runs present "
        f"({pct:.0f}%), {report.error_count} errors, {report.missing_count} missing"
    )
    if report.missing_keys:
        sample = ", ".join(_format_run_key(key) for key in report.missing_keys[:5])
        suffix = ", ..." if len(report.missing_keys) > 5 else ""
        print(f"[plots] Missing runs: {sample}{suffix}")
    if report.error_keys:
        sample = ", ".join(_format_run_key(key) for key in report.error_keys[:5])
        suffix = ", ..." if len(report.error_keys) > 5 else ""
        print(f"[plots] Error runs: {sample}{suffix}")


def _metric_values(rows: Iterable[ResultRow], metric_key: str) -> list[float]:
    return [float(getattr(row, metric_key)) for row in rows]


def _yerr_from_stats(stats_by_point: Sequence[MetricStats], *, use_ci: bool) -> Any:
    if use_ci:
        lower = []
        upper = []
        for stats in stats_by_point:
            if not (
                math.isfinite(stats.mean)
                and math.isfinite(stats.ci_lower)
                and math.isfinite(stats.ci_upper)
            ):
                lower.append(0.0)
                upper.append(0.0)
                continue
            lower.append(max(0.0, float(stats.mean - stats.ci_lower)))
            upper.append(max(0.0, float(stats.ci_upper - stats.mean)))
        return [lower, upper]

    out = []
    for stats in stats_by_point:
        if math.isfinite(stats.std):
            out.append(max(0.0, float(stats.std)))
        else:
            out.append(0.0)
    return out


def _build_cross_split_summary_rows(summary_rows: Sequence[dict[str, Any]], *, ci: bool) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float, str, str, str, bool, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        key = (
            str(row.get("dataset_id", "")),
            str(row.get("scenario_id", "")),
            float(row.get("severity", 0.0) or 0.0),
            str(row.get("model_id", "")),
            str(row.get("protocol", "")),
            str(row.get("metric", "")),
            bool(row.get("oracle_labels", False)),
            str(row.get("source", "")),
        )
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for key in sorted(groups):
        dataset_id, scenario_id, severity, model_id, protocol, metric, oracle_labels, source = key
        rows = groups[key]
        split_means = [_safe_float(row.get("mean", "")) for row in rows]
        stats = _aggregate_metric_stats(split_means, ci=ci)
        out.append(
            {
                "dataset_id": dataset_id,
                "scenario_id": scenario_id,
                "severity": float(severity),
                "model_id": model_id,
                "protocol": protocol,
                "metric": metric,
                "oracle_labels": bool(oracle_labels),
                "mean": stats.mean,
                "std": stats.std,
                "ci_lower": stats.ci_lower,
                "ci_upper": stats.ci_upper,
                "n_splits": stats.n,
                "source": source,
            }
        )
    return out


def _build_cross_split_drop_rows(drop_rows: Sequence[dict[str, Any]], *, ci: bool) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str, float, bool], list[dict[str, Any]]] = {}
    for row in drop_rows:
        key = (
            str(row.get("dataset_id", "")),
            str(row.get("scenario_id", "")),
            str(row.get("model_id", "")),
            str(row.get("protocol", "")),
            str(row.get("metric", "")),
            float(row.get("max_severity", 0.0) or 0.0),
            bool(row.get("oracle_labels", False)),
        )
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for key in sorted(groups):
        dataset_id, scenario_id, model_id, protocol, metric, max_severity, oracle_labels = key
        rows = groups[key]
        clean_stats = _aggregate_metric_stats([_safe_float(row.get("clean_mean", "")) for row in rows], ci=ci)
        stressed_stats = _aggregate_metric_stats([_safe_float(row.get("max_severity_mean", "")) for row in rows], ci=ci)
        drop_stats = _aggregate_metric_stats([_safe_float(row.get("drop_mean", "")) for row in rows], ci=ci)
        out.append(
            {
                "dataset_id": dataset_id,
                "scenario_id": scenario_id,
                "model_id": model_id,
                "protocol": protocol,
                "metric": metric,
                "max_severity": float(max_severity),
                "oracle_labels": bool(oracle_labels),
                "clean_mean": clean_stats.mean,
                "clean_std": clean_stats.std,
                "clean_ci_lower": clean_stats.ci_lower,
                "clean_ci_upper": clean_stats.ci_upper,
                "max_severity_mean": stressed_stats.mean,
                "max_severity_std": stressed_stats.std,
                "max_severity_ci_lower": stressed_stats.ci_lower,
                "max_severity_ci_upper": stressed_stats.ci_upper,
                "drop_mean": drop_stats.mean,
                "drop_std": drop_stats.std,
                "drop_ci_lower": drop_stats.ci_lower,
                "drop_ci_upper": drop_stats.ci_upper,
                "n_splits": drop_stats.n,
            }
        )
    return out


def _build_cross_split_robust_rows(robust_rows: Sequence[dict[str, Any]], *, ci: bool) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str, bool], list[dict[str, Any]]] = {}
    for row in robust_rows:
        key = (
            str(row.get("dataset_id", "")),
            str(row.get("scenario_id", "")),
            str(row.get("model_id", "")),
            str(row.get("protocol", "")),
            str(row.get("metric", "")),
            bool(row.get("oracle_labels", False)),
        )
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for key in sorted(groups):
        dataset_id, scenario_id, model_id, protocol, metric, oracle_labels = key
        rows = groups[key]
        auc_stats = _aggregate_metric_stats([_safe_float(row.get("robustness_auc_mean", "")) for row in rows], ci=ci)
        avg_stats = _aggregate_metric_stats(
            [_safe_float(row.get("robustness_avg_metric_mean", "")) for row in rows],
            ci=ci,
        )
        out.append(
            {
                "dataset_id": dataset_id,
                "scenario_id": scenario_id,
                "model_id": model_id,
                "protocol": protocol,
                "metric": metric,
                "oracle_labels": bool(oracle_labels),
                "robustness_auc_mean": auc_stats.mean,
                "robustness_auc_std": auc_stats.std,
                "robustness_auc_ci_lower": auc_stats.ci_lower,
                "robustness_auc_ci_upper": auc_stats.ci_upper,
                "robustness_avg_metric_mean": avg_stats.mean,
                "robustness_avg_metric_std": avg_stats.std,
                "robustness_avg_metric_ci_lower": avg_stats.ci_lower,
                "robustness_avg_metric_ci_upper": avg_stats.ci_upper,
                "n_splits": avg_stats.n,
            }
        )
    return out


def _plot_summary_rows(
    summary_rows: Sequence[dict[str, Any]],
    *,
    out_group_dir: Path,
    dataset_id: str,
    split_label: str,
    protocol: str,
    scenario_id: str,
    scenario_display_name: str,
    oracle_labels: bool,
    metric_key: str,
    metric_label: str,
    model_ids: Sequence[str],
    severity_values: Sequence[float],
    colors: dict[str, str],
    ci: bool,
    plt: Any,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=200)
    title = f"{dataset_id}/{split_label} [{protocol}] - {scenario_display_name}: {metric_label} vs severity"
    legend_title = "model (oracle scenario)" if oracle_labels else "model"

    for model_id in model_ids:
        points = []
        for severity in severity_values:
            found = None
            for row in summary_rows:
                if (
                    str(row.get("model_id", "")) == str(model_id)
                    and str(row.get("scenario_id", "")) == str(scenario_id)
                    and str(row.get("metric", "")) == str(metric_key)
                    and float(row.get("severity", 0.0) or 0.0) == float(severity)
                ):
                    found = row
                    break
            if found is None:
                points.append(MetricStats(float("nan"), float("nan"), 0, float("nan"), float("nan")))
                continue
            points.append(
                MetricStats(
                    mean=_safe_float(found.get("mean", "")),
                    std=_safe_float(found.get("std", "")),
                    n=_safe_int(found.get("n_runs", found.get("n_splits", 0))),
                    ci_lower=_safe_float(found.get("ci_lower", "")),
                    ci_upper=_safe_float(found.get("ci_upper", "")),
                )
            )

        ax.errorbar(
            severity_values,
            [stats.mean for stats in points],
            yerr=_yerr_from_stats(points, use_ci=bool(ci)),
            label=model_id,
            color=colors.get(model_id, None),
            marker="o",
            linewidth=1.8,
            markersize=4,
            capsize=3,
        )

    ax.set_title(title)
    ax.set_xlabel("Severity")
    ax.set_ylabel(metric_key)
    ax.set_xticks(list(severity_values))
    ax.grid(True, alpha=0.25)
    ax.legend(title=legend_title, fontsize=8, title_fontsize=8, loc="best")
    fig.tight_layout()

    out_group_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_group_dir / f"curve__{scenario_id}__{metric_key}.png"
    fig.savefig(out_png)
    plt.close(fig)


def _plot_audit_rows(
    audit_rows: Sequence[dict[str, Any]],
    *,
    out_group_dir: Path,
    dataset_id: str,
    split_label: str,
    protocol: str,
    scenario_id: str,
    scenario_display_name: str,
    oracle_labels: bool,
    audit_metric: str,
    audit_metric_label: str,
    audit_metric_unit: str,
    severity_values: Sequence[float],
    ci: bool,
    plt: Any,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=200)
    title = f"{dataset_id}/{split_label} [{protocol}] - {scenario_display_name}: {audit_metric_label} vs severity"
    ylabel = str(audit_metric_label)
    if audit_metric_unit and audit_metric_unit != "none":
        ylabel = f"{ylabel} [{audit_metric_unit}]"

    points: list[MetricStats] = []
    for severity in severity_values:
        found = None
        for row in audit_rows:
            if (
                str(row.get("audit_metric", "")) == str(audit_metric)
                and float(row.get("severity", 0.0) or 0.0) == float(severity)
            ):
                found = row
                break
        if found is None:
            points.append(MetricStats(float("nan"), float("nan"), 0, float("nan"), float("nan")))
            continue
        points.append(
            MetricStats(
                mean=_safe_float(found.get("mean", "")),
                std=_safe_float(found.get("std", "")),
                n=_safe_int(found.get("n_graphs", found.get("n_splits", 0))),
                ci_lower=_safe_float(found.get("ci_lower", "")),
                ci_upper=_safe_float(found.get("ci_upper", "")),
            )
        )

    ax.errorbar(
        severity_values,
        [stats.mean for stats in points],
        yerr=_yerr_from_stats(points, use_ci=bool(ci)),
        color="#444444",
        marker="o",
        linewidth=1.8,
        markersize=4,
        capsize=3,
    )
    ax.set_title(title)
    ax.set_xlabel("Severity")
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(severity_values))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out_group_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_group_dir / f"audit__{scenario_id}__{audit_metric}.png"
    fig.savefig(out_png)
    plt.close(fig)


def run_plots_stage(
    cfg: dict[str, Any],
    *,
    out_dir: Path,
    include_noop: bool,
    only_clean: bool,
    max_variants: int | None,
    max_training_seeds: int | None,
    ci: bool = False,
) -> None:
    out_dir = out_dir.resolve()
    results_csv = out_dir / "results.csv"
    variants_csv = out_dir / "graph_variants.csv"
    variant_audit_csv = out_dir / "variant_audit.csv"
    rows = _read_results_csv(results_csv)
    ok = [r for r in rows if r.status == "ok"]
    variants = filter_variants(
        read_variants_csv(variants_csv),
        include_noop=bool(include_noop),
        only_clean=bool(only_clean),
        max_variants=max_variants,
    )
    variant_by_key = _index_variants_by_key(variants)
    audit_by_key = _index_audit_rows_by_key(_read_variant_audit_csv(variant_audit_csv))
    _require_audit_keys_for_variants(variant_by_key, audit_by_key)

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    completeness = _completeness_report(
        cfg,
        out_dir=out_dir,
        include_noop=bool(include_noop),
        only_clean=bool(only_clean),
        max_variants=max_variants,
        max_training_seeds=max_training_seeds,
    )
    _print_completeness_report(completeness)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for --stage plots. Install it with: py -m pip install matplotlib") from e

    if ci:
        _require_numpy()

    severity_grid = _severity_grid_from_config(cfg)
    model_ids = [m for m in _model_ids_from_config(cfg) if m in {"mlp", "sage", "pmp", "secgfd"}]
    scenario_meta = _scenario_meta_by_id(cfg)
    protocols_by_dataset_split = _protocols_by_dataset_split(rows, model_ids=model_ids)
    audit_metric_specs = _selected_audit_metric_specs(cfg)
    filtered_audit_rows = [audit_by_key[key] for key in variant_by_key]
    performance_audit_join_rows = _build_performance_audit_join_rows(
        rows,
        model_ids=model_ids,
        variant_by_key=variant_by_key,
        audit_by_key=audit_by_key,
    )
    audit_summary_rows = _build_audit_summary_rows(
        filtered_audit_rows,
        protocols_by_dataset_split=protocols_by_dataset_split,
        metric_specs=audit_metric_specs,
        ci=ci,
    )

    metrics = [
        ("roc_auc", "ROC-AUC"),
        ("average_precision", "Average Precision"),
        ("f1_macro", "F1-macro"),
    ]
    colors = {
        "mlp": "#1f77b4",
        "sage": "#ff7f0e",
        "pmp": "#2ca02c",
        "secgfd": "#d62728",
    }

    summary_rows: list[dict[str, Any]] = []
    drop_rows: list[dict[str, Any]] = []
    robust_rows: list[dict[str, Any]] = []

    ds_splits = sorted(set((r.dataset_id, r.split_id, r.protocol) for r in ok))

    for dataset_id, split_id, protocol in ds_splits:
        ok_g = [r for r in ok if r.dataset_id == dataset_id and r.split_id == split_id and r.protocol == protocol]

        clean_by_model: dict[str, dict[str, MetricStats]] = {}
        clean_rows_by_model: dict[str, dict[str, list[float]]] = {}
        for model_id in model_ids:
            sel = [
                r
                for r in ok_g
                if r.model_id == model_id and r.scenario_id == "clean" and float(r.severity) == 0.0
            ]
            clean_by_model[model_id] = {}
            clean_rows_by_model[model_id] = {}
            for metric_key, _metric_label in metrics:
                vals = _metric_values(sel, metric_key)
                clean_by_model[model_id][metric_key] = _aggregate_metric_stats(vals, ci=ci)
                clean_rows_by_model[model_id][metric_key] = vals

        out_group_dir = plots_dir / protocol / dataset_id / split_id

        for scenario_cfg in cfg.get("scenarios", []):
            scenario_id = str(scenario_cfg.get("scenario_id", ""))
            oracle_labels = _scenario_oracle_labels(scenario_meta, scenario_id)
            scenario_display_name = _scenario_display_name(scenario_meta, scenario_id)
            sevs = severity_grid.get(scenario_id, [])
            sevs_nonzero = [s for s in sevs if float(s) != 0.0]
            sevs_plot = [0.0] + sorted(sevs_nonzero)

            for metric_key, metric_label in metrics:
                plot_rows: list[dict[str, Any]] = []

                for model_id in model_ids:
                    clean_stats = clean_by_model.get(model_id, {}).get(
                        metric_key,
                        MetricStats(float("nan"), float("nan"), 0, float("nan"), float("nan")),
                    )
                    clean_row = {
                        "dataset_id": dataset_id,
                        "split_id": split_id,
                        "scenario_id": scenario_id,
                        "severity": 0.0,
                        "model_id": model_id,
                        "protocol": protocol,
                        "metric": metric_key,
                        "oracle_labels": oracle_labels,
                        "mean": clean_stats.mean,
                        "std": clean_stats.std,
                        "ci_lower": clean_stats.ci_lower,
                        "ci_upper": clean_stats.ci_upper,
                        "n_runs": clean_stats.n,
                        "source": "clean",
                    }
                    summary_rows.append(dict(clean_row))
                    plot_rows.append(dict(clean_row))

                    for severity in sorted(sevs_nonzero):
                        sel = [
                            r
                            for r in ok_g
                            if r.model_id == model_id
                            and r.scenario_id == scenario_id
                            and float(r.severity) == float(severity)
                        ]
                        stats = _aggregate_metric_stats(_metric_values(sel, metric_key), ci=ci)
                        row = {
                            "dataset_id": dataset_id,
                            "split_id": split_id,
                            "scenario_id": scenario_id,
                            "severity": float(severity),
                            "model_id": model_id,
                            "protocol": protocol,
                            "metric": metric_key,
                            "oracle_labels": oracle_labels,
                            "mean": stats.mean,
                            "std": stats.std,
                            "ci_lower": stats.ci_lower,
                            "ci_upper": stats.ci_upper,
                            "n_runs": stats.n,
                            "source": "scenario",
                        }
                        summary_rows.append(dict(row))
                        plot_rows.append(dict(row))

                _plot_summary_rows(
                    plot_rows,
                    out_group_dir=out_group_dir,
                    dataset_id=dataset_id,
                    split_label=split_id,
                    protocol=protocol,
                    scenario_id=scenario_id,
                    scenario_display_name=scenario_display_name,
                    oracle_labels=oracle_labels,
                    metric_key=metric_key,
                    metric_label=metric_label,
                    model_ids=model_ids,
                    severity_values=sevs_plot,
                    colors=colors,
                    ci=ci,
                    plt=plt,
                )

                max_severity = float(max(sevs_nonzero)) if sevs_nonzero else None
                if max_severity is not None:
                    for model_id in model_ids:
                        clean_vals = clean_rows_by_model.get(model_id, {}).get(metric_key, [])
                        stressed_sel = [
                            r
                            for r in ok_g
                            if r.model_id == model_id
                            and r.scenario_id == scenario_id
                            and float(r.severity) == float(max_severity)
                        ]
                        stressed_vals = _metric_values(stressed_sel, metric_key)
                        clean_stats = _aggregate_metric_stats(clean_vals, ci=ci)
                        stressed_stats = _aggregate_metric_stats(stressed_vals, ci=ci)
                        drop_stats = _bootstrap_difference_stats(clean_vals, stressed_vals, ci=ci)
                        drop_rows.append(
                            {
                                "dataset_id": dataset_id,
                                "split_id": split_id,
                                "scenario_id": scenario_id,
                                "model_id": model_id,
                                "protocol": protocol,
                                "metric": metric_key,
                                "max_severity": max_severity,
                                "oracle_labels": oracle_labels,
                                "clean_mean": clean_stats.mean,
                                "clean_std": clean_stats.std,
                                "clean_ci_lower": clean_stats.ci_lower,
                                "clean_ci_upper": clean_stats.ci_upper,
                                "max_severity_mean": stressed_stats.mean,
                                "max_severity_std": stressed_stats.std,
                                "max_severity_ci_lower": stressed_stats.ci_lower,
                                "max_severity_ci_upper": stressed_stats.ci_upper,
                                "drop_mean": drop_stats.mean,
                                "drop_std": drop_stats.std,
                                "drop_ci_lower": drop_stats.ci_lower,
                                "drop_ci_upper": drop_stats.ci_upper,
                                "n_clean_runs": clean_stats.n,
                                "n_max_runs": stressed_stats.n,
                            }
                        )

                if len(sevs_plot) < 2:
                    continue

                x_values = [float(sev) for sev in sevs_plot]
                for model_id in model_ids:
                    values_by_severity = []
                    for severity in sevs_plot:
                        if float(severity) == 0.0:
                            values_by_severity.append(clean_rows_by_model.get(model_id, {}).get(metric_key, []))
                        else:
                            values_by_severity.append(
                                _metric_values(
                                    [
                                        r
                                        for r in ok_g
                                        if r.model_id == model_id
                                        and r.scenario_id == scenario_id
                                        and float(r.severity) == float(severity)
                                    ],
                                    metric_key,
                                )
                            )
                    auc_stats, avg_stats = _bootstrap_robustness_stats(x_values, values_by_severity, ci=ci)
                    robust_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "split_id": split_id,
                            "scenario_id": scenario_id,
                            "model_id": model_id,
                            "protocol": protocol,
                            "metric": metric_key,
                            "oracle_labels": oracle_labels,
                            "robustness_auc_mean": auc_stats.mean,
                            "robustness_auc_std": auc_stats.std,
                            "robustness_auc_ci_lower": auc_stats.ci_lower,
                            "robustness_auc_ci_upper": auc_stats.ci_upper,
                            "robustness_avg_metric_mean": avg_stats.mean,
                            "robustness_avg_metric_std": avg_stats.std,
                            "robustness_avg_metric_ci_lower": avg_stats.ci_lower,
                            "robustness_avg_metric_ci_upper": avg_stats.ci_upper,
                            "min_n_runs_per_severity": avg_stats.n,
                        }
                    )

    _write_csv(
        plots_dir / "summary_curves.csv",
        [
            "dataset_id",
            "split_id",
            "scenario_id",
            "severity",
            "model_id",
            "protocol",
            "metric",
            "oracle_labels",
            "mean",
            "std",
            "ci_lower",
            "ci_upper",
            "n_runs",
            "source",
        ],
        summary_rows,
    )
    _write_csv(
        plots_dir / "performance_drop_max_stress.csv",
        [
            "dataset_id",
            "split_id",
            "scenario_id",
            "model_id",
            "protocol",
            "metric",
            "max_severity",
            "oracle_labels",
            "clean_mean",
            "clean_std",
            "clean_ci_lower",
            "clean_ci_upper",
            "max_severity_mean",
            "max_severity_std",
            "max_severity_ci_lower",
            "max_severity_ci_upper",
            "drop_mean",
            "drop_std",
            "drop_ci_lower",
            "drop_ci_upper",
            "n_clean_runs",
            "n_max_runs",
        ],
        drop_rows,
    )
    _write_csv(
        plots_dir / "robustness_scores.csv",
        [
            "dataset_id",
            "split_id",
            "scenario_id",
            "model_id",
            "protocol",
            "metric",
            "oracle_labels",
            "robustness_auc_mean",
            "robustness_auc_std",
            "robustness_auc_ci_lower",
            "robustness_auc_ci_upper",
            "robustness_avg_metric_mean",
            "robustness_avg_metric_std",
            "robustness_avg_metric_ci_lower",
            "robustness_avg_metric_ci_upper",
            "min_n_runs_per_severity",
        ],
        robust_rows,
    )
    _write_csv(
        plots_dir / "performance_audit_join.csv",
        [
            "experiment_name",
            "dataset_id",
            "split_id",
            "scenario_id",
            "severity",
            "graph_seed",
            "training_seed",
            "model_id",
            "protocol",
            "status",
            "oracle_labels",
            "scenario_applied",
            "graph_view_mode",
            "scenario_family",
            "scenario_method",
            "severity_param",
            "requested_severity",
            "requested_change",
            "requested_change_unit",
            "realized_change",
            "realized_change_unit",
            "roc_auc",
            "average_precision",
            "f1_macro",
            "base_graph_path",
            "graph_path",
            "variant_n_nodes",
            "variant_n_edges",
            "variant_heterophily_ratio",
            "variant_pos_rate",
            "heterophily_ratio_before",
            "heterophily_ratio_after",
            "mean_cosine_to_sampled_normal_before",
            "mean_cosine_to_sampled_normal_after",
            "fraud_to_normal_neighbor_ratio_before",
            "fraud_to_normal_neighbor_ratio_after",
            "n_added_edge_pairs_actual",
            "n_rewired_edges_actual",
        ],
        performance_audit_join_rows,
    )
    _write_csv(
        plots_dir / "audit_curves.csv",
        [
            "dataset_id",
            "split_id",
            "scenario_id",
            "severity",
            "protocol",
            "audit_metric",
            "audit_metric_label",
            "audit_metric_unit",
            "oracle_labels",
            "graph_view_mode",
            "mean",
            "std",
            "ci_lower",
            "ci_upper",
            "n_graphs",
            "source",
        ],
        audit_summary_rows,
    )

    cross_split_summary_rows = _build_cross_split_summary_rows(summary_rows, ci=ci)
    cross_split_drop_rows = _build_cross_split_drop_rows(drop_rows, ci=ci)
    cross_split_robust_rows = _build_cross_split_robust_rows(robust_rows, ci=ci)
    cross_split_audit_rows = _build_cross_split_audit_rows(audit_summary_rows, ci=ci)

    _write_csv(
        plots_dir / "summary_curves_cross_split.csv",
        [
            "dataset_id",
            "scenario_id",
            "severity",
            "model_id",
            "protocol",
            "metric",
            "oracle_labels",
            "mean",
            "std",
            "ci_lower",
            "ci_upper",
            "n_splits",
            "source",
        ],
        cross_split_summary_rows,
    )
    _write_csv(
        plots_dir / "performance_drop_max_stress_cross_split.csv",
        [
            "dataset_id",
            "scenario_id",
            "model_id",
            "protocol",
            "metric",
            "max_severity",
            "oracle_labels",
            "clean_mean",
            "clean_std",
            "clean_ci_lower",
            "clean_ci_upper",
            "max_severity_mean",
            "max_severity_std",
            "max_severity_ci_lower",
            "max_severity_ci_upper",
            "drop_mean",
            "drop_std",
            "drop_ci_lower",
            "drop_ci_upper",
            "n_splits",
        ],
        cross_split_drop_rows,
    )
    _write_csv(
        plots_dir / "robustness_scores_cross_split.csv",
        [
            "dataset_id",
            "scenario_id",
            "model_id",
            "protocol",
            "metric",
            "oracle_labels",
            "robustness_auc_mean",
            "robustness_auc_std",
            "robustness_auc_ci_lower",
            "robustness_auc_ci_upper",
            "robustness_avg_metric_mean",
            "robustness_avg_metric_std",
            "robustness_avg_metric_ci_lower",
            "robustness_avg_metric_ci_upper",
            "n_splits",
        ],
        cross_split_robust_rows,
    )
    _write_csv(
        plots_dir / "audit_curves_cross_split.csv",
        [
            "dataset_id",
            "scenario_id",
            "severity",
            "protocol",
            "audit_metric",
            "audit_metric_label",
            "audit_metric_unit",
            "oracle_labels",
            "graph_view_mode",
            "mean",
            "std",
            "ci_lower",
            "ci_upper",
            "n_splits",
            "source",
        ],
        cross_split_audit_rows,
    )

    split_counts_by_dataset_protocol: dict[tuple[str, str], set[str]] = {}
    for row in summary_rows:
        key = (str(row.get("dataset_id", "")), str(row.get("protocol", "")))
        split_counts_by_dataset_protocol.setdefault(key, set()).add(str(row.get("split_id", "")))

    for dataset_id, protocol in sorted(split_counts_by_dataset_protocol):
        if len(split_counts_by_dataset_protocol[(dataset_id, protocol)]) <= 1:
            continue
        dataset_rows = [
            row
            for row in cross_split_summary_rows
            if str(row.get("dataset_id", "")) == dataset_id and str(row.get("protocol", "")) == protocol
        ]
        out_group_dir = plots_dir / protocol / dataset_id / "across_splits"
        for scenario_cfg in cfg.get("scenarios", []):
            scenario_id = str(scenario_cfg.get("scenario_id", ""))
            oracle_labels = _scenario_oracle_labels(scenario_meta, scenario_id)
            scenario_display_name = _scenario_display_name(scenario_meta, scenario_id)
            sevs = severity_grid.get(scenario_id, [])
            sevs_nonzero = [s for s in sevs if float(s) != 0.0]
            sevs_plot = [0.0] + sorted(sevs_nonzero)
            scenario_rows = [row for row in dataset_rows if str(row.get("scenario_id", "")) == scenario_id]
            if not scenario_rows:
                continue
            for metric_key, metric_label in metrics:
                metric_rows = [row for row in scenario_rows if str(row.get("metric", "")) == metric_key]
                if not metric_rows:
                    continue
                _plot_summary_rows(
                    metric_rows,
                    out_group_dir=out_group_dir,
                    dataset_id=dataset_id,
                    split_label="across_splits",
                    protocol=protocol,
                    scenario_id=scenario_id,
                    scenario_display_name=scenario_display_name,
                    oracle_labels=oracle_labels,
                    metric_key=metric_key,
                    metric_label=metric_label,
                    model_ids=model_ids,
                    severity_values=sevs_plot,
                    colors=colors,
                    ci=ci,
                    plt=plt,
                )

    for (dataset_id, split_id), protocols in sorted(protocols_by_dataset_split.items()):
        for protocol in sorted(protocols):
            out_group_dir = plots_dir / protocol / dataset_id / split_id
            dataset_rows = [
                row
                for row in audit_summary_rows
                if str(row.get("dataset_id", "")) == dataset_id
                and str(row.get("split_id", "")) == split_id
                and str(row.get("protocol", "")) == protocol
            ]
            for scenario_cfg in cfg.get("scenarios", []):
                scenario_id = str(scenario_cfg.get("scenario_id", ""))
                oracle_labels = _scenario_oracle_labels(scenario_meta, scenario_id)
                scenario_display_name = _scenario_display_name(scenario_meta, scenario_id)
                scenario_rows = [row for row in dataset_rows if str(row.get("scenario_id", "")) == scenario_id]
                if not scenario_rows:
                    continue
                for audit_metric in sorted({str(row.get("audit_metric", "")) for row in scenario_rows}):
                    metric_rows = [row for row in scenario_rows if str(row.get("audit_metric", "")) == audit_metric]
                    if not metric_rows:
                        continue
                    severity_values = sorted({float(row.get("severity", 0.0) or 0.0) for row in metric_rows})
                    _plot_audit_rows(
                        metric_rows,
                        out_group_dir=out_group_dir,
                        dataset_id=dataset_id,
                        split_label=split_id,
                        protocol=protocol,
                        scenario_id=scenario_id,
                        scenario_display_name=scenario_display_name,
                        oracle_labels=oracle_labels,
                        audit_metric=audit_metric,
                        audit_metric_label=str(metric_rows[0].get("audit_metric_label", audit_metric)),
                        audit_metric_unit=str(metric_rows[0].get("audit_metric_unit", "")),
                        severity_values=severity_values,
                        ci=ci,
                        plt=plt,
                    )

    split_counts_by_dataset_protocol_audit: dict[tuple[str, str], set[str]] = {}
    for row in audit_summary_rows:
        key = (str(row.get("dataset_id", "")), str(row.get("protocol", "")))
        split_counts_by_dataset_protocol_audit.setdefault(key, set()).add(str(row.get("split_id", "")))

    for dataset_id, protocol in sorted(split_counts_by_dataset_protocol_audit):
        if len(split_counts_by_dataset_protocol_audit[(dataset_id, protocol)]) <= 1:
            continue
        dataset_rows = [
            row
            for row in cross_split_audit_rows
            if str(row.get("dataset_id", "")) == dataset_id and str(row.get("protocol", "")) == protocol
        ]
        out_group_dir = plots_dir / protocol / dataset_id / "across_splits"
        for scenario_cfg in cfg.get("scenarios", []):
            scenario_id = str(scenario_cfg.get("scenario_id", ""))
            oracle_labels = _scenario_oracle_labels(scenario_meta, scenario_id)
            scenario_display_name = _scenario_display_name(scenario_meta, scenario_id)
            scenario_rows = [row for row in dataset_rows if str(row.get("scenario_id", "")) == scenario_id]
            if not scenario_rows:
                continue
            for audit_metric in sorted({str(row.get("audit_metric", "")) for row in scenario_rows}):
                metric_rows = [row for row in scenario_rows if str(row.get("audit_metric", "")) == audit_metric]
                if not metric_rows:
                    continue
                severity_values = sorted({float(row.get("severity", 0.0) or 0.0) for row in metric_rows})
                _plot_audit_rows(
                    metric_rows,
                    out_group_dir=out_group_dir,
                    dataset_id=dataset_id,
                    split_label="across_splits",
                    protocol=protocol,
                    scenario_id=scenario_id,
                    scenario_display_name=scenario_display_name,
                    oracle_labels=oracle_labels,
                    audit_metric=audit_metric,
                    audit_metric_label=str(metric_rows[0].get("audit_metric_label", audit_metric)),
                    audit_metric_unit=str(metric_rows[0].get("audit_metric_unit", "")),
                    severity_values=severity_values,
                    ci=ci,
                    plt=plt,
                )
