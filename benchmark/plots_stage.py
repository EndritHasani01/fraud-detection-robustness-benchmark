from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .config import get_training_seeds, scenario_oracle_labels
from .results import (
    PROTOCOL_TRAIN_CLEAN_EVAL_ALL,
    PROTOCOL_TRAIN_ON_VARIANT,
    normalize_protocol,
)
from .variants import filter_variants, read_variants_csv


BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
CI_METHOD_NOT_REQUESTED = "not_requested"
CI_METHOD_CROSSED_SEEDS = "crossed_training_graph_seed_bootstrap"
CI_METHOD_PAIRED_DROP = "paired_drop_crossed_training_graph_seed_bootstrap"
CI_METHOD_PAIRED_RETENTION = "paired_retention_crossed_training_graph_seed_bootstrap"
CI_METHOD_PAIRED_CURVE = "paired_curve_crossed_training_graph_seed_bootstrap"
CI_METHOD_PAIRED_PROTOCOL = "paired_protocol_crossed_training_graph_seed_bootstrap"
CI_METHOD_GRAPH_SEED = "graph_seed_bootstrap"
CI_METHOD_SPLIT = "split_bootstrap_over_split_means"
UNSPECIFIED_SPLIT_REGIME_ID = "allocation_unspecified"


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
class SeededMetricStats:
    """Metric summary whose uncertainty respects the crossed seed design."""

    stats: MetricStats
    n_training_seeds: int
    n_graph_seeds: int
    n_seed_cells: int
    ci_method: str


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


@dataclass(frozen=True)
class SplitAllocation:
    split_regime_id: str
    train_size: float | None
    val_size: float | None
    test_size: float | None


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


def _format_allocation_rate(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def _split_allocations_from_config(cfg: Mapping[str, Any]) -> dict[str, SplitAllocation]:
    """Map split IDs to comparable train/validation/test allocation regimes."""

    out: dict[str, SplitAllocation] = {}
    data_splits = cfg.get("data_splits", [])
    if not isinstance(data_splits, list):
        return out

    for split_cfg in data_splits:
        if not isinstance(split_cfg, Mapping):
            continue
        split_id = str(split_cfg.get("split_id", "")).strip()
        if not split_id:
            continue
        try:
            train_size = float(split_cfg["train_size"])
            val_size = float(split_cfg["val_size"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(train_size) and math.isfinite(val_size)):
            continue
        test_size = round(1.0 - train_size - val_size, 12)
        explicit_regime_id = str(split_cfg.get("split_regime_id", "")).strip()
        regime_id = explicit_regime_id or (
            f"train_{_format_allocation_rate(train_size)}_"
            f"val_{_format_allocation_rate(val_size)}_"
            f"test_{_format_allocation_rate(test_size)}"
        )
        out[split_id] = SplitAllocation(
            split_regime_id=regime_id,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )
    return out


def _allocation_for_split(
    split_id: Any,
    split_allocations: Mapping[str, SplitAllocation],
) -> SplitAllocation:
    return split_allocations.get(
        str(split_id),
        SplitAllocation(
            split_regime_id=UNSPECIFIED_SPLIT_REGIME_ID,
            train_size=None,
            val_size=None,
            test_size=None,
        ),
    )


def _allocation_fields(allocation: SplitAllocation) -> dict[str, Any]:
    return {
        "split_regime_id": allocation.split_regime_id,
        "train_size": "" if allocation.train_size is None else allocation.train_size,
        "val_size": "" if allocation.val_size is None else allocation.val_size,
        "test_size": "" if allocation.test_size is None else allocation.test_size,
    }


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


def _ci_method(enabled: bool, method: str) -> str:
    return str(method) if bool(enabled) else CI_METHOD_NOT_REQUESTED


def _empty_seeded_stats(*, ci: bool, method: str) -> SeededMetricStats:
    return SeededMetricStats(
        stats=MetricStats(
            mean=float("nan"),
            std=float("nan"),
            n=0,
            ci_lower=float("nan"),
            ci_upper=float("nan"),
        ),
        n_training_seeds=0,
        n_graph_seeds=0,
        n_seed_cells=0,
        ci_method=_ci_method(ci, method),
    )


def _seed_cell_values(rows: Iterable[ResultRow], metric_key: str) -> dict[tuple[int, int], float]:
    """Collapse duplicate ledger rows into one value per crossed seed cell."""

    grouped: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        value = float(getattr(row, metric_key))
        if not math.isfinite(value):
            continue
        grouped.setdefault((int(row.training_seed), int(row.graph_seed)), []).append(value)
    return {
        key: float(sum(values) / len(values))
        for key, values in grouped.items()
        if values
    }


def _values_by_training_seed(rows: Iterable[ResultRow], metric_key: str) -> dict[int, float]:
    """Aggregate graph realizations within each training seed."""

    grouped: dict[int, list[float]] = {}
    for (_training_seed, _graph_seed), value in _seed_cell_values(rows, metric_key).items():
        grouped.setdefault(int(_training_seed), []).append(float(value))
    return {
        training_seed: float(sum(values) / len(values))
        for training_seed, values in grouped.items()
        if values
    }


def _crossed_seed_bootstrap_means(
    cell_values: Mapping[tuple[int, int], float],
    *,
    num_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[float]:
    """Bootstrap both crossed random axes, never the flattened result rows."""

    if not cell_values:
        return []
    np = _require_numpy()
    training_seeds = sorted({int(key[0]) for key in cell_values})
    graph_seeds = sorted({int(key[1]) for key in cell_values})
    rng = np.random.default_rng(int(seed))
    samples: list[float] = []
    for _ in range(int(num_resamples)):
        sampled_training = rng.choice(training_seeds, size=len(training_seeds), replace=True)
        sampled_graph = rng.choice(graph_seeds, size=len(graph_seeds), replace=True)
        values = [
            float(cell_values[(int(training_seed), int(graph_seed))])
            for training_seed in sampled_training
            for graph_seed in sampled_graph
            if (int(training_seed), int(graph_seed)) in cell_values
        ]
        if values:
            samples.append(float(sum(values) / len(values)))
    return samples


def _seeded_stats_from_cells(
    cell_values: Mapping[tuple[int, int], float],
    *,
    ci: bool,
    method: str = CI_METHOD_CROSSED_SEEDS,
) -> SeededMetricStats:
    finite_cells = {
        (int(training_seed), int(graph_seed)): float(value)
        for (training_seed, graph_seed), value in cell_values.items()
        if math.isfinite(float(value))
    }
    if not finite_cells:
        return _empty_seeded_stats(ci=ci, method=method)

    raw_stats = _aggregate_metric_stats(list(finite_cells.values()), ci=False)
    samples: list[float] = []
    if ci:
        samples = _crossed_seed_bootstrap_means(finite_cells)

    if ci and samples:
        np = _require_numpy()
        ci_lower, ci_upper = np.percentile(np.asarray(samples, dtype=np.float64), [2.5, 97.5])
    else:
        ci_lower = ci_upper = float("nan")

    return SeededMetricStats(
        stats=MetricStats(
            mean=raw_stats.mean,
            std=raw_stats.std,
            n=raw_stats.n,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
        ),
        n_training_seeds=len({key[0] for key in finite_cells}),
        n_graph_seeds=len({key[1] for key in finite_cells}),
        n_seed_cells=len(finite_cells),
        ci_method=_ci_method(ci, method),
    )


def _aggregate_seeded_metric_stats(
    rows: Iterable[ResultRow],
    metric_key: str,
    *,
    ci: bool,
) -> SeededMetricStats:
    return _seeded_stats_from_cells(_seed_cell_values(rows, metric_key), ci=ci)


def _paired_clean_stress_stats(
    clean_rows: Iterable[ResultRow],
    stressed_rows: Iterable[ResultRow],
    metric_key: str,
    *,
    ci: bool,
) -> SeededMetricStats:
    """Compute clean-minus-stress after pairing by training seed.

    The clean graph has one graph identifier while stressed runs have one or
    more graph seeds. Replicating each clean value only within its matching
    training seed preserves both the training and graph clusters and makes an
    exact per-seed invariant exactly zero under every bootstrap resample.
    """

    clean_by_training = _values_by_training_seed(clean_rows, metric_key)
    stressed_cells = _seed_cell_values(stressed_rows, metric_key)
    paired_cells = {
        (training_seed, graph_seed): float(clean_by_training[training_seed] - stressed_value)
        for (training_seed, graph_seed), stressed_value in stressed_cells.items()
        if training_seed in clean_by_training
    }
    return _seeded_stats_from_cells(
        paired_cells,
        ci=ci,
        method=CI_METHOD_PAIRED_DROP,
    )


def _paired_retention_stats(
    clean_rows: Iterable[ResultRow],
    stressed_rows: Iterable[ResultRow],
    metric_key: str,
    *,
    ci: bool,
) -> SeededMetricStats:
    """Compute stressed/clean retention within matching training and graph cells."""

    clean_by_training = _values_by_training_seed(clean_rows, metric_key)
    stressed_cells = _seed_cell_values(stressed_rows, metric_key)
    retention_cells = {
        (training_seed, graph_seed): float(stressed_value / clean_by_training[training_seed])
        for (training_seed, graph_seed), stressed_value in stressed_cells.items()
        if training_seed in clean_by_training
        and math.isfinite(clean_by_training[training_seed])
        and abs(clean_by_training[training_seed]) > 1e-15
    }
    return _seeded_stats_from_cells(
        retention_cells,
        ci=ci,
        method=CI_METHOD_PAIRED_RETENTION,
    )


def _paired_same_cell_contrast_stats(
    left_rows: Iterable[ResultRow],
    right_rows: Iterable[ResultRow],
    metric_key: str,
    *,
    ci: bool,
) -> tuple[SeededMetricStats, SeededMetricStats, SeededMetricStats]:
    left_cells = _seed_cell_values(left_rows, metric_key)
    right_cells = _seed_cell_values(right_rows, metric_key)
    paired_keys = sorted(set(left_cells) & set(right_cells))
    paired_left = {key: left_cells[key] for key in paired_keys}
    paired_right = {key: right_cells[key] for key in paired_keys}
    contrast = {key: float(left_cells[key] - right_cells[key]) for key in paired_keys}
    return (
        _seeded_stats_from_cells(paired_left, ci=ci),
        _seeded_stats_from_cells(paired_right, ci=ci),
        _seeded_stats_from_cells(
            contrast,
            ci=ci,
            method=CI_METHOD_PAIRED_PROTOCOL,
        ),
    )


def _paired_curve_robustness_stats(
    x_values: Sequence[float],
    rows_by_severity: Sequence[Sequence[ResultRow]],
    metric_key: str,
    *,
    ci: bool,
) -> tuple[SeededMetricStats, SeededMetricStats, int]:
    """Summarize curve AUC from complete, seed-paired trajectories."""

    if len(x_values) < 2 or len(x_values) != len(rows_by_severity):
        empty = _empty_seeded_stats(ci=ci, method=CI_METHOD_PAIRED_CURVE)
        return empty, empty, 0
    x = [float(value) for value in x_values]
    x_range = max(x) - min(x)
    if x_range <= 0.0:
        empty = _empty_seeded_stats(ci=ci, method=CI_METHOD_PAIRED_CURVE)
        return empty, empty, 0

    clean_by_training = _values_by_training_seed(rows_by_severity[0], metric_key)
    stressed_by_severity = [_seed_cell_values(rows, metric_key) for rows in rows_by_severity[1:]]
    if not clean_by_training or any(not cells for cells in stressed_by_severity):
        empty = _empty_seeded_stats(ci=ci, method=CI_METHOD_PAIRED_CURVE)
        return empty, empty, 0

    complete_keys = set(stressed_by_severity[0])
    for cells in stressed_by_severity[1:]:
        complete_keys &= set(cells)
    complete_keys = {key for key in complete_keys if key[0] in clean_by_training}

    auc_cells: dict[tuple[int, int], float] = {}
    avg_cells: dict[tuple[int, int], float] = {}
    for key in sorted(complete_keys):
        curve = [clean_by_training[key[0]]] + [cells[key] for cells in stressed_by_severity]
        auc = float(
            sum(
                (x[idx + 1] - x[idx]) * (curve[idx + 1] + curve[idx]) * 0.5
                for idx in range(len(x) - 1)
            )
        )
        auc_cells[key] = auc
        avg_cells[key] = float(auc / x_range)

    min_runs = min(
        [len(clean_by_training)] + [len(cells) for cells in stressed_by_severity]
    )
    return (
        _seeded_stats_from_cells(
            auc_cells,
            ci=ci,
            method=CI_METHOD_PAIRED_CURVE,
        ),
        _seeded_stats_from_cells(
            avg_cells,
            ci=ci,
            method=CI_METHOD_PAIRED_CURVE,
        ),
        int(min_runs),
    )


def _scenario_oracle_labels(scenario_meta: dict[str, dict[str, Any]], scenario_id: str) -> bool:
    meta = scenario_meta.get(str(scenario_id), {})
    return bool(meta.get("oracle_labels", False))


def _scenario_display_name(scenario_meta: dict[str, dict[str, Any]], scenario_id: str) -> str:
    meta = scenario_meta.get(str(scenario_id), {})
    return str(meta.get("display_name", str(scenario_id)))


def _claim_scope(*, oracle_labels: bool, protocol: str) -> str:
    if not bool(oracle_labels):
        return "non_oracle_controlled_stress"
    if normalize_protocol(protocol) == PROTOCOL_TRAIN_CLEAN_EVAL_ALL:
        return "oracle_shift_sensitivity_diagnostic"
    return "oracle_privileged_training_diagnostic"


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
                "claim_scope": _claim_scope(oracle_labels=bool(oracle_labels), protocol=protocol),
                "operational_ranking_eligible": not bool(oracle_labels),
                "graph_view_mode": graph_view_mode,
                "mean": stats.mean,
                "std": stats.std,
                "ci_lower": stats.ci_lower,
                "ci_upper": stats.ci_upper,
                "n_graphs": stats.n,
                "n_graph_seeds": stats.n,
                "ci_method": _ci_method(ci, CI_METHOD_GRAPH_SEED),
                "source": source,
            }
        )
    return out


def _build_cross_split_audit_rows(
    audit_rows: Sequence[dict[str, Any]],
    *,
    split_allocations: Mapping[str, SplitAllocation],
    ci: bool,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, float, str, str, str, str, bool, str, str], list[dict[str, Any]]] = {}
    for row in audit_rows:
        allocation = _allocation_for_split(row.get("split_id", ""), split_allocations)
        key = (
            allocation.split_regime_id,
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
            split_regime_id,
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
        allocation = next(
            _allocation_for_split(row.get("split_id", ""), split_allocations)
            for row in groups[key]
        )
        if allocation.split_regime_id != split_regime_id:
            raise RuntimeError("Cross-split audit allocation regime mismatch")
        split_means = [_safe_float(row.get("mean", "")) for row in groups[key]]
        stats = _aggregate_metric_stats(split_means, ci=ci)
        out.append(
            {
                **_allocation_fields(allocation),
                "dataset_id": dataset_id,
                "scenario_id": scenario_id,
                "severity": float(severity),
                "protocol": protocol,
                "audit_metric": metric_id,
                "audit_metric_label": metric_label,
                "audit_metric_unit": metric_unit,
                "oracle_labels": bool(oracle_labels),
                "claim_scope": _claim_scope(oracle_labels=bool(oracle_labels), protocol=protocol),
                "operational_ranking_eligible": not bool(oracle_labels),
                "graph_view_mode": graph_view_mode,
                "mean": stats.mean,
                "std": stats.std,
                "ci_lower": stats.ci_lower,
                "ci_upper": stats.ci_upper,
                "n_splits": stats.n,
                "min_graph_seeds_per_split": min(
                    (_safe_int(row.get("n_graph_seeds", row.get("n_graphs", 0))) for row in groups[key]),
                    default=0,
                ),
                "ci_method": _ci_method(ci, CI_METHOD_SPLIT),
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


def _result_rows_for_point(
    rows: Sequence[ResultRow],
    *,
    model_id: str,
    protocol: str,
    scenario_id: str,
    severity: float,
) -> list[ResultRow]:
    return [
        row
        for row in rows
        if row.model_id == model_id
        and row.protocol == protocol
        and row.scenario_id == scenario_id
        and float(row.severity) == float(severity)
    ]


def _build_protocol_contrast_rows(
    rows: Sequence[ResultRow],
    *,
    severity_grid: Mapping[str, Sequence[float]],
    scenario_meta: Mapping[str, Mapping[str, Any]],
    model_ids: Sequence[str],
    metric_keys: Sequence[str],
    ci: bool,
) -> list[dict[str, Any]]:
    """Pair train-on-variant minus clean-train runs on both seed axes."""

    grouped: dict[tuple[str, str], list[ResultRow]] = {}
    for row in rows:
        grouped.setdefault((row.dataset_id, row.split_id), []).append(row)

    out: list[dict[str, Any]] = []
    for (dataset_id, split_id), group_rows in sorted(grouped.items()):
        protocols = {row.protocol for row in group_rows}
        if not {PROTOCOL_TRAIN_ON_VARIANT, PROTOCOL_TRAIN_CLEAN_EVAL_ALL}.issubset(protocols):
            continue
        for scenario_id, configured_severities in severity_grid.items():
            oracle_labels = bool(scenario_meta.get(scenario_id, {}).get("oracle_labels", False))
            severities = [0.0] + sorted(
                {float(value) for value in configured_severities if float(value) != 0.0}
            )
            for severity in severities:
                point_scenario = "clean" if float(severity) == 0.0 else str(scenario_id)
                source = "clean" if float(severity) == 0.0 else "scenario"
                for model_id in model_ids:
                    left_rows = _result_rows_for_point(
                        group_rows,
                        model_id=model_id,
                        protocol=PROTOCOL_TRAIN_ON_VARIANT,
                        scenario_id=point_scenario,
                        severity=severity,
                    )
                    right_rows = _result_rows_for_point(
                        group_rows,
                        model_id=model_id,
                        protocol=PROTOCOL_TRAIN_CLEAN_EVAL_ALL,
                        scenario_id=point_scenario,
                        severity=severity,
                    )
                    for metric_key in metric_keys:
                        left_stats, right_stats, contrast_stats = _paired_same_cell_contrast_stats(
                            left_rows,
                            right_rows,
                            metric_key,
                            ci=ci,
                        )
                        if contrast_stats.stats.n <= 0:
                            continue
                        out.append(
                            {
                                "dataset_id": dataset_id,
                                "split_id": split_id,
                                "scenario_id": scenario_id,
                                "severity": float(severity),
                                "model_id": model_id,
                                "metric": metric_key,
                                "oracle_labels": oracle_labels,
                                "claim_scope": (
                                    "oracle_protocol_comparison_diagnostic"
                                    if oracle_labels
                                    else "non_oracle_controlled_stress"
                                ),
                                "left_claim_scope": _claim_scope(
                                    oracle_labels=oracle_labels,
                                    protocol=PROTOCOL_TRAIN_ON_VARIANT,
                                ),
                                "right_claim_scope": _claim_scope(
                                    oracle_labels=oracle_labels,
                                    protocol=PROTOCOL_TRAIN_CLEAN_EVAL_ALL,
                                ),
                                "operational_ranking_eligible": not oracle_labels,
                                "left_protocol": PROTOCOL_TRAIN_ON_VARIANT,
                                "right_protocol": PROTOCOL_TRAIN_CLEAN_EVAL_ALL,
                                "contrast_definition": (
                                    f"{PROTOCOL_TRAIN_ON_VARIANT} - {PROTOCOL_TRAIN_CLEAN_EVAL_ALL}"
                                ),
                                "left_mean": left_stats.stats.mean,
                                "right_mean": right_stats.stats.mean,
                                "contrast_mean": contrast_stats.stats.mean,
                                "contrast_std": contrast_stats.stats.std,
                                "contrast_ci_lower": contrast_stats.stats.ci_lower,
                                "contrast_ci_upper": contrast_stats.stats.ci_upper,
                                "n_pairs": contrast_stats.stats.n,
                                "n_training_seeds": contrast_stats.n_training_seeds,
                                "n_graph_seeds": contrast_stats.n_graph_seeds,
                                "ci_method": contrast_stats.ci_method,
                                "source": source,
                            }
                        )
    return out


def _build_worst_case_rows(
    summary_rows: Sequence[dict[str, Any]],
    result_rows: Sequence[ResultRow],
    *,
    ci: bool,
) -> list[dict[str, Any]]:
    """Select the lowest observed non-clean point per model/protocol/metric."""

    candidates: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        mean_value = _safe_float(row.get("mean", ""))
        if str(row.get("source", "")) != "scenario" or not math.isfinite(mean_value):
            continue
        key = (
            str(row.get("dataset_id", "")),
            str(row.get("split_id", "")),
            str(row.get("model_id", "")),
            str(row.get("protocol", "")),
            str(row.get("metric", "")),
        )
        candidates.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for key in sorted(candidates):
        dataset_id, split_id, model_id, protocol, metric_key = key
        selected = min(
            candidates[key],
            key=lambda row: (
                _safe_float(row.get("mean", "")),
                str(row.get("scenario_id", "")),
                float(row.get("severity", 0.0) or 0.0),
            ),
        )
        scenario_id = str(selected.get("scenario_id", ""))
        severity = float(selected.get("severity", 0.0) or 0.0)
        group_rows = [
            row
            for row in result_rows
            if row.dataset_id == dataset_id
            and row.split_id == split_id
            and row.model_id == model_id
            and row.protocol == protocol
        ]
        clean_rows = _result_rows_for_point(
            group_rows,
            model_id=model_id,
            protocol=protocol,
            scenario_id="clean",
            severity=0.0,
        )
        stressed_rows = _result_rows_for_point(
            group_rows,
            model_id=model_id,
            protocol=protocol,
            scenario_id=scenario_id,
            severity=severity,
        )
        clean_stats = _aggregate_seeded_metric_stats(clean_rows, metric_key, ci=ci)
        drop_stats = _paired_clean_stress_stats(clean_rows, stressed_rows, metric_key, ci=ci)
        retention_stats = _paired_retention_stats(clean_rows, stressed_rows, metric_key, ci=ci)
        out.append(
            {
                "dataset_id": dataset_id,
                "split_id": split_id,
                "model_id": model_id,
                "protocol": protocol,
                "metric": metric_key,
                "worst_scenario_id": scenario_id,
                "worst_severity": severity,
                "oracle_labels": bool(selected.get("oracle_labels", False)),
                "claim_scope": _claim_scope(
                    oracle_labels=bool(selected.get("oracle_labels", False)),
                    protocol=protocol,
                ),
                "operational_ranking_eligible": not bool(selected.get("oracle_labels", False)),
                "worst_mean": _safe_float(selected.get("mean", "")),
                "worst_std": _safe_float(selected.get("std", "")),
                "worst_ci_lower": _safe_float(selected.get("ci_lower", "")),
                "worst_ci_upper": _safe_float(selected.get("ci_upper", "")),
                "clean_mean": clean_stats.stats.mean,
                "drop_from_clean_mean": drop_stats.stats.mean,
                "drop_from_clean_std": drop_stats.stats.std,
                "drop_from_clean_ci_lower": drop_stats.stats.ci_lower,
                "drop_from_clean_ci_upper": drop_stats.stats.ci_upper,
                "retention_fraction_mean": retention_stats.stats.mean,
                "retention_fraction_std": retention_stats.stats.std,
                "retention_fraction_ci_lower": retention_stats.stats.ci_lower,
                "retention_fraction_ci_upper": retention_stats.stats.ci_upper,
                "n_runs": _safe_int(selected.get("n_runs", 0)),
                "n_training_seeds": _safe_int(selected.get("n_training_seeds", 0)),
                "n_graph_seeds": _safe_int(selected.get("n_graph_seeds", 0)),
                "n_paired_seed_cells": drop_stats.n_seed_cells,
                "ci_method": str(selected.get("ci_method", CI_METHOD_NOT_REQUESTED)),
                "drop_ci_method": drop_stats.ci_method,
                "retention_ci_method": retention_stats.ci_method,
                "selection_method": "minimum_configured_nonzero_point_mean",
            }
        )
    return out


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


def _build_cross_split_summary_rows(
    summary_rows: Sequence[dict[str, Any]],
    *,
    split_allocations: Mapping[str, SplitAllocation],
    ci: bool,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, float, str, str, str, bool, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        allocation = _allocation_for_split(row.get("split_id", ""), split_allocations)
        key = (
            allocation.split_regime_id,
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
        split_regime_id, dataset_id, scenario_id, severity, model_id, protocol, metric, oracle_labels, source = key
        rows = groups[key]
        allocation = _allocation_for_split(rows[0].get("split_id", ""), split_allocations)
        if allocation.split_regime_id != split_regime_id:
            raise RuntimeError("Cross-split summary allocation regime mismatch")
        split_means = [_safe_float(row.get("mean", "")) for row in rows]
        stats = _aggregate_metric_stats(split_means, ci=ci)
        out.append(
            {
                **_allocation_fields(allocation),
                "dataset_id": dataset_id,
                "scenario_id": scenario_id,
                "severity": float(severity),
                "model_id": model_id,
                "protocol": protocol,
                "metric": metric,
                "oracle_labels": bool(oracle_labels),
                "claim_scope": _claim_scope(oracle_labels=bool(oracle_labels), protocol=protocol),
                "operational_ranking_eligible": not bool(oracle_labels),
                "mean": stats.mean,
                "std": stats.std,
                "ci_lower": stats.ci_lower,
                "ci_upper": stats.ci_upper,
                "n_splits": stats.n,
                "min_training_seeds_per_split": min(
                    (_safe_int(row.get("n_training_seeds", 0)) for row in rows),
                    default=0,
                ),
                "min_graph_seeds_per_split": min(
                    (_safe_int(row.get("n_graph_seeds", 0)) for row in rows),
                    default=0,
                ),
                "ci_method": _ci_method(ci, CI_METHOD_SPLIT),
                "source": source,
            }
        )
    return out


def _build_cross_split_drop_rows(
    drop_rows: Sequence[dict[str, Any]],
    *,
    split_allocations: Mapping[str, SplitAllocation],
    ci: bool,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str, str, float, bool], list[dict[str, Any]]] = {}
    for row in drop_rows:
        allocation = _allocation_for_split(row.get("split_id", ""), split_allocations)
        key = (
            allocation.split_regime_id,
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
        split_regime_id, dataset_id, scenario_id, model_id, protocol, metric, max_severity, oracle_labels = key
        rows = groups[key]
        allocation = _allocation_for_split(rows[0].get("split_id", ""), split_allocations)
        if allocation.split_regime_id != split_regime_id:
            raise RuntimeError("Cross-split drop allocation regime mismatch")
        clean_stats = _aggregate_metric_stats([_safe_float(row.get("clean_mean", "")) for row in rows], ci=ci)
        stressed_stats = _aggregate_metric_stats([_safe_float(row.get("max_severity_mean", "")) for row in rows], ci=ci)
        drop_stats = _aggregate_metric_stats([_safe_float(row.get("drop_mean", "")) for row in rows], ci=ci)
        out.append(
            {
                **_allocation_fields(allocation),
                "dataset_id": dataset_id,
                "scenario_id": scenario_id,
                "model_id": model_id,
                "protocol": protocol,
                "metric": metric,
                "max_severity": float(max_severity),
                "oracle_labels": bool(oracle_labels),
                "claim_scope": _claim_scope(oracle_labels=bool(oracle_labels), protocol=protocol),
                "operational_ranking_eligible": not bool(oracle_labels),
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
                "min_paired_training_seeds_per_split": min(
                    (_safe_int(row.get("n_paired_training_seeds", 0)) for row in rows),
                    default=0,
                ),
                "min_graph_seeds_per_split": min(
                    (_safe_int(row.get("n_graph_seeds", 0)) for row in rows),
                    default=0,
                ),
                "ci_method": _ci_method(ci, CI_METHOD_SPLIT),
            }
        )
    return out


def _build_cross_split_robust_rows(
    robust_rows: Sequence[dict[str, Any]],
    *,
    split_allocations: Mapping[str, SplitAllocation],
    ci: bool,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str, str, bool], list[dict[str, Any]]] = {}
    for row in robust_rows:
        allocation = _allocation_for_split(row.get("split_id", ""), split_allocations)
        key = (
            allocation.split_regime_id,
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
        split_regime_id, dataset_id, scenario_id, model_id, protocol, metric, oracle_labels = key
        rows = groups[key]
        allocation = _allocation_for_split(rows[0].get("split_id", ""), split_allocations)
        if allocation.split_regime_id != split_regime_id:
            raise RuntimeError("Cross-split robustness allocation regime mismatch")
        auc_stats = _aggregate_metric_stats([_safe_float(row.get("robustness_auc_mean", "")) for row in rows], ci=ci)
        avg_stats = _aggregate_metric_stats(
            [_safe_float(row.get("robustness_avg_metric_mean", "")) for row in rows],
            ci=ci,
        )
        out.append(
            {
                **_allocation_fields(allocation),
                "dataset_id": dataset_id,
                "scenario_id": scenario_id,
                "model_id": model_id,
                "protocol": protocol,
                "metric": metric,
                "oracle_labels": bool(oracle_labels),
                "claim_scope": _claim_scope(oracle_labels=bool(oracle_labels), protocol=protocol),
                "operational_ranking_eligible": not bool(oracle_labels),
                "robustness_auc_mean": auc_stats.mean,
                "robustness_auc_std": auc_stats.std,
                "robustness_auc_ci_lower": auc_stats.ci_lower,
                "robustness_auc_ci_upper": auc_stats.ci_upper,
                "robustness_avg_metric_mean": avg_stats.mean,
                "robustness_avg_metric_std": avg_stats.std,
                "robustness_avg_metric_ci_lower": avg_stats.ci_lower,
                "robustness_avg_metric_ci_upper": avg_stats.ci_upper,
                "n_splits": avg_stats.n,
                "min_training_seeds_per_split": min(
                    (_safe_int(row.get("n_training_seeds", 0)) for row in rows),
                    default=0,
                ),
                "min_graph_seeds_per_split": min(
                    (_safe_int(row.get("n_graph_seeds", 0)) for row in rows),
                    default=0,
                ),
                "ci_method": _ci_method(ci, CI_METHOD_SPLIT),
            }
        )
    return out


def _build_cross_split_protocol_contrast_rows(
    contrast_rows: Sequence[dict[str, Any]],
    *,
    split_allocations: Mapping[str, SplitAllocation],
    ci: bool,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in contrast_rows:
        allocation = _allocation_for_split(row.get("split_id", ""), split_allocations)
        key = (
            allocation.split_regime_id,
            str(row.get("dataset_id", "")),
            str(row.get("scenario_id", "")),
            float(row.get("severity", 0.0) or 0.0),
            str(row.get("model_id", "")),
            str(row.get("metric", "")),
            bool(row.get("oracle_labels", False)),
            str(row.get("claim_scope", "")),
            str(row.get("left_claim_scope", "")),
            str(row.get("right_claim_scope", "")),
            bool(row.get("operational_ranking_eligible", False)),
            str(row.get("left_protocol", "")),
            str(row.get("right_protocol", "")),
            str(row.get("contrast_definition", "")),
            str(row.get("source", "")),
        )
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for key in sorted(groups):
        (
            split_regime_id,
            dataset_id,
            scenario_id,
            severity,
            model_id,
            metric,
            oracle_labels,
            claim_scope,
            left_claim_scope,
            right_claim_scope,
            operational_ranking_eligible,
            left_protocol,
            right_protocol,
            contrast_definition,
            source,
        ) = key
        rows = groups[key]
        allocation = _allocation_for_split(rows[0].get("split_id", ""), split_allocations)
        if allocation.split_regime_id != split_regime_id:
            raise RuntimeError("Cross-split protocol contrast allocation regime mismatch")
        left_stats = _aggregate_metric_stats(
            [_safe_float(row.get("left_mean", "")) for row in rows],
            ci=ci,
        )
        right_stats = _aggregate_metric_stats(
            [_safe_float(row.get("right_mean", "")) for row in rows],
            ci=ci,
        )
        contrast_stats = _aggregate_metric_stats(
            [_safe_float(row.get("contrast_mean", "")) for row in rows],
            ci=ci,
        )
        out.append(
            {
                **_allocation_fields(allocation),
                "dataset_id": dataset_id,
                "scenario_id": scenario_id,
                "severity": float(severity),
                "model_id": model_id,
                "metric": metric,
                "oracle_labels": bool(oracle_labels),
                "claim_scope": claim_scope,
                "left_claim_scope": left_claim_scope,
                "right_claim_scope": right_claim_scope,
                "operational_ranking_eligible": bool(operational_ranking_eligible),
                "left_protocol": left_protocol,
                "right_protocol": right_protocol,
                "contrast_definition": contrast_definition,
                "left_mean": left_stats.mean,
                "right_mean": right_stats.mean,
                "contrast_mean": contrast_stats.mean,
                "contrast_std": contrast_stats.std,
                "contrast_ci_lower": contrast_stats.ci_lower,
                "contrast_ci_upper": contrast_stats.ci_upper,
                "n_splits": contrast_stats.n,
                "min_pairs_per_split": min(
                    (_safe_int(row.get("n_pairs", 0)) for row in rows),
                    default=0,
                ),
                "min_training_seeds_per_split": min(
                    (_safe_int(row.get("n_training_seeds", 0)) for row in rows),
                    default=0,
                ),
                "min_graph_seeds_per_split": min(
                    (_safe_int(row.get("n_graph_seeds", 0)) for row in rows),
                    default=0,
                ),
                "ci_method": _ci_method(ci, CI_METHOD_SPLIT),
                "source": source,
            }
        )
    return out


def _build_cross_split_worst_case_rows(
    summary_rows: Sequence[dict[str, Any]],
    result_rows: Sequence[ResultRow],
    *,
    split_allocations: Mapping[str, SplitAllocation],
    ci: bool,
) -> list[dict[str, Any]]:
    """Select the lowest regime-level point, then aggregate paired split means."""

    candidates: dict[tuple[str, str, str, str, str], dict[tuple[str, float, bool], list[dict[str, Any]]]] = {}
    for row in summary_rows:
        mean_value = _safe_float(row.get("mean", ""))
        if str(row.get("source", "")) != "scenario" or not math.isfinite(mean_value):
            continue
        allocation = _allocation_for_split(row.get("split_id", ""), split_allocations)
        group_key = (
            allocation.split_regime_id,
            str(row.get("dataset_id", "")),
            str(row.get("model_id", "")),
            str(row.get("protocol", "")),
            str(row.get("metric", "")),
        )
        point_key = (
            str(row.get("scenario_id", "")),
            float(row.get("severity", 0.0) or 0.0),
            bool(row.get("oracle_labels", False)),
        )
        candidates.setdefault(group_key, {}).setdefault(point_key, []).append(row)

    out: list[dict[str, Any]] = []
    for group_key in sorted(candidates):
        split_regime_id, dataset_id, model_id, protocol, metric_key = group_key
        point_groups = candidates[group_key]
        selected_key, selected_rows = min(
            point_groups.items(),
            key=lambda item: (
                _aggregate_metric_stats(
                    [_safe_float(row.get("mean", "")) for row in item[1]],
                    ci=False,
                ).mean,
                item[0][0],
                item[0][1],
            ),
        )
        scenario_id, severity, oracle_labels = selected_key
        allocation = _allocation_for_split(
            selected_rows[0].get("split_id", ""),
            split_allocations,
        )
        if allocation.split_regime_id != split_regime_id:
            raise RuntimeError("Cross-split worst-case allocation regime mismatch")

        worst_split_means: list[float] = []
        clean_split_means: list[float] = []
        drop_split_means: list[float] = []
        retention_split_means: list[float] = []
        for summary_row in selected_rows:
            split_id = str(summary_row.get("split_id", ""))
            split_results = [
                row
                for row in result_rows
                if row.dataset_id == dataset_id
                and row.split_id == split_id
                and row.model_id == model_id
                and row.protocol == protocol
            ]
            clean_rows = _result_rows_for_point(
                split_results,
                model_id=model_id,
                protocol=protocol,
                scenario_id="clean",
                severity=0.0,
            )
            stressed_rows = _result_rows_for_point(
                split_results,
                model_id=model_id,
                protocol=protocol,
                scenario_id=scenario_id,
                severity=severity,
            )
            clean_stats = _aggregate_seeded_metric_stats(clean_rows, metric_key, ci=False)
            drop_stats = _paired_clean_stress_stats(clean_rows, stressed_rows, metric_key, ci=False)
            retention_stats = _paired_retention_stats(clean_rows, stressed_rows, metric_key, ci=False)
            worst_split_means.append(_safe_float(summary_row.get("mean", "")))
            clean_split_means.append(clean_stats.stats.mean)
            drop_split_means.append(drop_stats.stats.mean)
            retention_split_means.append(retention_stats.stats.mean)

        worst_stats = _aggregate_metric_stats(worst_split_means, ci=ci)
        clean_stats = _aggregate_metric_stats(clean_split_means, ci=ci)
        drop_stats = _aggregate_metric_stats(drop_split_means, ci=ci)
        retention_stats = _aggregate_metric_stats(retention_split_means, ci=ci)
        claim_scope = _claim_scope(oracle_labels=oracle_labels, protocol=protocol)
        out.append(
            {
                **_allocation_fields(allocation),
                "dataset_id": dataset_id,
                "model_id": model_id,
                "protocol": protocol,
                "metric": metric_key,
                "worst_scenario_id": scenario_id,
                "worst_severity": float(severity),
                "oracle_labels": bool(oracle_labels),
                "claim_scope": claim_scope,
                "operational_ranking_eligible": not bool(oracle_labels),
                "worst_mean": worst_stats.mean,
                "worst_std": worst_stats.std,
                "worst_ci_lower": worst_stats.ci_lower,
                "worst_ci_upper": worst_stats.ci_upper,
                "clean_mean": clean_stats.mean,
                "clean_std": clean_stats.std,
                "clean_ci_lower": clean_stats.ci_lower,
                "clean_ci_upper": clean_stats.ci_upper,
                "drop_from_clean_mean": drop_stats.mean,
                "drop_from_clean_std": drop_stats.std,
                "drop_from_clean_ci_lower": drop_stats.ci_lower,
                "drop_from_clean_ci_upper": drop_stats.ci_upper,
                "retention_fraction_mean": retention_stats.mean,
                "retention_fraction_std": retention_stats.std,
                "retention_fraction_ci_lower": retention_stats.ci_lower,
                "retention_fraction_ci_upper": retention_stats.ci_upper,
                "n_splits": worst_stats.n,
                "min_runs_per_split": min(
                    (_safe_int(row.get("n_runs", 0)) for row in selected_rows),
                    default=0,
                ),
                "min_training_seeds_per_split": min(
                    (_safe_int(row.get("n_training_seeds", 0)) for row in selected_rows),
                    default=0,
                ),
                "min_graph_seeds_per_split": min(
                    (_safe_int(row.get("n_graph_seeds", 0)) for row in selected_rows),
                    default=0,
                ),
                "ci_method": _ci_method(ci, CI_METHOD_SPLIT),
                "drop_ci_method": _ci_method(ci, CI_METHOD_SPLIT),
                "retention_ci_method": _ci_method(ci, CI_METHOD_SPLIT),
                "selection_method": "minimum_configured_nonzero_point_cross_split_mean",
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

        clean_by_model: dict[str, dict[str, SeededMetricStats]] = {}
        clean_rows_by_model: dict[str, list[ResultRow]] = {}
        for model_id in model_ids:
            sel = [
                r
                for r in ok_g
                if r.model_id == model_id and r.scenario_id == "clean" and float(r.severity) == 0.0
            ]
            clean_by_model[model_id] = {}
            clean_rows_by_model[model_id] = list(sel)
            for metric_key, _metric_label in metrics:
                clean_by_model[model_id][metric_key] = _aggregate_seeded_metric_stats(
                    sel,
                    metric_key,
                    ci=ci,
                )

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
                    clean_seeded = clean_by_model.get(model_id, {}).get(
                        metric_key,
                        _empty_seeded_stats(ci=ci, method=CI_METHOD_CROSSED_SEEDS),
                    )
                    clean_stats = clean_seeded.stats
                    clean_row = {
                        "dataset_id": dataset_id,
                        "split_id": split_id,
                        "scenario_id": scenario_id,
                        "severity": 0.0,
                        "model_id": model_id,
                        "protocol": protocol,
                        "metric": metric_key,
                        "oracle_labels": oracle_labels,
                        "claim_scope": _claim_scope(oracle_labels=oracle_labels, protocol=protocol),
                        "operational_ranking_eligible": not oracle_labels,
                        "mean": clean_stats.mean,
                        "std": clean_stats.std,
                        "ci_lower": clean_stats.ci_lower,
                        "ci_upper": clean_stats.ci_upper,
                        "n_runs": clean_stats.n,
                        "n_training_seeds": clean_seeded.n_training_seeds,
                        "n_graph_seeds": clean_seeded.n_graph_seeds,
                        "n_seed_cells": clean_seeded.n_seed_cells,
                        "ci_method": clean_seeded.ci_method,
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
                        seeded = _aggregate_seeded_metric_stats(sel, metric_key, ci=ci)
                        stats = seeded.stats
                        row = {
                            "dataset_id": dataset_id,
                            "split_id": split_id,
                            "scenario_id": scenario_id,
                            "severity": float(severity),
                            "model_id": model_id,
                            "protocol": protocol,
                            "metric": metric_key,
                            "oracle_labels": oracle_labels,
                            "claim_scope": _claim_scope(oracle_labels=oracle_labels, protocol=protocol),
                            "operational_ranking_eligible": not oracle_labels,
                            "mean": stats.mean,
                            "std": stats.std,
                            "ci_lower": stats.ci_lower,
                            "ci_upper": stats.ci_upper,
                            "n_runs": stats.n,
                            "n_training_seeds": seeded.n_training_seeds,
                            "n_graph_seeds": seeded.n_graph_seeds,
                            "n_seed_cells": seeded.n_seed_cells,
                            "ci_method": seeded.ci_method,
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
                        clean_sel = clean_rows_by_model.get(model_id, [])
                        stressed_sel = [
                            r
                            for r in ok_g
                            if r.model_id == model_id
                            and r.scenario_id == scenario_id
                            and float(r.severity) == float(max_severity)
                        ]
                        clean_seeded = _aggregate_seeded_metric_stats(clean_sel, metric_key, ci=ci)
                        stressed_seeded = _aggregate_seeded_metric_stats(stressed_sel, metric_key, ci=ci)
                        paired_drop = _paired_clean_stress_stats(
                            clean_sel,
                            stressed_sel,
                            metric_key,
                            ci=ci,
                        )
                        clean_stats = clean_seeded.stats
                        stressed_stats = stressed_seeded.stats
                        drop_stats = paired_drop.stats
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
                                "claim_scope": _claim_scope(oracle_labels=oracle_labels, protocol=protocol),
                                "operational_ranking_eligible": not oracle_labels,
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
                                "n_paired_seed_cells": paired_drop.n_seed_cells,
                                "n_paired_training_seeds": paired_drop.n_training_seeds,
                                "n_graph_seeds": paired_drop.n_graph_seeds,
                                "ci_method": paired_drop.ci_method,
                            }
                        )

                if len(sevs_plot) < 2:
                    continue

                x_values = [float(sev) for sev in sevs_plot]
                for model_id in model_ids:
                    rows_by_severity: list[list[ResultRow]] = []
                    for severity in sevs_plot:
                        if float(severity) == 0.0:
                            rows_by_severity.append(clean_rows_by_model.get(model_id, []))
                        else:
                            rows_by_severity.append(
                                [
                                    r
                                    for r in ok_g
                                    if r.model_id == model_id
                                    and r.scenario_id == scenario_id
                                    and float(r.severity) == float(severity)
                                ]
                            )
                    auc_seeded, avg_seeded, min_runs = _paired_curve_robustness_stats(
                        x_values,
                        rows_by_severity,
                        metric_key,
                        ci=ci,
                    )
                    auc_stats = auc_seeded.stats
                    avg_stats = avg_seeded.stats
                    robust_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "split_id": split_id,
                            "scenario_id": scenario_id,
                            "model_id": model_id,
                            "protocol": protocol,
                            "metric": metric_key,
                            "oracle_labels": oracle_labels,
                            "claim_scope": _claim_scope(oracle_labels=oracle_labels, protocol=protocol),
                            "operational_ranking_eligible": not oracle_labels,
                            "robustness_auc_mean": auc_stats.mean,
                            "robustness_auc_std": auc_stats.std,
                            "robustness_auc_ci_lower": auc_stats.ci_lower,
                            "robustness_auc_ci_upper": auc_stats.ci_upper,
                            "robustness_avg_metric_mean": avg_stats.mean,
                            "robustness_avg_metric_std": avg_stats.std,
                            "robustness_avg_metric_ci_lower": avg_stats.ci_lower,
                            "robustness_avg_metric_ci_upper": avg_stats.ci_upper,
                            "min_n_runs_per_severity": min_runs,
                            "n_paired_seed_cells": avg_seeded.n_seed_cells,
                            "n_training_seeds": avg_seeded.n_training_seeds,
                            "n_graph_seeds": avg_seeded.n_graph_seeds,
                            "ci_method": avg_seeded.ci_method,
                        }
                    )

    protocol_contrast_rows = _build_protocol_contrast_rows(
        ok,
        severity_grid=severity_grid,
        scenario_meta=scenario_meta,
        model_ids=model_ids,
        metric_keys=[metric_key for metric_key, _metric_label in metrics],
        ci=ci,
    )
    worst_case_rows = _build_worst_case_rows(summary_rows, ok, ci=ci)

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
            "claim_scope",
            "operational_ranking_eligible",
            "mean",
            "std",
            "ci_lower",
            "ci_upper",
            "n_runs",
            "n_training_seeds",
            "n_graph_seeds",
            "n_seed_cells",
            "ci_method",
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
            "claim_scope",
            "operational_ranking_eligible",
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
            "n_paired_seed_cells",
            "n_paired_training_seeds",
            "n_graph_seeds",
            "ci_method",
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
            "claim_scope",
            "operational_ranking_eligible",
            "robustness_auc_mean",
            "robustness_auc_std",
            "robustness_auc_ci_lower",
            "robustness_auc_ci_upper",
            "robustness_avg_metric_mean",
            "robustness_avg_metric_std",
            "robustness_avg_metric_ci_lower",
            "robustness_avg_metric_ci_upper",
            "min_n_runs_per_severity",
            "n_paired_seed_cells",
            "n_training_seeds",
            "n_graph_seeds",
            "ci_method",
        ],
        robust_rows,
    )
    _write_csv(
        plots_dir / "protocol_contrasts.csv",
        [
            "dataset_id",
            "split_id",
            "scenario_id",
            "severity",
            "model_id",
            "metric",
            "oracle_labels",
            "claim_scope",
            "left_claim_scope",
            "right_claim_scope",
            "operational_ranking_eligible",
            "left_protocol",
            "right_protocol",
            "contrast_definition",
            "left_mean",
            "right_mean",
            "contrast_mean",
            "contrast_std",
            "contrast_ci_lower",
            "contrast_ci_upper",
            "n_pairs",
            "n_training_seeds",
            "n_graph_seeds",
            "ci_method",
            "source",
        ],
        protocol_contrast_rows,
    )
    _write_csv(
        plots_dir / "worst_case_performance.csv",
        [
            "dataset_id",
            "split_id",
            "model_id",
            "protocol",
            "metric",
            "worst_scenario_id",
            "worst_severity",
            "oracle_labels",
            "claim_scope",
            "operational_ranking_eligible",
            "worst_mean",
            "worst_std",
            "worst_ci_lower",
            "worst_ci_upper",
            "clean_mean",
            "drop_from_clean_mean",
            "drop_from_clean_std",
            "drop_from_clean_ci_lower",
            "drop_from_clean_ci_upper",
            "retention_fraction_mean",
            "retention_fraction_std",
            "retention_fraction_ci_lower",
            "retention_fraction_ci_upper",
            "n_runs",
            "n_training_seeds",
            "n_graph_seeds",
            "n_paired_seed_cells",
            "ci_method",
            "drop_ci_method",
            "retention_ci_method",
            "selection_method",
        ],
        worst_case_rows,
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
            "claim_scope",
            "operational_ranking_eligible",
            "graph_view_mode",
            "mean",
            "std",
            "ci_lower",
            "ci_upper",
            "n_graphs",
            "n_graph_seeds",
            "ci_method",
            "source",
        ],
        audit_summary_rows,
    )

    split_allocations = _split_allocations_from_config(cfg)
    cross_split_summary_rows = _build_cross_split_summary_rows(
        summary_rows,
        split_allocations=split_allocations,
        ci=ci,
    )
    cross_split_drop_rows = _build_cross_split_drop_rows(
        drop_rows,
        split_allocations=split_allocations,
        ci=ci,
    )
    cross_split_robust_rows = _build_cross_split_robust_rows(
        robust_rows,
        split_allocations=split_allocations,
        ci=ci,
    )
    cross_split_audit_rows = _build_cross_split_audit_rows(
        audit_summary_rows,
        split_allocations=split_allocations,
        ci=ci,
    )
    cross_split_protocol_contrast_rows = _build_cross_split_protocol_contrast_rows(
        protocol_contrast_rows,
        split_allocations=split_allocations,
        ci=ci,
    )
    cross_split_worst_case_rows = _build_cross_split_worst_case_rows(
        summary_rows,
        ok,
        split_allocations=split_allocations,
        ci=ci,
    )

    _write_csv(
        plots_dir / "summary_curves_cross_split.csv",
        [
            "split_regime_id",
            "train_size",
            "val_size",
            "test_size",
            "dataset_id",
            "scenario_id",
            "severity",
            "model_id",
            "protocol",
            "metric",
            "oracle_labels",
            "claim_scope",
            "operational_ranking_eligible",
            "mean",
            "std",
            "ci_lower",
            "ci_upper",
            "n_splits",
            "min_training_seeds_per_split",
            "min_graph_seeds_per_split",
            "ci_method",
            "source",
        ],
        cross_split_summary_rows,
    )
    _write_csv(
        plots_dir / "performance_drop_max_stress_cross_split.csv",
        [
            "split_regime_id",
            "train_size",
            "val_size",
            "test_size",
            "dataset_id",
            "scenario_id",
            "model_id",
            "protocol",
            "metric",
            "max_severity",
            "oracle_labels",
            "claim_scope",
            "operational_ranking_eligible",
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
            "min_paired_training_seeds_per_split",
            "min_graph_seeds_per_split",
            "ci_method",
        ],
        cross_split_drop_rows,
    )
    _write_csv(
        plots_dir / "robustness_scores_cross_split.csv",
        [
            "split_regime_id",
            "train_size",
            "val_size",
            "test_size",
            "dataset_id",
            "scenario_id",
            "model_id",
            "protocol",
            "metric",
            "oracle_labels",
            "claim_scope",
            "operational_ranking_eligible",
            "robustness_auc_mean",
            "robustness_auc_std",
            "robustness_auc_ci_lower",
            "robustness_auc_ci_upper",
            "robustness_avg_metric_mean",
            "robustness_avg_metric_std",
            "robustness_avg_metric_ci_lower",
            "robustness_avg_metric_ci_upper",
            "n_splits",
            "min_training_seeds_per_split",
            "min_graph_seeds_per_split",
            "ci_method",
        ],
        cross_split_robust_rows,
    )
    _write_csv(
        plots_dir / "audit_curves_cross_split.csv",
        [
            "split_regime_id",
            "train_size",
            "val_size",
            "test_size",
            "dataset_id",
            "scenario_id",
            "severity",
            "protocol",
            "audit_metric",
            "audit_metric_label",
            "audit_metric_unit",
            "oracle_labels",
            "claim_scope",
            "operational_ranking_eligible",
            "graph_view_mode",
            "mean",
            "std",
            "ci_lower",
            "ci_upper",
            "n_splits",
            "min_graph_seeds_per_split",
            "ci_method",
            "source",
        ],
        cross_split_audit_rows,
    )
    _write_csv(
        plots_dir / "protocol_contrasts_cross_split.csv",
        [
            "split_regime_id",
            "train_size",
            "val_size",
            "test_size",
            "dataset_id",
            "scenario_id",
            "severity",
            "model_id",
            "metric",
            "oracle_labels",
            "claim_scope",
            "left_claim_scope",
            "right_claim_scope",
            "operational_ranking_eligible",
            "left_protocol",
            "right_protocol",
            "contrast_definition",
            "left_mean",
            "right_mean",
            "contrast_mean",
            "contrast_std",
            "contrast_ci_lower",
            "contrast_ci_upper",
            "n_splits",
            "min_pairs_per_split",
            "min_training_seeds_per_split",
            "min_graph_seeds_per_split",
            "ci_method",
            "source",
        ],
        cross_split_protocol_contrast_rows,
    )
    _write_csv(
        plots_dir / "worst_case_performance_cross_split.csv",
        [
            "split_regime_id",
            "train_size",
            "val_size",
            "test_size",
            "dataset_id",
            "model_id",
            "protocol",
            "metric",
            "worst_scenario_id",
            "worst_severity",
            "oracle_labels",
            "claim_scope",
            "operational_ranking_eligible",
            "worst_mean",
            "worst_std",
            "worst_ci_lower",
            "worst_ci_upper",
            "clean_mean",
            "clean_std",
            "clean_ci_lower",
            "clean_ci_upper",
            "drop_from_clean_mean",
            "drop_from_clean_std",
            "drop_from_clean_ci_lower",
            "drop_from_clean_ci_upper",
            "retention_fraction_mean",
            "retention_fraction_std",
            "retention_fraction_ci_lower",
            "retention_fraction_ci_upper",
            "n_splits",
            "min_runs_per_split",
            "min_training_seeds_per_split",
            "min_graph_seeds_per_split",
            "ci_method",
            "drop_ci_method",
            "retention_ci_method",
            "selection_method",
        ],
        cross_split_worst_case_rows,
    )

    split_counts_by_dataset_protocol: dict[tuple[str, str, str], set[str]] = {}
    for row in summary_rows:
        allocation = _allocation_for_split(row.get("split_id", ""), split_allocations)
        key = (
            str(row.get("dataset_id", "")),
            str(row.get("protocol", "")),
            allocation.split_regime_id,
        )
        split_counts_by_dataset_protocol.setdefault(key, set()).add(str(row.get("split_id", "")))

    regimes_by_dataset_protocol: dict[tuple[str, str], set[str]] = {}
    for dataset_id, protocol, split_regime_id in split_counts_by_dataset_protocol:
        regimes_by_dataset_protocol.setdefault((dataset_id, protocol), set()).add(split_regime_id)

    for dataset_id, protocol, split_regime_id in sorted(split_counts_by_dataset_protocol):
        if len(split_counts_by_dataset_protocol[(dataset_id, protocol, split_regime_id)]) <= 1:
            continue
        dataset_rows = [
            row
            for row in cross_split_summary_rows
            if str(row.get("dataset_id", "")) == dataset_id
            and str(row.get("protocol", "")) == protocol
            and str(row.get("split_regime_id", "")) == split_regime_id
        ]
        out_group_dir = plots_dir / protocol / dataset_id / "across_splits"
        if len(regimes_by_dataset_protocol[(dataset_id, protocol)]) > 1:
            out_group_dir = out_group_dir / split_regime_id
        split_label = (
            "across_splits"
            if split_regime_id == UNSPECIFIED_SPLIT_REGIME_ID
            else f"across_splits [{split_regime_id}]"
        )
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
                    split_label=split_label,
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

    split_counts_by_dataset_protocol_audit: dict[tuple[str, str, str], set[str]] = {}
    for row in audit_summary_rows:
        allocation = _allocation_for_split(row.get("split_id", ""), split_allocations)
        key = (
            str(row.get("dataset_id", "")),
            str(row.get("protocol", "")),
            allocation.split_regime_id,
        )
        split_counts_by_dataset_protocol_audit.setdefault(key, set()).add(str(row.get("split_id", "")))

    regimes_by_dataset_protocol_audit: dict[tuple[str, str], set[str]] = {}
    for dataset_id, protocol, split_regime_id in split_counts_by_dataset_protocol_audit:
        regimes_by_dataset_protocol_audit.setdefault((dataset_id, protocol), set()).add(split_regime_id)

    for dataset_id, protocol, split_regime_id in sorted(split_counts_by_dataset_protocol_audit):
        if len(split_counts_by_dataset_protocol_audit[(dataset_id, protocol, split_regime_id)]) <= 1:
            continue
        dataset_rows = [
            row
            for row in cross_split_audit_rows
            if str(row.get("dataset_id", "")) == dataset_id
            and str(row.get("protocol", "")) == protocol
            and str(row.get("split_regime_id", "")) == split_regime_id
        ]
        out_group_dir = plots_dir / protocol / dataset_id / "across_splits"
        if len(regimes_by_dataset_protocol_audit[(dataset_id, protocol)]) > 1:
            out_group_dir = out_group_dir / split_regime_id
        split_label = (
            "across_splits"
            if split_regime_id == UNSPECIFIED_SPLIT_REGIME_ID
            else f"across_splits [{split_regime_id}]"
        )
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
                    split_label=split_label,
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
