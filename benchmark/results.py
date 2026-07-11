from __future__ import annotations

import csv
import math
import threading
from pathlib import Path
from typing import Any, Mapping


PROTOCOL_TRAIN_ON_VARIANT = "train_on_variant"
PROTOCOL_TRAIN_CLEAN_EVAL_ALL = "train_clean_eval_all"
RESULTS_COLUMN_DEFAULTS = {
    "protocol": PROTOCOL_TRAIN_ON_VARIANT,
    "train_graph_ref": "",
}

RunKey = tuple[str, str, str, float, int, int, str, str]

_CSV_LOCK = threading.RLock()


RESULTS_COLUMNS = [
    # identifiers
    "experiment_name",
    "dataset_id",
    "split_id",
    "graph_seed",
    "training_seed",
    "scenario_id",
    "severity",
    "model_id",
    "protocol",
    # metrics (may be empty until models are integrated)
    "roc_auc",
    "average_precision",
    "f1_macro",
    "threshold",
    "duration_sec",
    # graph stats (always available)
    "n_nodes",
    "n_edges",
    "mean_in_degree",
    "median_in_degree",
    "mean_out_degree",
    "median_out_degree",
    "heterophily_ratio",
    "pos_rate",
    # paths and misc
    "base_graph_path",
    "graph_path",
    "train_graph_ref",
    "status",
    "error",
]


VARIANTS_COLUMNS = [
    "experiment_name",
    "dataset_id",
    "split_id",
    "graph_seed",
    "scenario_id",
    "severity",
    "oracle_labels",
    "scenario_applied",
    "base_graph_path",
    "graph_path",
    "n_nodes",
    "n_edges",
    "mean_in_degree",
    "median_in_degree",
    "mean_out_degree",
    "median_out_degree",
    "heterophily_ratio",
    "pos_rate",
]


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return int(default)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def normalize_protocol(protocol: Any) -> str:
    text = str(protocol or "").strip()
    return text or PROTOCOL_TRAIN_ON_VARIANT


def make_run_key(
    *,
    dataset_id: Any,
    split_id: Any,
    scenario_id: Any,
    severity: Any,
    graph_seed: Any,
    training_seed: Any,
    model_id: Any,
    protocol: Any = PROTOCOL_TRAIN_ON_VARIANT,
) -> RunKey:
    return (
        str(dataset_id),
        str(split_id),
        str(scenario_id),
        _safe_float(severity, 0.0),
        _safe_int(graph_seed, 0),
        _safe_int(training_seed, 0),
        str(model_id),
        normalize_protocol(protocol),
    )


def row_run_key(row: Mapping[str, Any]) -> RunKey:
    return make_run_key(
        dataset_id=row.get("dataset_id", ""),
        split_id=row.get("split_id", ""),
        scenario_id=row.get("scenario_id", ""),
        severity=row.get("severity", 0.0),
        graph_seed=row.get("graph_seed", 0),
        training_seed=row.get("training_seed", 0),
        model_id=row.get("model_id", ""),
        protocol=row.get("protocol", ""),
    )


def truncate_error_message(error: Any, *, max_len: int = 500) -> str:
    text = str(error).replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= int(max_len):
        return text
    if max_len <= 3:
        return text[:max_len]
    return text[: max_len - 3].rstrip() + "..."


def _row_with_defaults(columns: list[str], row: Mapping[str, Any], row_defaults: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in columns:
        value = row.get(col, "")
        if (col not in row or value == "") and col in row_defaults:
            value = row_defaults[col]
        out[col] = value
    return out


def _current_csv_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader, [])


def ensure_csv_header(
    path: Path,
    columns: list[str],
    *,
    overwrite: bool = False,
    row_defaults: Mapping[str, Any] | None = None,
) -> None:
    row_defaults = row_defaults or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    with _CSV_LOCK:
        if overwrite or not path.exists():
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(columns)
            return

        current_header = _current_csv_header(path)
        if current_header == columns:
            return

        existing_rows: list[dict[str, Any]] = []
        if current_header:
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for existing_row in existing_rows:
                writer.writerow(_row_with_defaults(columns, existing_row, row_defaults))


def append_csv_row(
    path: Path,
    columns: list[str],
    row: Mapping[str, Any],
    *,
    row_defaults: Mapping[str, Any] | None = None,
) -> None:
    row_defaults = row_defaults or {}
    with _CSV_LOCK:
        ensure_csv_header(path, columns, overwrite=False, row_defaults=row_defaults)
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writerow(_row_with_defaults(columns, row, row_defaults))


def ensure_results_csv(path: Path, *, overwrite: bool = False) -> None:
    ensure_csv_header(path, RESULTS_COLUMNS, overwrite=overwrite, row_defaults=RESULTS_COLUMN_DEFAULTS)


def append_result_row(path: Path, row: Mapping[str, Any]) -> None:
    append_csv_row(path, RESULTS_COLUMNS, row, row_defaults=RESULTS_COLUMN_DEFAULTS)


def build_result_row_common(
    variant: Any,
    *,
    model_id: str,
    training_seed: int,
    protocol: str,
    train_graph_ref: Any | None = None,
) -> dict[str, Any]:
    graph_ref = getattr(variant, "graph_path", "") if train_graph_ref is None else train_graph_ref
    return {
        "experiment_name": getattr(variant, "experiment_name", ""),
        "dataset_id": getattr(variant, "dataset_id", ""),
        "split_id": getattr(variant, "split_id", ""),
        "graph_seed": int(getattr(variant, "graph_seed", 0)),
        "training_seed": int(training_seed),
        "scenario_id": getattr(variant, "scenario_id", ""),
        "severity": float(getattr(variant, "severity", 0.0)),
        "model_id": str(model_id),
        "protocol": normalize_protocol(protocol),
        "n_nodes": int(getattr(variant, "n_nodes", 0)),
        "n_edges": int(getattr(variant, "n_edges", 0)),
        "mean_in_degree": float(getattr(variant, "mean_in_degree", 0.0)),
        "median_in_degree": float(getattr(variant, "median_in_degree", 0.0)),
        "mean_out_degree": float(getattr(variant, "mean_out_degree", 0.0)),
        "median_out_degree": float(getattr(variant, "median_out_degree", 0.0)),
        "heterophily_ratio": getattr(variant, "heterophily_ratio", None),
        "pos_rate": getattr(variant, "pos_rate", None),
        "base_graph_path": getattr(variant, "base_graph_path", ""),
        "graph_path": getattr(variant, "graph_path", ""),
        "train_graph_ref": graph_ref,
    }


def write_result_row(
    results_csv: Path,
    variant: Any,
    *,
    model_id: str,
    training_seed: int,
    protocol: str,
    metrics: Mapping[str, Any] | None,
    error: Any = "",
    duration_sec: float | None = None,
    train_graph_ref: Any | None = None,
) -> None:
    row = build_result_row_common(
        variant,
        model_id=model_id,
        training_seed=training_seed,
        protocol=protocol,
        train_graph_ref=train_graph_ref,
    )
    if metrics is None:
        row.update(
            {
                "duration_sec": float(0.0 if duration_sec is None else duration_sec),
                "status": "error",
                "error": truncate_error_message(error),
            }
        )
    else:
        effective_duration = metrics.get("duration_sec", "")
        if duration_sec is not None:
            effective_duration = float(duration_sec)
        validated_metrics: dict[str, float] = {}
        for metric_name in ("roc_auc", "average_precision", "f1_macro"):
            try:
                metric_value = float(metrics.get(metric_name, ""))
            except Exception as e:
                raise ValueError(f"Result metric '{metric_name}' is not numeric.") from e
            if not math.isfinite(metric_value) or not 0.0 <= metric_value <= 1.0:
                raise ValueError(
                    f"Result metric '{metric_name}' must be finite and within [0, 1]; "
                    f"found {metric_value!r}."
                )
            validated_metrics[metric_name] = metric_value
        try:
            threshold = float(metrics.get("threshold", ""))
        except Exception as e:
            raise ValueError("Result metric 'threshold' is not numeric.") from e
        if not math.isfinite(threshold):
            raise ValueError(
                "Result metric 'threshold' must be finite; "
                f"found {threshold!r}."
            )
        validated_metrics["threshold"] = threshold
        try:
            effective_duration = float(effective_duration)
        except Exception as e:
            raise ValueError("Result duration_sec is not numeric.") from e
        if not math.isfinite(effective_duration) or effective_duration < 0.0:
            raise ValueError(
                f"Result duration_sec must be finite and non-negative; found {effective_duration!r}."
            )
        row.update(
            {
                **validated_metrics,
                "duration_sec": effective_duration,
                "status": "ok",
                "error": "",
            }
        )
    append_result_row(results_csv, row)


def load_completed_keys(path: Path, *, retry_errors: bool = False) -> set[RunKey]:
    ensure_results_csv(path, overwrite=False)

    statuses = {"ok"}
    if not retry_errors:
        statuses.add("error")

    completed: set[RunKey] = set()
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = str(row.get("status", "")).strip().lower()
            if status not in statuses:
                continue
            completed.add(row_run_key(row))
    return completed
