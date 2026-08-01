from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

from .results import normalize_protocol


@dataclass(frozen=True)
class SummaryRow:
    experiment_name: str
    dataset_id: str
    split_id: str
    scenario_id: str
    severity: float
    graph_seed: int
    model_id: str
    protocol: str
    n_runs: int
    roc_auc_mean: float
    roc_auc_std: float
    average_precision_mean: float
    average_precision_std: float
    f1_macro_mean: float
    f1_macro_std: float


SUMMARY_COLUMNS = [
    "experiment_name",
    "dataset_id",
    "split_id",
    "scenario_id",
    "severity",
    "graph_seed",
    "model_id",
    "protocol",
    "n_runs",
    "roc_auc_mean",
    "roc_auc_std",
    "average_precision_mean",
    "average_precision_std",
    "f1_macro_mean",
    "f1_macro_std",
]


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def _safe_int(x, default: int = 0):
    try:
        return int(float(x))
    except Exception:
        return int(default)


def summarize_results_by_training_seed(
    results_csv_path: Path,
    *,
    out_csv_path: Path,
    model_ids: set[str] | None = None,
    protocols: set[str] | None = None,
) -> None:
    """Summarize `results.csv` into mean/std across training seeds.

    Groups by (experiment_name, dataset_id, split_id, scenario_id, severity, graph_seed, model_id, protocol).
    Filters to status=="ok". Optional model and protocol filters are normalized before use.
    """
    if not results_csv_path.exists():
        raise FileNotFoundError(f"results.csv not found: {results_csv_path}")

    groups: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    normalized_protocols = None
    if protocols is not None:
        normalized_protocols = {normalize_protocol(value) for value in protocols}

    with results_csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status", "") != "ok":
                continue
            mid = str(row.get("model_id", ""))
            if model_ids is not None and mid not in model_ids:
                continue
            protocol = normalize_protocol(row.get("protocol", ""))
            if normalized_protocols is not None and protocol not in normalized_protocols:
                continue

            key = (
                str(row.get("experiment_name", "")),
                str(row.get("dataset_id", "")),
                str(row.get("split_id", "")),
                str(row.get("scenario_id", "")),
                float(row.get("severity", 0.0) or 0.0),
                _safe_int(row.get("graph_seed", 0)),
                mid,
                protocol,
            )

            groups[key]["roc_auc"].append(_safe_float(row.get("roc_auc", "")))
            groups[key]["average_precision"].append(_safe_float(row.get("average_precision", "")))
            groups[key]["f1_macro"].append(_safe_float(row.get("f1_macro", "")))

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()

        for key in sorted(groups.keys()):
            exp, ds, split, scenario, sev, gseed, mid, protocol = key
            m = groups[key]

            def _m_std(vals: list[float]) -> tuple[float, float, int]:
                vv = [v for v in vals if math.isfinite(v)]
                if not vv:
                    return float("nan"), float("nan"), 0
                if len(vv) == 1:
                    return float(vv[0]), 0.0, 1
                return float(mean(vv)), float(stdev(vv)), len(vv)

            roc_m, roc_s, n1 = _m_std(m["roc_auc"])
            ap_m, ap_s, n2 = _m_std(m["average_precision"])
            f1_m, f1_s, n3 = _m_std(m["f1_macro"])
            n_runs = min(n1, n2, n3) if min(n1, n2, n3) > 0 else max(n1, n2, n3)

            w.writerow(
                {
                    "experiment_name": exp,
                    "dataset_id": ds,
                    "split_id": split,
                    "scenario_id": scenario,
                    "severity": f"{float(sev):.6g}",
                    "graph_seed": int(gseed),
                    "model_id": mid,
                    "protocol": protocol,
                    "n_runs": int(n_runs),
                    "roc_auc_mean": roc_m,
                    "roc_auc_std": roc_s,
                    "average_precision_mean": ap_m,
                    "average_precision_std": ap_s,
                    "f1_macro_mean": f1_m,
                    "f1_macro_std": f1_s,
                }
            )
