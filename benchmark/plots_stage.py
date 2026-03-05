from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .results import PROTOCOL_TRAIN_ON_VARIANT, normalize_protocol


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


def _aggregate_mean_std(values: list[float]) -> tuple[float, float, int]:
    vv = [v for v in values if math.isfinite(v)]
    if not vv:
        return float("nan"), float("nan"), 0
    if len(vv) == 1:
        return float(vv[0]), 0.0, 1
    m = float(sum(vv) / len(vv))
    var = float(sum((v - m) ** 2 for v in vv) / (len(vv) - 1))
    return m, float(math.sqrt(max(0.0, var))), int(len(vv))


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
        # Keep only unique severities, sorted.
        sevs = sorted(set(sevs))
        grid[sid] = sevs
    return grid


def _model_ids_from_config(cfg: dict[str, Any]) -> list[str]:
    mids = []
    for m in cfg.get("models", []):
        mid = str(m.get("model_id", ""))
        if mid:
            mids.append(mid)
    # Preserve order but drop duplicates.
    seen = set()
    out = []
    for mid in mids:
        if mid in seen:
            continue
        seen.add(mid)
        out.append(mid)
    return out


def _completeness_report(
    cfg: dict[str, Any],
    *,
    out_dir: Path,
    include_noop: bool,
    only_clean: bool,
    max_variants: int | None,
    max_training_seeds: int | None,
) -> tuple[int, int]:
    """Write missing/error run keys based on graph_variants.csv and results.csv."""
    variants_csv = out_dir / "graph_variants.csv"
    results_csv = out_dir / "results.csv"

    if not variants_csv.exists() or not results_csv.exists():
        return 0, 0

    # Expected keys are derived from graph_variants.csv, matching stage filters.
    expected_variant_rows = []
    with variants_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            scenario_id = str(r.get("scenario_id", ""))
            severity = float(r.get("severity", 0.0) or 0.0)
            scenario_applied = str(r.get("scenario_applied", "")).strip().lower() in {"1", "true", "t", "yes", "y"}

            if only_clean and scenario_id != "clean":
                continue
            if (not include_noop) and (not scenario_applied) and scenario_id != "clean":
                continue

            expected_variant_rows.append(
                (
                    str(r.get("dataset_id", "")),
                    str(r.get("split_id", "")),
                    scenario_id,
                    float(severity),
                    _safe_int(r.get("graph_seed", 0)),
                    PROTOCOL_TRAIN_ON_VARIANT,
                )
            )

    if max_variants is not None:
        expected_variant_rows = expected_variant_rows[: int(max_variants)]

    training_seeds = [int(s) for s in cfg.get("seeds", {}).get("training_seeds", [])]
    if max_training_seeds is not None:
        training_seeds = training_seeds[: int(max_training_seeds)]

    model_ids = [m for m in _model_ids_from_config(cfg) if m in {"mlp", "sage", "pmp", "secgfd"}]

    expected_keys: set[tuple] = set()
    for ds, split, scenario_id, severity, graph_seed, protocol in expected_variant_rows:
        for tr in training_seeds:
            for mid in model_ids:
                expected_keys.add((ds, split, scenario_id, float(severity), int(graph_seed), int(tr), mid, protocol))

    found_ok: set[tuple] = set()
    found_err: set[tuple] = set()
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
            status = str(r.get("status", ""))
            if status == "ok":
                found_ok.add(key)
            elif status == "error":
                found_err.add(key)

    missing = sorted(expected_keys - found_ok - found_err)
    errored = sorted(expected_keys & found_err)

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
    for ds, split, scenario_id, severity, graph_seed, tr, mid, protocol in errored:
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

    return len(expected_keys), len(missing) + len(errored)


def run_plots_stage(
    cfg: dict[str, Any],
    *,
    out_dir: Path,
    include_noop: bool,
    only_clean: bool,
    max_variants: int | None,
    max_training_seeds: int | None,
) -> None:
    """Generate plots and plot-ready CSV summaries from results.csv."""
    out_dir = out_dir.resolve()
    results_csv = out_dir / "results.csv"
    rows = _read_results_csv(results_csv)
    ok = [r for r in rows if r.status == "ok"]

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Completeness report helps catch missing combos before the report is written.
    _completeness_report(
        cfg,
        out_dir=out_dir,
        include_noop=bool(include_noop),
        only_clean=bool(only_clean),
        max_variants=max_variants,
        max_training_seeds=max_training_seeds,
    )

    # Import matplotlib lazily so other stages don't need it.
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for --stage plots. Install it with: py -m pip install matplotlib") from e

    severity_grid = _severity_grid_from_config(cfg)
    model_ids = [m for m in _model_ids_from_config(cfg) if m in {"mlp", "sage", "pmp", "secgfd"}]

    # Build plot summary rows (mean/std across all replicates).
    summary_rows: list[dict[str, Any]] = []

    metrics = [
        ("roc_auc", "ROC-AUC"),
        ("average_precision", "Average Precision"),
        ("f1_macro", "F1-macro"),
    ]

    # Colors chosen to be stable and distinct.
    colors = {
        "mlp": "#1f77b4",
        "sage": "#ff7f0e",
        "pmp": "#2ca02c",
        "secgfd": "#d62728",
    }

    ds_splits = sorted(set((r.dataset_id, r.split_id, r.protocol) for r in ok))

    for dataset_id, split_id, protocol in ds_splits:
        ok_g = [r for r in ok if r.dataset_id == dataset_id and r.split_id == split_id and r.protocol == protocol]

        # Clean baseline stats per model (within this dataset/split).
        clean_by_model = {}
        for mid in model_ids:
            sel = [
                r
                for r in ok_g
                if r.model_id == mid and r.scenario_id == "clean" and float(r.severity) == 0.0
            ]
            clean_by_model[mid] = {
                "roc_auc": _aggregate_mean_std([r.roc_auc for r in sel]),
                "average_precision": _aggregate_mean_std([r.average_precision for r in sel]),
                "f1_macro": _aggregate_mean_std([r.f1_macro for r in sel]),
            }

        out_group_dir = plots_dir / protocol / dataset_id / split_id
        out_group_dir.mkdir(parents=True, exist_ok=True)

        for scenario in cfg.get("scenarios", []):
            scenario_id = str(scenario.get("scenario_id", ""))
            sevs = severity_grid.get(scenario_id, [])
            sevs_nonzero = [s for s in sevs if float(s) != 0.0]
            sevs_plot = [0.0] + sorted(sevs_nonzero)

            for metric_key, _metric_label in metrics:
                fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=200)

                for mid in model_ids:
                    y = []
                    yerr = []

                    # Baseline (severity 0.0) comes from the single clean graph row.
                    m0, s0, n0 = clean_by_model.get(mid, {}).get(metric_key, (float("nan"), float("nan"), 0))
                    y.append(m0)
                    yerr.append(s0)
                    summary_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "split_id": split_id,
                            "scenario_id": scenario_id,
                            "severity": 0.0,
                            "model_id": mid,
                            "protocol": protocol,
                            "metric": metric_key,
                            "mean": m0,
                            "std": s0,
                            "n_runs": n0,
                            "source": "clean",
                        }
                    )

                    for sev in sorted(sevs_nonzero):
                        sel = [
                            r
                            for r in ok_g
                            if r.model_id == mid and r.scenario_id == scenario_id and float(r.severity) == float(sev)
                        ]
                        vals = [getattr(r, metric_key) for r in sel]
                        m, s, n = _aggregate_mean_std(vals)
                        y.append(m)
                        yerr.append(s)
                        summary_rows.append(
                            {
                                "dataset_id": dataset_id,
                                "split_id": split_id,
                                "scenario_id": scenario_id,
                                "severity": float(sev),
                                "model_id": mid,
                                "protocol": protocol,
                                "metric": metric_key,
                                "mean": m,
                                "std": s,
                                "n_runs": n,
                                "source": "scenario",
                            }
                        )

                    ax.errorbar(
                        sevs_plot,
                        y,
                        yerr=yerr,
                        label=mid,
                        color=colors.get(mid, None),
                        marker="o",
                        linewidth=1.8,
                        markersize=4,
                        capsize=3,
                    )

                ax.set_title(f"{dataset_id}/{split_id} [{protocol}] - {scenario_id}: {metric_key} vs severity")
                ax.set_xlabel("Severity")
                ax.set_ylabel(metric_key)
                ax.set_xticks(sevs_plot)
                ax.grid(True, alpha=0.25)
                ax.legend(title="model", fontsize=8, title_fontsize=8, loc="best")
                fig.tight_layout()

                out_png = out_group_dir / f"curve__{scenario_id}__{metric_key}.png"
                fig.savefig(out_png)
                plt.close(fig)

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
            "mean",
            "std",
            "n_runs",
            "source",
        ],
        summary_rows,
    )

    # Performance drop: clean -> max severity for each scenario/model/metric.
    drop_rows: list[dict[str, Any]] = []
    for dataset_id, split_id, protocol in ds_splits:
        ok_g = [r for r in ok if r.dataset_id == dataset_id and r.split_id == split_id and r.protocol == protocol]

        clean_by_model = {}
        for mid in model_ids:
            sel = [
                r
                for r in ok_g
                if r.model_id == mid and r.scenario_id == "clean" and float(r.severity) == 0.0
            ]
            clean_by_model[mid] = {
                "roc_auc": _aggregate_mean_std([r.roc_auc for r in sel]),
                "average_precision": _aggregate_mean_std([r.average_precision for r in sel]),
                "f1_macro": _aggregate_mean_std([r.f1_macro for r in sel]),
            }

        for scenario in cfg.get("scenarios", []):
            scenario_id = str(scenario.get("scenario_id", ""))
            sevs = severity_grid.get(scenario_id, [])
            sevs_nonzero = [s for s in sevs if float(s) != 0.0]
            if not sevs_nonzero:
                continue
            max_sev = float(max(sevs_nonzero))

            for mid in model_ids:
                for metric_key, _metric_label in metrics:
                    clean_mean, _clean_std, _ = clean_by_model.get(mid, {}).get(metric_key, (float("nan"), float("nan"), 0))
                    sel = [
                        r
                        for r in ok_g
                        if r.model_id == mid and r.scenario_id == scenario_id and float(r.severity) == float(max_sev)
                    ]
                    m_max, _s_max, _n_max = _aggregate_mean_std([getattr(r, metric_key) for r in sel])
                    drop = clean_mean - m_max if (math.isfinite(clean_mean) and math.isfinite(m_max)) else float("nan")
                    drop_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "split_id": split_id,
                            "scenario_id": scenario_id,
                            "model_id": mid,
                            "protocol": protocol,
                            "metric": metric_key,
                            "max_severity": max_sev,
                            "clean_mean": clean_mean,
                            "max_severity_mean": m_max,
                            "drop": drop,
                        }
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
            "clean_mean",
            "max_severity_mean",
            "drop",
        ],
        drop_rows,
    )

    # Robustness score (optional): average metric across severity range (trapz / range).
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None

    if np is not None:
        robust_rows: list[dict[str, Any]] = []
        for dataset_id, split_id, protocol in ds_splits:
            ok_g = [r for r in ok if r.dataset_id == dataset_id and r.split_id == split_id and r.protocol == protocol]

            clean_by_model = {}
            for mid in model_ids:
                sel = [
                    r
                    for r in ok_g
                    if r.model_id == mid and r.scenario_id == "clean" and float(r.severity) == 0.0
                ]
                clean_by_model[mid] = {
                    "roc_auc": _aggregate_mean_std([r.roc_auc for r in sel]),
                    "average_precision": _aggregate_mean_std([r.average_precision for r in sel]),
                    "f1_macro": _aggregate_mean_std([r.f1_macro for r in sel]),
                }

            for scenario in cfg.get("scenarios", []):
                scenario_id = str(scenario.get("scenario_id", ""))
                sevs = severity_grid.get(scenario_id, [])
                sevs_nonzero = [s for s in sevs if float(s) != 0.0]
                sevs_plot = [0.0] + sorted(sevs_nonzero)
                if len(sevs_plot) < 2:
                    continue
                x = np.array(sevs_plot, dtype=np.float64)
                x_range = float(x.max() - x.min())
                if x_range <= 0:
                    continue

                for mid in model_ids:
                    for metric_key, _metric_label in metrics:
                        y_vals = []
                        for sev in sevs_plot:
                            if float(sev) == 0.0:
                                m0, _s0, _n0 = clean_by_model.get(mid, {}).get(metric_key, (float("nan"), float("nan"), 0))
                                y_vals.append(m0)
                            else:
                                sel = [
                                    r
                                    for r in ok_g
                                    if r.model_id == mid
                                    and r.scenario_id == scenario_id
                                    and float(r.severity) == float(sev)
                                ]
                                m, _s, _n = _aggregate_mean_std([getattr(r, metric_key) for r in sel])
                                y_vals.append(m)

                        y = np.array(y_vals, dtype=np.float64)
                        if not np.isfinite(y).all():
                            continue
                        auc = float(np.trapz(y, x))
                        avg_over_range = float(auc / x_range)
                        robust_rows.append(
                            {
                                "dataset_id": dataset_id,
                                "split_id": split_id,
                                "scenario_id": scenario_id,
                                "model_id": mid,
                                "protocol": protocol,
                                "metric": metric_key,
                                "robustness_auc": auc,
                                "robustness_avg_metric": avg_over_range,
                            }
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
                "robustness_auc",
                "robustness_avg_metric",
            ],
            robust_rows,
        )
