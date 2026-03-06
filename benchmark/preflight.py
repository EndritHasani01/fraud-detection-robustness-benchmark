from __future__ import annotations

import csv
import math
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from .config import get_graph_seeds
from .results import RunKey, make_run_key, normalize_protocol, row_run_key
from .variants import filter_variants


def parse_requested_model_ids(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    parts = [part.strip() for part in str(raw).split(",")]
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if not part:
            continue
        model_id = part.lower()
        if model_id in seen:
            continue
        seen.add(model_id)
        out.append(model_id)
    return out or None


def select_model_ids(
    cfg: dict[str, Any],
    *,
    supported_model_ids: Iterable[str],
    requested_model_ids: Sequence[str] | None,
    default_order: Sequence[str] | None = None,
) -> list[str]:
    supported = set(str(mid) for mid in supported_model_ids)

    ordered: list[str] = []
    seen: set[str] = set()
    for model_cfg in cfg.get("models", []):
        model_id = str(model_cfg.get("model_id", ""))
        if model_id not in supported or model_id in seen:
            continue
        seen.add(model_id)
        ordered.append(model_id)

    for model_id in default_order or []:
        mid = str(model_id)
        if mid not in supported or mid in seen:
            continue
        seen.add(mid)
        ordered.append(mid)

    if requested_model_ids is None:
        return ordered

    requested = set(str(mid) for mid in requested_model_ids)
    return [mid for mid in ordered if mid in requested]


def warn_no_matching_models(stage_label: str, requested_model_ids: Sequence[str] | None) -> None:
    requested = ",".join(requested_model_ids or [])
    suffix = f" from --models={requested}" if requested else ""
    print(f"[warn] {stage_label}: no matching models selected{suffix}; nothing to do")


def filter_variant_rows(
    variants: Sequence[Any],
    *,
    include_noop: bool,
    only_clean: bool,
    max_variants: int | None,
) -> list[Any]:
    return filter_variants(
        variants,
        include_noop=include_noop,
        only_clean=only_clean,
        max_variants=max_variants,
    )


def build_expected_run_keys(
    variants: Sequence[Any],
    *,
    training_seeds: Sequence[int],
    model_ids: Sequence[str],
    protocol: str,
) -> set[RunKey]:
    expected: set[RunKey] = set()
    for variant in variants:
        for training_seed in training_seeds:
            for model_id in model_ids:
                expected.add(
                    make_run_key(
                        dataset_id=getattr(variant, "dataset_id"),
                        split_id=getattr(variant, "split_id"),
                        scenario_id=getattr(variant, "scenario_id"),
                        severity=getattr(variant, "severity"),
                        graph_seed=getattr(variant, "graph_seed"),
                        training_seed=training_seed,
                        model_id=model_id,
                        protocol=protocol,
                    )
                )
    return expected


def count_graph_variant_rows(cfg: dict[str, Any]) -> int:
    n_datasets = len(cfg.get("datasets", []))
    n_splits = len(cfg.get("data_splits", []))
    graph_seeds = get_graph_seeds(cfg)
    clean_rows = n_datasets * n_splits
    scenario_rows = 0
    for scenario_cfg in cfg.get("scenarios", []):
        scenario_rows += len(scenario_cfg.get("severity_values", [])) * len(graph_seeds)
    return clean_rows + (n_datasets * n_splits * scenario_rows)


def print_graphs_preflight(cfg: dict[str, Any]) -> None:
    total_variants = count_graph_variant_rows(cfg)
    print(f"[preflight] graphs will generate {total_variants} variant rows")


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def load_duration_medians(
    results_csv: Path,
    *,
    model_ids: Sequence[str],
    protocol: str | None,
) -> dict[str, float]:
    if not results_csv.exists():
        return {}

    selected = set(str(mid) for mid in model_ids)
    durations: dict[str, list[float]] = {mid: [] for mid in selected}

    with results_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("status", "")).strip().lower() != "ok":
                continue
            model_id = str(row.get("model_id", ""))
            if model_id not in selected:
                continue
            if protocol is not None and normalize_protocol(row.get("protocol", "")) != normalize_protocol(protocol):
                continue
            duration = _safe_float(row.get("duration_sec", ""))
            if math.isfinite(duration) and duration >= 0.0:
                durations[model_id].append(duration)

    medians: dict[str, float] = {}
    for model_id, values in durations.items():
        if values:
            medians[model_id] = float(median(values))
    return medians


@dataclass(frozen=True)
class TrainingPreflightSummary:
    total_runs: int
    already_done: int
    remaining_runs: int
    estimated_remaining_sec: float | None
    medians_by_model: dict[str, float]


def summarize_training_preflight(
    results_csv: Path,
    *,
    expected_keys: set[RunKey],
    completed_keys: set[RunKey],
    skip_existing: bool,
    model_ids: Sequence[str],
    protocol: str,
) -> TrainingPreflightSummary:
    total_runs = len(expected_keys)
    already_done = len(expected_keys & completed_keys) if skip_existing else 0
    remaining_keys = expected_keys - completed_keys if skip_existing else set(expected_keys)
    remaining_runs = len(remaining_keys)
    medians_by_model = load_duration_medians(results_csv, model_ids=model_ids, protocol=protocol)

    remaining_counts = Counter(key[6] for key in remaining_keys)
    if not remaining_counts:
        estimated_remaining_sec: float | None = 0.0
    elif all(model_id in medians_by_model for model_id in remaining_counts):
        estimated_remaining_sec = float(
            sum(int(count) * float(medians_by_model[model_id]) for model_id, count in remaining_counts.items())
        )
    else:
        estimated_remaining_sec = None

    return TrainingPreflightSummary(
        total_runs=total_runs,
        already_done=already_done,
        remaining_runs=remaining_runs,
        estimated_remaining_sec=estimated_remaining_sec,
        medians_by_model=medians_by_model,
    )


def format_duration_compact(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "n/a"

    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.0f}s"

    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m" if sec == 0 else f"{minutes}m {sec}s"

    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def print_training_preflight(
    *,
    variant_count: int,
    training_seed_count: int,
    model_ids: Sequence[str],
    summary: TrainingPreflightSummary,
) -> None:
    print(
        f"[preflight] {variant_count} variants x {training_seed_count} seeds x {len(model_ids)} models = "
        f"{summary.total_runs} runs total"
    )
    print(f"[preflight] {summary.already_done} already done, {summary.remaining_runs} remaining")

    if summary.estimated_remaining_sec is None:
        print("[preflight] no estimate available")
        return

    detail_parts = [
        f"{model_id} {summary.medians_by_model[model_id]:.1f}s/run"
        for model_id in model_ids
        if model_id in summary.medians_by_model
    ]
    detail = ", ".join(detail_parts)
    if detail:
        print(f"[preflight] estimated ~{format_duration_compact(summary.estimated_remaining_sec)} ({detail})")
    else:
        print(f"[preflight] estimated ~{format_duration_compact(summary.estimated_remaining_sec)}")


class ProgressTracker:
    def __init__(self, *, stage_label: str, total_runs: int, already_done: int) -> None:
        self.stage_label = str(stage_label)
        self.total_runs = int(total_runs)
        self.already_done = int(already_done)
        self.completed_now = 0
        self.recent_durations: deque[float] = deque(maxlen=5)

    def record(
        self,
        *,
        model_id: str,
        scenario_id: str,
        severity: float,
        graph_seed: int,
        training_seed: int,
        status: str,
        duration_sec: float,
        roc_auc: float | None = None,
    ) -> None:
        self.completed_now += 1
        duration_sec = float(duration_sec)
        if math.isfinite(duration_sec) and duration_sec >= 0.0:
            self.recent_durations.append(duration_sec)

        position = self.already_done + self.completed_now
        status_text = str(status).strip().lower() or "unknown"
        auc_text = ""
        if status_text == "ok" and roc_auc is not None and math.isfinite(float(roc_auc)):
            auc_text = f"  auc={float(roc_auc):.3f}"

        print(
            f"[{self.stage_label}] [{position}/{self.total_runs}] "
            f"{model_id} / {scenario_id} / sev={float(severity):g} / gs={int(graph_seed)} / ts={int(training_seed)}  "
            f"{status_text}{auc_text}  {duration_sec:.1f}s"
        )

        should_print_eta = self.total_runs < 50 or (self.completed_now % 10 == 0)
        if should_print_eta and self.recent_durations:
            avg_recent = sum(self.recent_durations) / len(self.recent_durations)
            remaining = max(0, self.total_runs - position)
            eta_sec = avg_recent * remaining
            print(f"[{self.stage_label}] [{position}/{self.total_runs}] ETA ~{format_duration_compact(eta_sec)}")
