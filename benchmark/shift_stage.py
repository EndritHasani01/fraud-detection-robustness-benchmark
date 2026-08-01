from __future__ import annotations

import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

from .baselines_stage import (
    build_baseline_hparams,
    eval_baseline_model,
    train_baseline_model,
)
from .config import get_training_seeds
from .preflight import (
    ProgressTracker,
    build_expected_run_keys,
    print_training_preflight,
    select_model_ids,
    summarize_training_preflight,
    warn_no_matching_models,
)
from .pmp_stage import eval_pmp_model, resolve_pmp_config, train_pmp_model
from .results import (
    PROTOCOL_TRAIN_CLEAN_EVAL_ALL,
    ensure_results_csv,
    load_completed_keys,
    make_run_key,
    truncate_error_message,
    write_result_row,
)
from .secgfd_stage import (
    eval_secgfd_model,
    resolve_secgfd_hparams,
    resolve_secgfd_training_controls,
    train_secgfd_model,
)
from .summarize import summarize_results_by_training_seed
from .variants import VariantRow, filter_variants, load_graph_bin, require_variants_csv_rows


def _supported_model_ids(cfg: dict[str, Any]) -> list[str]:
    mids = []
    for m in cfg.get("models", []):
        mid = str(m.get("model_id", ""))
        if mid in {"mlp", "sage", "pmp", "secgfd"}:
            mids.append(mid)
    seen = set()
    out = []
    for mid in mids:
        if mid in seen:
            continue
        seen.add(mid)
        out.append(mid)
    return out


def _group_variants_by_split(variants: list[VariantRow]) -> list[tuple[tuple[str, str], list[VariantRow]]]:
    grouped: dict[tuple[str, str], list[VariantRow]] = defaultdict(list)
    order: list[tuple[str, str]] = []
    for v in variants:
        key = (v.dataset_id, v.split_id)
        if key not in grouped:
            order.append(key)
        grouped[key].append(v)
    return [(key, grouped[key]) for key in order]


def _clean_variant_for_group(variants: list[VariantRow]) -> VariantRow:
    for v in variants:
        if v.scenario_id == "clean" and float(v.severity) == 0.0:
            return v
    raise RuntimeError("Shift stage requires a clean variant row for each dataset/split group.")


def _train_shift_artifact(
    model_id: str,
    clean_g,
    *,
    dataset_id: str,
    dataset_cfg_by_id: dict[str, dict[str, Any]],
    model_cfg_by_id: dict[str, dict[str, Any]],
    training_seed: int,
    device: str,
    max_epochs: int | None,
    patience: int | None,
    secgfd_hid_dim: int | None,
    secgfd_order_d: int | None,
    secgfd_high_order: int | None,
):
    if model_id in {"mlp", "sage"}:
        hp = build_baseline_hparams(model_id, max_epochs=max_epochs, patience=patience)
        return train_baseline_model(
            model_id,
            clean_g,
            training_seed=int(training_seed),
            device=str(device),
            hparams=hp,
        )

    if model_id == "pmp":
        model_cfg = model_cfg_by_id.get("pmp", {})
        repo_root = Path(str(model_cfg.get("repo_path", "")))
        if not repo_root.exists():
            raise FileNotFoundError(f"Configured PMP repo_path does not exist: {repo_root}")
        ds_cfg = dataset_cfg_by_id.get(dataset_id, {})
        dataset_source_name = str(ds_cfg.get("source_name", "yelp")).strip().lower()
        cfg_pmp = resolve_pmp_config(
            repo_root,
            dataset_source_name=dataset_source_name,
            model_cfg=model_cfg,
            model_name="LA-SAGE-S",
        )
        return train_pmp_model(
            clean_g,
            repo_root=repo_root,
            cfg_pmp=cfg_pmp,
            training_seed=int(training_seed),
            device=str(device),
            max_epochs=max_epochs,
            patience=patience,
        )

    if model_id == "secgfd":
        model_cfg = model_cfg_by_id.get("secgfd", {})
        repo_root = Path(str(model_cfg.get("repo_path", "")))
        if not repo_root.exists():
            raise FileNotFoundError(f"Configured SEC-GFD repo_path does not exist: {repo_root}")
        secgfd_hparams = resolve_secgfd_hparams(
            model_cfg,
            hid_dim_override=secgfd_hid_dim,
            order_d_override=secgfd_order_d,
            high_order_override=secgfd_high_order,
        )
        effective_max_epochs, effective_patience = resolve_secgfd_training_controls(
            max_epochs=max_epochs,
            patience=patience,
        )
        return train_secgfd_model(
            clean_g,
            repo_root=repo_root,
            training_seed=int(training_seed),
            device=str(device),
            max_epochs=effective_max_epochs,
            patience=effective_patience,
            hid_dim=int(secgfd_hparams["hid_dim"]),
            order_d=int(secgfd_hparams["order_d"]),
            high_order=int(secgfd_hparams["high_order"]),
            lemda=float(secgfd_hparams["lemda"]),
            lr=float(secgfd_hparams["lr"]),
            weight_decay=float(secgfd_hparams["weight_decay"]),
        )

    raise ValueError(f"Unsupported shift model_id: {model_id}")


def _eval_shift_artifact(model_id: str, artifact, g) -> dict[str, Any]:
    if model_id in {"mlp", "sage"}:
        return eval_baseline_model(artifact, g)
    if model_id == "pmp":
        return eval_pmp_model(artifact, g)
    if model_id == "secgfd":
        return eval_secgfd_model(artifact, g)
    raise ValueError(f"Unsupported shift model_id: {model_id}")


def run_shift_stage(
    cfg: dict[str, Any],
    *,
    out_dir: Path,
    force: bool,
    skip_existing: bool,
    retry_errors: bool,
    device: str,
    include_noop: bool,
    only_clean: bool,
    max_variants: int | None,
    max_training_seeds: int | None,
    max_epochs: int | None,
    patience: int | None,
    secgfd_hid_dim: int | None = None,
    secgfd_order_d: int | None = None,
    secgfd_high_order: int | None = None,
    selected_model_ids: list[str] | None = None,
) -> None:
    out_dir = out_dir.resolve()
    results_csv = out_dir / "results.csv"
    variants_csv = out_dir / "graph_variants.csv"

    ensure_results_csv(results_csv, overwrite=bool(force))
    completed_keys = load_completed_keys(results_csv, retry_errors=bool(retry_errors)) if skip_existing else set()

    variants = require_variants_csv_rows(variants_csv)

    filtered = filter_variants(
        variants,
        include_noop=bool(include_noop),
        only_clean=bool(only_clean),
        max_variants=max_variants,
    )
    if not filtered:
        summarize_results_by_training_seed(
            results_csv,
            out_csv_path=out_dir / "results_summary_shift.csv",
            model_ids=set(),
            protocols={PROTOCOL_TRAIN_CLEAN_EVAL_ALL},
        )
        return

    training_seeds = get_training_seeds(cfg)
    if max_training_seeds is not None:
        training_seeds = training_seeds[: int(max_training_seeds)]

    dataset_cfg_by_id = {str(d.get("dataset_id", "")): d for d in cfg.get("datasets", [])}
    model_cfg_by_id = {str(m.get("model_id", "")): m for m in cfg.get("models", [])}
    model_ids = select_model_ids(
        cfg,
        supported_model_ids={"mlp", "sage", "pmp", "secgfd"},
        requested_model_ids=selected_model_ids,
    )
    if not model_ids:
        warn_no_matching_models("shift", selected_model_ids)
        return

    expected_keys = build_expected_run_keys(
        filtered,
        training_seeds=training_seeds,
        model_ids=model_ids,
        protocol=PROTOCOL_TRAIN_CLEAN_EVAL_ALL,
    )
    preflight = summarize_training_preflight(
        results_csv,
        expected_keys=expected_keys,
        completed_keys=completed_keys,
        skip_existing=bool(skip_existing),
        model_ids=model_ids,
        protocol=PROTOCOL_TRAIN_CLEAN_EVAL_ALL,
    )
    print_training_preflight(
        variant_count=len(filtered),
        training_seed_count=len(training_seeds),
        model_ids=model_ids,
        summary=preflight,
    )
    progress = ProgressTracker(stage_label="shift", total_runs=preflight.total_runs, already_done=preflight.already_done)

    all_groups = dict(_group_variants_by_split(variants))
    for (dataset_id, _split_id), group_variants in _group_variants_by_split(filtered):
        clean_variant = _clean_variant_for_group(all_groups[(dataset_id, _split_id)])
        clean_graph_path = str(clean_variant.graph_path)

        clean_graph = None
        clean_graph_error: str | None = None
        clean_graph_error_dt = 0.0

        for model_id in model_ids:
            for training_seed in training_seeds:
                pending_variants: list[tuple[VariantRow, tuple[str, str, str, float, int, int, str, str]]] = []

                for v in group_variants:
                    run_key = make_run_key(
                        dataset_id=v.dataset_id,
                        split_id=v.split_id,
                        scenario_id=v.scenario_id,
                        severity=v.severity,
                        graph_seed=v.graph_seed,
                        training_seed=training_seed,
                        model_id=model_id,
                        protocol=PROTOCOL_TRAIN_CLEAN_EVAL_ALL,
                    )
                    if run_key in completed_keys:
                        print(
                            f"[skip] {model_id} / {v.scenario_id} / sev={float(v.severity):g} / "
                            f"gs={int(v.graph_seed)} / ts={int(training_seed)} already done"
                        )
                        continue
                    pending_variants.append((v, run_key))

                if not pending_variants:
                    continue

                if clean_graph is None and clean_graph_error is None:
                    clean_load_t0 = time.perf_counter()
                    try:
                        clean_graph = load_graph_bin(Path(clean_graph_path))
                    except Exception as e:
                        traceback.print_exc()
                        clean_graph_error = truncate_error_message(e)
                        clean_graph_error_dt = time.perf_counter() - clean_load_t0

                if clean_graph_error is not None:
                    for variant, run_key in pending_variants:
                        write_result_row(
                            results_csv,
                            variant,
                            model_id=str(model_id),
                            training_seed=int(training_seed),
                            protocol=PROTOCOL_TRAIN_CLEAN_EVAL_ALL,
                            metrics=None,
                            error=clean_graph_error,
                            duration_sec=float(clean_graph_error_dt),
                            train_graph_ref=clean_graph_path,
                        )
                        progress.record(
                            model_id=str(model_id),
                            scenario_id=variant.scenario_id,
                            severity=float(variant.severity),
                            graph_seed=int(variant.graph_seed),
                            training_seed=int(training_seed),
                            status="error",
                            duration_sec=float(clean_graph_error_dt),
                        )
                        completed_keys.add(run_key)
                    continue

                train_t0 = time.perf_counter()
                try:
                    artifact = _train_shift_artifact(
                        model_id,
                        clean_graph,
                        dataset_id=dataset_id,
                        dataset_cfg_by_id=dataset_cfg_by_id,
                        model_cfg_by_id=model_cfg_by_id,
                        training_seed=int(training_seed),
                        device=str(device),
                        max_epochs=max_epochs,
                        patience=patience,
                        secgfd_hid_dim=secgfd_hid_dim,
                        secgfd_order_d=secgfd_order_d,
                        secgfd_high_order=secgfd_high_order,
                    )
                    train_dt = time.perf_counter() - train_t0
                except Exception as e:
                    traceback.print_exc()
                    train_error = truncate_error_message(e)
                    dt = time.perf_counter() - train_t0
                    for variant, run_key in pending_variants:
                        write_result_row(
                            results_csv,
                            variant,
                            model_id=str(model_id),
                            training_seed=int(training_seed),
                            protocol=PROTOCOL_TRAIN_CLEAN_EVAL_ALL,
                            metrics=None,
                            error=train_error,
                            duration_sec=float(dt),
                            train_graph_ref=clean_graph_path,
                        )
                        progress.record(
                            model_id=str(model_id),
                            scenario_id=variant.scenario_id,
                            severity=float(variant.severity),
                            graph_seed=int(variant.graph_seed),
                            training_seed=int(training_seed),
                            status="error",
                            duration_sec=float(dt),
                        )
                        completed_keys.add(run_key)
                    continue

                for variant, run_key in pending_variants:
                    eval_t0 = time.perf_counter()
                    try:
                        eval_graph = load_graph_bin(Path(variant.graph_path))
                        out = _eval_shift_artifact(model_id, artifact, eval_graph)
                        duration = float(out["duration_sec"])
                        if variant.scenario_id == "clean" and float(variant.severity) == 0.0:
                            duration += float(train_dt)
                        write_result_row(
                            results_csv,
                            variant,
                            model_id=str(model_id),
                            training_seed=int(training_seed),
                            protocol=PROTOCOL_TRAIN_CLEAN_EVAL_ALL,
                            metrics=out,
                            duration_sec=float(duration),
                            train_graph_ref=clean_graph_path,
                        )
                        progress.record(
                            model_id=str(model_id),
                            scenario_id=variant.scenario_id,
                            severity=float(variant.severity),
                            graph_seed=int(variant.graph_seed),
                            training_seed=int(training_seed),
                            status="ok",
                            duration_sec=float(duration),
                            roc_auc=float(out["roc_auc"]),
                        )
                    except Exception as e:
                        traceback.print_exc()
                        dt = time.perf_counter() - eval_t0
                        if variant.scenario_id == "clean" and float(variant.severity) == 0.0:
                            dt += float(train_dt)
                        write_result_row(
                            results_csv,
                            variant,
                            model_id=str(model_id),
                            training_seed=int(training_seed),
                            protocol=PROTOCOL_TRAIN_CLEAN_EVAL_ALL,
                            metrics=None,
                            error=e,
                            duration_sec=float(dt),
                            train_graph_ref=clean_graph_path,
                        )
                        progress.record(
                            model_id=str(model_id),
                            scenario_id=variant.scenario_id,
                            severity=float(variant.severity),
                            graph_seed=int(variant.graph_seed),
                            training_seed=int(training_seed),
                            status="error",
                            duration_sec=float(dt),
                        )
                    finally:
                        completed_keys.add(run_key)

    summarize_results_by_training_seed(
        results_csv,
        out_csv_path=out_dir / "results_summary_shift.csv",
        model_ids=set(model_ids),
        protocols={PROTOCOL_TRAIN_CLEAN_EVAL_ALL},
    )
