from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import get_training_seeds
from .preflight import (
    build_expected_run_keys,
    filter_variant_rows,
    print_training_preflight,
    select_model_ids,
    summarize_training_preflight,
    warn_no_matching_models,
)
from .results import PROTOCOL_TRAIN_CLEAN_EVAL_ALL, PROTOCOL_TRAIN_ON_VARIANT, load_completed_keys


def run_matrix_stage(
    cfg: dict[str, Any],
    *,
    out_dir: Path,
    force: bool,
    protocol: str,
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
    """Run the full experiment matrix for all integrated models.

    This is a thin launcher that runs:
    - baselines (mlp, sage)
    - pmp
    - secgfd

    It assumes `--stage graphs` already produced cached graphs and `graph_variants.csv`
    in the same out_dir.
    """
    out_dir = out_dir.resolve()
    variants_csv = out_dir / "graph_variants.csv"
    if not variants_csv.exists():
        raise FileNotFoundError(
            f"Missing {variants_csv}. Run `py -m benchmark.run --stage graphs ...` first (same --out directory)."
        )

    from .baselines_stage import _read_variants_csv

    configured_integrated_model_ids = select_model_ids(
        cfg,
        supported_model_ids={"mlp", "sage", "pmp", "secgfd"},
        requested_model_ids=None,
    )
    integrated_model_ids = select_model_ids(
        cfg,
        supported_model_ids={"mlp", "sage", "pmp", "secgfd"},
        requested_model_ids=selected_model_ids,
        default_order=(["mlp", "sage", "pmp", "secgfd"] if not configured_integrated_model_ids else None),
    )
    if not integrated_model_ids:
        warn_no_matching_models("matrix", selected_model_ids)
        return

    variants = _read_variants_csv(variants_csv)
    filtered_variants = filter_variant_rows(
        variants,
        include_noop=bool(include_noop),
        only_clean=bool(only_clean),
        max_variants=max_variants,
    )
    training_seeds = get_training_seeds(cfg)
    if max_training_seeds is not None:
        training_seeds = training_seeds[: int(max_training_seeds)]

    results_csv = out_dir / "results.csv"
    completed_keys = load_completed_keys(results_csv, retry_errors=bool(retry_errors)) if skip_existing else set()
    active_protocol = PROTOCOL_TRAIN_CLEAN_EVAL_ALL if str(protocol) == PROTOCOL_TRAIN_CLEAN_EVAL_ALL else PROTOCOL_TRAIN_ON_VARIANT
    expected_keys = build_expected_run_keys(
        filtered_variants,
        training_seeds=training_seeds,
        model_ids=integrated_model_ids,
        protocol=active_protocol,
    )
    summary = summarize_training_preflight(
        results_csv,
        expected_keys=expected_keys,
        completed_keys=completed_keys,
        skip_existing=bool(skip_existing),
        model_ids=integrated_model_ids,
        protocol=active_protocol,
    )
    print_training_preflight(
        variant_count=len(filtered_variants),
        training_seed_count=len(training_seeds),
        model_ids=integrated_model_ids,
        summary=summary,
    )

    if str(protocol) == PROTOCOL_TRAIN_CLEAN_EVAL_ALL:
        from .shift_stage import run_shift_stage

        run_shift_stage(
            cfg,
            out_dir=out_dir,
            force=bool(force),
            skip_existing=bool(skip_existing),
            retry_errors=bool(retry_errors),
            device=str(device),
            include_noop=bool(include_noop),
            only_clean=bool(only_clean),
            max_variants=max_variants,
            max_training_seeds=max_training_seeds,
            max_epochs=max_epochs,
            patience=patience,
            secgfd_hid_dim=secgfd_hid_dim,
            secgfd_order_d=secgfd_order_d,
            secgfd_high_order=secgfd_high_order,
            selected_model_ids=selected_model_ids,
        )
        return

    from .baselines_stage import run_baselines_stage
    from .pmp_stage import run_pmp_stage
    from .secgfd_stage import run_secgfd_stage

    # Only the first stage should overwrite results.csv when --force is used.
    if any(mid in {"mlp", "sage"} for mid in integrated_model_ids):
        run_baselines_stage(
            cfg,
            out_dir=out_dir,
            force=bool(force),
            skip_existing=bool(skip_existing),
            retry_errors=bool(retry_errors),
            device=str(device),
            include_noop=bool(include_noop),
            only_clean=bool(only_clean),
            max_variants=max_variants,
            max_training_seeds=max_training_seeds,
            max_epochs=max_epochs,
            patience=patience,
            selected_model_ids=selected_model_ids,
        )

    if "pmp" in integrated_model_ids:
        run_pmp_stage(
            cfg,
            out_dir=out_dir,
            force=False,
            skip_existing=bool(skip_existing),
            retry_errors=bool(retry_errors),
            device=str(device),
            include_noop=bool(include_noop),
            only_clean=bool(only_clean),
            max_variants=max_variants,
            max_training_seeds=max_training_seeds,
            max_epochs=max_epochs,
            patience=patience,
            selected_model_ids=selected_model_ids,
        )

    if "secgfd" in integrated_model_ids:
        run_secgfd_stage(
            cfg,
            out_dir=out_dir,
            force=False,
            skip_existing=bool(skip_existing),
            retry_errors=bool(retry_errors),
            device=str(device),
            include_noop=bool(include_noop),
            only_clean=bool(only_clean),
            max_variants=max_variants,
            max_training_seeds=max_training_seeds,
            max_epochs=max_epochs,
            patience=patience,
            secgfd_hid_dim=secgfd_hid_dim,
            secgfd_order_d=secgfd_order_d,
            secgfd_high_order=secgfd_high_order,
            selected_model_ids=selected_model_ids,
        )
