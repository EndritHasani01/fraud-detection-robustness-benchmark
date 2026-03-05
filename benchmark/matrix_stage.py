from __future__ import annotations

from pathlib import Path
from typing import Any

from .results import PROTOCOL_TRAIN_CLEAN_EVAL_ALL


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
        )
        return

    from .baselines_stage import run_baselines_stage
    from .pmp_stage import run_pmp_stage
    from .secgfd_stage import run_secgfd_stage

    # Only the first stage should overwrite results.csv when --force is used.
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
        secgfd_hid_dim=secgfd_hid_dim,
        secgfd_order_d=secgfd_order_d,
        secgfd_high_order=secgfd_high_order,
    )

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
    )

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
    )
