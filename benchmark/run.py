from __future__ import annotations

import argparse
import time
from pathlib import Path

from .config import ConfigError, get_graph_seeds, init_paths, load_json, validate_config, write_json
from .paths import base_graph_path, variant_graph_path
from .results import (
    PROTOCOL_TRAIN_CLEAN_EVAL_ALL,
    PROTOCOL_TRAIN_ON_VARIANT,
    VARIANTS_COLUMNS,
    append_csv_row,
    ensure_csv_header,
    ensure_results_csv,
)


def _default_out_dir(cfg: dict, config_path: Path) -> Path:
    # Default: runs/<experiment_name> next to repo root.
    # If this is run from somewhere else, it still behaves reasonably.
    return Path("runs") / cfg["experiment_name"]


def _make_base_graph(cfg: dict, dataset_cfg: dict, split_cfg: dict, *, out_dir: Path, force_reload: bool):
    from .data import (
        ensure_feature_dtype,
        ensure_label_dtype,
        ensure_masks,
        load_dgl_fraud_dataset,
        to_canonical_graph,
    )

    dataset_id = dataset_cfg["dataset_id"]
    source_name = dataset_cfg["source_name"]
    split_id = split_cfg["split_id"]
    split_seed = int(split_cfg["split_seed"])
    train_size = float(split_cfg["train_size"])
    val_size = float(split_cfg["val_size"])

    raw_dir = out_dir / "data" / "dgl"
    loaded = load_dgl_fraud_dataset(dataset_id, source_name, raw_dir, force_reload=force_reload)
    g = loaded.graph

    # Make labels/features consistent types before splitting.
    ensure_feature_dtype(g)
    ensure_label_dtype(g)

    # Overwrite masks so we have full control.
    ensure_masks(g, train_size=train_size, val_size=val_size, split_seed=split_seed)

    canonical = cfg["graph_representation"]["canonical_view"]
    g_can = to_canonical_graph(g, canonical)
    return g_can


def _graphs_only(cfg: dict, *, out_dir: Path, force: bool, force_reload: bool) -> None:
    from .cache import save_graph, save_graph_ref
    from .scenarios import ScenarioSpec, apply_scenario
    from .stats import compute_graph_stats

    paths = init_paths(out_dir)
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    # Copy config into output for reproducibility.
    write_json(paths.config_copy_path, cfg)

    # Ensure CSVs exist with correct schema. If --force is set, start fresh.
    ensure_results_csv(paths.results_csv_path, overwrite=force)
    ensure_csv_header(paths.variants_csv_path, VARIANTS_COLUMNS, overwrite=force)

    experiment_name = cfg["experiment_name"]
    graph_seeds = get_graph_seeds(cfg)

    for dataset_cfg in cfg["datasets"]:
        dataset_id = dataset_cfg["dataset_id"]

        for split_cfg in cfg["data_splits"]:
            split_id = split_cfg["split_id"]
            split_seed = int(split_cfg["split_seed"])

            g_base = _make_base_graph(cfg, dataset_cfg, split_cfg, out_dir=paths.out_dir, force_reload=force_reload)
            base_p = base_graph_path(paths.graphs_dir, dataset_id, split_id)

            base_meta = {
                "experiment_name": experiment_name,
                "dataset_id": dataset_id,
                "split_id": split_id,
                "split_seed": split_seed,
                "train_size": float(split_cfg["train_size"]),
                "val_size": float(split_cfg["val_size"]),
                "canonical_view": cfg["graph_representation"]["canonical_view"],
                "created_unix": time.time(),
            }
            save_graph(base_p, g_base, base_meta, force=force)

            base_stats = compute_graph_stats(g_base)

            # Record base as a variant row too (scenario_id="clean").
            append_csv_row(
                paths.variants_csv_path,
                VARIANTS_COLUMNS,
                {
                    "experiment_name": experiment_name,
                    "dataset_id": dataset_id,
                    "split_id": split_id,
                    "graph_seed": split_seed,
                    "scenario_id": "clean",
                    "severity": 0.0,
                    "oracle_labels": False,
                    "scenario_applied": True,
                    "base_graph_path": str(base_p.graph_bin_path),
                    "graph_path": str(base_p.graph_bin_path),
                    **base_stats,
                },
            )

            for scenario_cfg in cfg["scenarios"]:
                scenario_id = scenario_cfg["scenario_id"]
                oracle_labels = bool(scenario_cfg.get("oracle_labels", False))
                severity_values = scenario_cfg["severity_values"]
                scenario_params = {k: v for k, v in scenario_cfg.items() if k not in {"severity_values"}}

                for severity in severity_values:
                    for graph_seed in graph_seeds:
                        spec = ScenarioSpec(
                            scenario_id=scenario_id,
                            family=str(scenario_cfg.get("family", "")),
                            method=str(scenario_cfg.get("method", "")),
                            oracle_labels=oracle_labels,
                            severity_param=str(scenario_cfg.get("severity_param", "severity")),
                            severity=float(severity),
                            graph_seed=int(graph_seed),
                            params=scenario_params,
                        )
                        var_p = variant_graph_path(
                            paths.graphs_dir, dataset_id, split_id, scenario_id, float(severity), int(graph_seed)
                        )

                        g_var, applied, info = apply_scenario(g_base, spec)

                        var_meta = {
                            **base_meta,
                            "base_graph_bin": str(base_p.graph_bin_path.resolve()),
                            "scenario_id": scenario_id,
                            "severity": float(severity),
                            "graph_seed": int(graph_seed),
                            "oracle_labels": oracle_labels,
                            "scenario_applied": bool(applied),
                            "scenario_method": spec.method,
                            "scenario_params": scenario_params,
                            "scenario_info": info,
                        }

                        if applied:
                            save_graph(var_p, g_var, var_meta, force=force)
                            graph_path_for_row = var_p.graph_bin_path
                        else:
                            # No-op variants are stored as refs to avoid duplicating large binaries.
                            save_graph_ref(var_p, base_p.graph_bin_path, var_meta, force=force)
                            graph_path_for_row = base_p.graph_bin_path

                        var_stats = compute_graph_stats(g_var)
                        append_csv_row(
                            paths.variants_csv_path,
                            VARIANTS_COLUMNS,
                            {
                                "experiment_name": experiment_name,
                                "dataset_id": dataset_id,
                                "split_id": split_id,
                                "graph_seed": int(graph_seed),
                                "scenario_id": scenario_id,
                                "severity": float(severity),
                                "oracle_labels": oracle_labels,
                                "scenario_applied": bool(applied),
                                "base_graph_path": str(base_p.graph_bin_path),
                                "graph_path": str(graph_path_for_row),
                                **var_stats,
                            },
                        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Graph Fraud Detection Robustness Benchmark")
    p.add_argument("--config", type=str, required=True, help="Path to experiment JSON config.")
    p.add_argument("--out", type=str, default="", help="Output directory (default: runs/<experiment_name>).")
    p.add_argument(
        "--stage",
        type=str,
        default="graphs",
        choices=["graphs", "baselines", "pmp", "secgfd", "shift", "matrix", "plots"],
        help="Pipeline stage to run. 'graphs' builds splits and caches graph variants; "
        "'baselines' trains/evaluates MLP + GraphSAGE on cached graphs; "
        "'pmp' trains/evaluates PMP (LA-SAGE-S) from Repos/PMP-master on cached graphs; "
        "'secgfd' trains/evaluates SEC-GFD from Repos/SEC-GFD-main on cached graphs; "
        "'shift' trains once on the clean graph and evaluates all cached variants; "
        "'matrix' runs baselines + PMP + SEC-GFD; "
        "'plots' generates figures and plot-ready summaries from results.csv.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite cached outputs if they exist and disable skip-existing for training stages.",
    )
    p.add_argument("--force-reload", action="store_true", help="Force DGL dataset reload/redownload.")
    p.add_argument("--device", type=str, default="cpu", help="Device for model training ('cpu' or 'cuda').")
    p.add_argument(
        "--protocol",
        type=str,
        default=PROTOCOL_TRAIN_ON_VARIANT,
        choices=[PROTOCOL_TRAIN_ON_VARIANT, PROTOCOL_TRAIN_CLEAN_EVAL_ALL],
        help="Evaluation protocol. For --stage matrix, train_clean_eval_all runs the shift protocol.",
    )
    p.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=True,
        help="Skip completed run keys already present in results.csv (default: enabled).",
    )
    p.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Disable resumability checks and append fresh result rows.",
    )
    p.add_argument(
        "--retry-errors",
        action="store_true",
        help="Re-attempt run keys that only have status=error rows instead of skipping them.",
    )
    p.add_argument("--include-noop", action="store_true", help="Include no-op scenario rows (severity==0) in evaluation.")
    p.add_argument("--only-clean", action="store_true", help="Evaluate only the single clean base graph row.")
    p.add_argument("--max-variants", type=int, default=0, help="Evaluate at most N variant rows (0 = no limit).")
    p.add_argument(
        "--max-training-seeds",
        type=int,
        default=0,
        help="Use at most N training seeds from the config (0 = no limit).",
    )
    p.add_argument("--max-epochs", type=int, default=0, help="Override max epochs for training stages (0 = use defaults).")
    p.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Override early-stop patience for training stages (0 = use defaults).",
    )
    args = p.parse_args(argv)

    config_path = Path(args.config)
    try:
        cfg = load_json(config_path)
        validate_config(cfg)
    except ConfigError as e:
        print(f"[config] ERROR: {e}")
        return 2

    out_dir = Path(args.out) if args.out else _default_out_dir(cfg, config_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    effective_skip_existing = False if bool(args.force) else bool(args.skip_existing)

    t0 = time.time()
    try:
        if args.stage == "graphs":
            _graphs_only(cfg, out_dir=out_dir, force=bool(args.force), force_reload=bool(args.force_reload))
        elif args.stage == "baselines":
            from .baselines_stage import run_baselines_stage

            run_baselines_stage(
                cfg,
                out_dir=out_dir,
                force=bool(args.force),
                skip_existing=effective_skip_existing,
                retry_errors=bool(args.retry_errors),
                device=str(args.device),
                include_noop=bool(args.include_noop),
                only_clean=bool(args.only_clean),
                max_variants=(None if int(args.max_variants) <= 0 else int(args.max_variants)),
                max_training_seeds=(None if int(args.max_training_seeds) <= 0 else int(args.max_training_seeds)),
                max_epochs=(None if int(args.max_epochs) <= 0 else int(args.max_epochs)),
                patience=(None if int(args.patience) <= 0 else int(args.patience)),
            )
        elif args.stage == "pmp":
            from .pmp_stage import run_pmp_stage

            run_pmp_stage(
                cfg,
                out_dir=out_dir,
                force=bool(args.force),
                skip_existing=effective_skip_existing,
                retry_errors=bool(args.retry_errors),
                device=str(args.device),
                include_noop=bool(args.include_noop),
                only_clean=bool(args.only_clean),
                max_variants=(None if int(args.max_variants) <= 0 else int(args.max_variants)),
                max_training_seeds=(None if int(args.max_training_seeds) <= 0 else int(args.max_training_seeds)),
                max_epochs=(None if int(args.max_epochs) <= 0 else int(args.max_epochs)),
                patience=(None if int(args.patience) <= 0 else int(args.patience)),
            )
        elif args.stage == "secgfd":
            from .secgfd_stage import run_secgfd_stage

            run_secgfd_stage(
                cfg,
                out_dir=out_dir,
                force=bool(args.force),
                skip_existing=effective_skip_existing,
                retry_errors=bool(args.retry_errors),
                device=str(args.device),
                include_noop=bool(args.include_noop),
                only_clean=bool(args.only_clean),
                max_variants=(None if int(args.max_variants) <= 0 else int(args.max_variants)),
                max_training_seeds=(None if int(args.max_training_seeds) <= 0 else int(args.max_training_seeds)),
                max_epochs=(None if int(args.max_epochs) <= 0 else int(args.max_epochs)),
                patience=(None if int(args.patience) <= 0 else int(args.patience)),
            )
        elif args.stage == "shift":
            from .shift_stage import run_shift_stage

            run_shift_stage(
                cfg,
                out_dir=out_dir,
                force=bool(args.force),
                skip_existing=effective_skip_existing,
                retry_errors=bool(args.retry_errors),
                device=str(args.device),
                include_noop=bool(args.include_noop),
                only_clean=bool(args.only_clean),
                max_variants=(None if int(args.max_variants) <= 0 else int(args.max_variants)),
                max_training_seeds=(None if int(args.max_training_seeds) <= 0 else int(args.max_training_seeds)),
                max_epochs=(None if int(args.max_epochs) <= 0 else int(args.max_epochs)),
                patience=(None if int(args.patience) <= 0 else int(args.patience)),
            )
        elif args.stage == "matrix":
            from .matrix_stage import run_matrix_stage

            run_matrix_stage(
                cfg,
                out_dir=out_dir,
                force=bool(args.force),
                protocol=str(args.protocol),
                skip_existing=effective_skip_existing,
                retry_errors=bool(args.retry_errors),
                device=str(args.device),
                include_noop=bool(args.include_noop),
                only_clean=bool(args.only_clean),
                max_variants=(None if int(args.max_variants) <= 0 else int(args.max_variants)),
                max_training_seeds=(None if int(args.max_training_seeds) <= 0 else int(args.max_training_seeds)),
                max_epochs=(None if int(args.max_epochs) <= 0 else int(args.max_epochs)),
                patience=(None if int(args.patience) <= 0 else int(args.patience)),
            )
        elif args.stage == "plots":
            from .plots_stage import run_plots_stage

            run_plots_stage(
                cfg,
                out_dir=out_dir,
                include_noop=bool(args.include_noop),
                only_clean=bool(args.only_clean),
                max_variants=(None if int(args.max_variants) <= 0 else int(args.max_variants)),
                max_training_seeds=(None if int(args.max_training_seeds) <= 0 else int(args.max_training_seeds)),
            )
        else:
            raise RuntimeError(f"Unknown stage: {args.stage}")
    except Exception as e:
        print(f"[run] ERROR: {e}")
        return 1
    finally:
        dt = time.time() - t0
        print(f"[run] stage={args.stage} done in {dt:.2f}s; out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
