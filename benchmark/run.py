from __future__ import annotations

import argparse
import time
from pathlib import Path

from .config import ConfigError, init_paths, load_json, validate_config, write_json
from .paths import base_graph_path, variant_graph_path
from .results import RESULTS_COLUMNS, VARIANTS_COLUMNS, append_csv_row, ensure_csv_header


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

    # Ensure CSVs exist with correct schema.
    ensure_csv_header(paths.results_csv_path, RESULTS_COLUMNS)
    ensure_csv_header(paths.variants_csv_path, VARIANTS_COLUMNS)

    experiment_name = cfg["experiment_name"]
    graph_seeds = [int(s) for s in cfg["seeds"]["training_seeds"]]

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
                        )
                        var_p = variant_graph_path(
                            paths.graphs_dir, dataset_id, split_id, scenario_id, float(severity), int(graph_seed)
                        )

                        # Scenario application is currently a no-op until TODO-02.
                        g_var, applied = apply_scenario(g_base, spec)

                        var_meta = {
                            **base_meta,
                            "base_graph_bin": str(base_p.graph_bin_path.resolve()),
                            "scenario_id": scenario_id,
                            "severity": float(severity),
                            "graph_seed": int(graph_seed),
                            "oracle_labels": oracle_labels,
                            "scenario_applied": bool(applied),
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
        choices=["graphs"],
        help="Pipeline stage to run. 'graphs' builds splits and caches graph variants.",
    )
    p.add_argument("--force", action="store_true", help="Overwrite cached outputs if they exist.")
    p.add_argument("--force-reload", action="store_true", help="Force DGL dataset reload/redownload.")
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

    t0 = time.time()
    try:
        if args.stage == "graphs":
            _graphs_only(cfg, out_dir=out_dir, force=bool(args.force), force_reload=bool(args.force_reload))
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

