from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExperimentPaths:
    out_dir: Path
    config_copy_path: Path
    results_csv_path: Path
    variants_csv_path: Path
    graphs_dir: Path


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"Config file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in config file: {path}") from e


def validate_config(cfg: dict[str, Any]) -> None:
    required_top = [
        "experiment_name",
        "datasets",
        "data_splits",
        "graph_representation",
        "models",
        "scenarios",
        "seeds",
        "evaluation",
    ]
    missing = [k for k in required_top if k not in cfg]
    if missing:
        raise ConfigError(f"Missing required config keys: {missing}")

    if not isinstance(cfg["datasets"], list) or not cfg["datasets"]:
        raise ConfigError("cfg['datasets'] must be a non-empty list")
    if not isinstance(cfg["data_splits"], list) or not cfg["data_splits"]:
        raise ConfigError("cfg['data_splits'] must be a non-empty list")
    if not isinstance(cfg["scenarios"], list) or not cfg["scenarios"]:
        raise ConfigError("cfg['scenarios'] must be a non-empty list")

    graph_rep = cfg["graph_representation"]
    if graph_rep.get("canonical_view") not in {"homogeneous", "heterograph"}:
        raise ConfigError("graph_representation.canonical_view must be 'homogeneous' or 'heterograph'")

    seeds = cfg["seeds"]
    tr_seeds = seeds.get("training_seeds")
    if not isinstance(tr_seeds, list) or not tr_seeds:
        raise ConfigError("seeds.training_seeds must be a non-empty list")

    eval_cfg = cfg["evaluation"]
    metrics = eval_cfg.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ConfigError("evaluation.metrics must be a non-empty list")


def init_paths(out_dir: Path) -> ExperimentPaths:
    out_dir = out_dir.resolve()
    graphs_dir = out_dir / "graphs"
    return ExperimentPaths(
        out_dir=out_dir,
        config_copy_path=out_dir / "config.json",
        results_csv_path=out_dir / "results.csv",
        variants_csv_path=out_dir / "graph_variants.csv",
        graphs_dir=graphs_dir,
    )


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")

