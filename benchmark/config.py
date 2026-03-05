from __future__ import annotations

import json
import sys
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


def _validated_seed_list(
    seeds_cfg: dict[str, Any],
    *,
    key: str,
    required: bool,
) -> list[int]:
    value = seeds_cfg.get(key)
    if value is None:
        if required:
            raise ConfigError(f"seeds.{key} must be a non-empty list of integers")
        return []
    if not isinstance(value, list) or not value:
        raise ConfigError(f"seeds.{key} must be a non-empty list of integers")

    out: list[int] = []
    for idx, raw in enumerate(value):
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ConfigError(f"seeds.{key}[{idx}] must be an integer")
        out.append(int(raw))
    return out


def get_training_seeds(cfg: dict[str, Any]) -> list[int]:
    seeds_cfg = cfg.get("seeds")
    if not isinstance(seeds_cfg, dict):
        raise ConfigError("cfg['seeds'] must be an object")
    return _validated_seed_list(seeds_cfg, key="training_seeds", required=True)


def get_graph_seeds(cfg: dict[str, Any]) -> list[int]:
    seeds_cfg = cfg.get("seeds")
    if not isinstance(seeds_cfg, dict):
        raise ConfigError("cfg['seeds'] must be an object")
    graph_seeds = _validated_seed_list(seeds_cfg, key="graph_seeds", required=False)
    if graph_seeds:
        return graph_seeds
    return get_training_seeds(cfg)


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
    if not isinstance(seeds, dict):
        raise ConfigError("cfg['seeds'] must be an object")

    if "training_seeds" not in seeds and "graph_seeds" not in seeds:
        print(
            "[config] WARNING: config is missing both seeds.training_seeds and seeds.graph_seeds.",
            file=sys.stderr,
        )
    _validated_seed_list(seeds, key="training_seeds", required=True)
    _validated_seed_list(seeds, key="graph_seeds", required=False)

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
