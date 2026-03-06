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
    variant_audit_csv_path: Path
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


def _validate_model_hparams(model_cfg: dict[str, Any]) -> None:
    model_id = str(model_cfg.get("model_id", ""))
    if model_id != "secgfd":
        return

    hparams = model_cfg.get("hparams")
    if hparams is None:
        return
    if not isinstance(hparams, dict):
        raise ConfigError("secgfd.hparams must be an object")

    int_keys = {"hid_dim", "order_d", "high_order"}
    float_keys = {"lemda", "lr", "weight_decay"}
    allowed_keys = int_keys | float_keys

    for key, value in hparams.items():
        if key not in allowed_keys:
            raise ConfigError(f"secgfd.hparams.{key} is not supported")
        if key in int_keys:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"secgfd.hparams.{key} must be an integer")
            if int(value) <= 0:
                raise ConfigError(f"secgfd.hparams.{key} must be > 0")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"secgfd.hparams.{key} must be a number")
            if key == "lr" and float(value) <= 0.0:
                raise ConfigError("secgfd.hparams.lr must be > 0")
            if key in {"lemda", "weight_decay"} and float(value) < 0.0:
                raise ConfigError(f"secgfd.hparams.{key} must be >= 0")


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
    if not isinstance(cfg["models"], list) or not cfg["models"]:
        raise ConfigError("cfg['models'] must be a non-empty list")
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

    for idx, model_cfg in enumerate(cfg["models"]):
        if not isinstance(model_cfg, dict):
            raise ConfigError(f"cfg['models'][{idx}] must be an object")
        if not str(model_cfg.get("model_id", "")).strip():
            raise ConfigError(f"cfg['models'][{idx}].model_id must be a non-empty string")
        _validate_model_hparams(model_cfg)

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
        variant_audit_csv_path=out_dir / "variant_audit.csv",
        graphs_dir=graphs_dir,
    )


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
