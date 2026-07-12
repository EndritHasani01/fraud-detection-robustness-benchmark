from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    pass


_SUPPORTED_SCENARIO_METHODS = {
    "rewire_edge_dst_to_opposite_label",
    "rewire_edge_dst_to_feature_pseudo_opposite_label",
    "replace_fraud_features_with_normal_features",
    "add_relation_camouflage_edges",
    "add_random_edges",
}

_ORACLE_MODE_ALIASES = {
    "oracle": "oracle",
    "non_oracle": "non_oracle",
    "nonoracle": "non_oracle",
    "non-oracle": "non_oracle",
}

_GRAPH_VIEW_MODE_ALIASES = {
    "canonical": "canonical",
    "homogeneous": "canonical",
    "heterograph_aware_generation": "heterograph_aware_generation",
    "heterograph-aware-generation": "heterograph_aware_generation",
    "heterograph": "heterograph_aware_generation",
}


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
    if len(out) != len(set(out)):
        raise ConfigError(f"seeds.{key} must not contain duplicate values")
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


def _require_non_empty_string(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path} must be a non-empty string")
    return str(value).strip()


def _require_bool(value: Any, *, path: str) -> bool:
    if isinstance(value, bool):
        return bool(value)
    raise ConfigError(f"{path} must be a boolean")


def _require_non_negative_number(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a number")
    value = float(value)
    if value < 0.0:
        raise ConfigError(f"{path} must be >= 0")
    return value


def _normalize_oracle_mode(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path} must be 'oracle' or 'non_oracle'")
    normalized = _ORACLE_MODE_ALIASES.get(str(value).strip().lower())
    if normalized is None:
        raise ConfigError(f"{path} must be 'oracle' or 'non_oracle'")
    return normalized


def scenario_oracle_labels(scenario_cfg: dict[str, Any]) -> bool:
    oracle_mode = scenario_cfg.get("oracle_mode")
    if oracle_mode is not None:
        return _normalize_oracle_mode(oracle_mode, path="scenario.oracle_mode") == "oracle"
    return bool(scenario_cfg.get("oracle_labels", False))


def scenario_graph_view_mode(scenario_cfg: dict[str, Any]) -> str:
    raw_value = scenario_cfg.get("graph_view_mode", "canonical")
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ConfigError("scenario.graph_view_mode must be 'canonical' or 'heterograph_aware_generation'")
    normalized = _GRAPH_VIEW_MODE_ALIASES.get(str(raw_value).strip().lower())
    if normalized is None:
        raise ConfigError("scenario.graph_view_mode must be 'canonical' or 'heterograph_aware_generation'")
    return normalized


def should_export_variant_audit(cfg: dict[str, Any]) -> bool:
    eval_cfg = cfg.get("evaluation")
    if not isinstance(eval_cfg, dict):
        return True
    return bool(eval_cfg.get("export_variant_audit", True))


def _validate_relation_filter(value: Any, *, path: str) -> None:
    if isinstance(value, str):
        if not value.strip():
            raise ConfigError(f"{path} must not be empty")
        return
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{path} must be a non-empty string or list of non-empty strings")
    for idx, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"{path}[{idx}] must be a non-empty string")


def _validate_string_list(value: Any, *, path: str) -> None:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{path} must be a non-empty list of strings")
    for idx, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"{path}[{idx}] must be a non-empty string")


def _validate_scenario_cfg(scenario_cfg: dict[str, Any], *, idx: int) -> None:
    prefix = f"cfg['scenarios'][{idx}]"
    _require_non_empty_string(scenario_cfg.get("scenario_id"), path=f"{prefix}.scenario_id")
    _require_non_empty_string(scenario_cfg.get("family"), path=f"{prefix}.family")
    _require_non_empty_string(scenario_cfg.get("severity_param"), path=f"{prefix}.severity_param")

    method = _require_non_empty_string(scenario_cfg.get("method"), path=f"{prefix}.method")
    if method not in _SUPPORTED_SCENARIO_METHODS:
        supported = ", ".join(sorted(_SUPPORTED_SCENARIO_METHODS))
        raise ConfigError(f"{prefix}.method must be one of: {supported}")

    severity_values = scenario_cfg.get("severity_values")
    if not isinstance(severity_values, list) or not severity_values:
        raise ConfigError(f"{prefix}.severity_values must be a non-empty list of numbers")
    for sev_idx, raw in enumerate(severity_values):
        _require_non_negative_number(raw, path=f"{prefix}.severity_values[{sev_idx}]")

    if "scenario_group" in scenario_cfg:
        _require_non_empty_string(scenario_cfg.get("scenario_group"), path=f"{prefix}.scenario_group")
    if "oracle_mode" in scenario_cfg:
        normalized_mode = _normalize_oracle_mode(scenario_cfg.get("oracle_mode"), path=f"{prefix}.oracle_mode")
        if "oracle_labels" in scenario_cfg:
            oracle_labels = _require_bool(scenario_cfg.get("oracle_labels"), path=f"{prefix}.oracle_labels")
            if oracle_labels != (normalized_mode == "oracle"):
                raise ConfigError(f"{prefix}.oracle_mode conflicts with {prefix}.oracle_labels")
    elif "oracle_labels" in scenario_cfg:
        _require_bool(scenario_cfg.get("oracle_labels"), path=f"{prefix}.oracle_labels")

    if "graph_view_mode" in scenario_cfg:
        scenario_graph_view_mode({"graph_view_mode": scenario_cfg.get("graph_view_mode")})
    if "relation_filter" in scenario_cfg:
        _validate_relation_filter(scenario_cfg.get("relation_filter"), path=f"{prefix}.relation_filter")
    if "camouflage_edges_per_node" in scenario_cfg:
        value = scenario_cfg.get("camouflage_edges_per_node")
        if isinstance(value, bool) or not isinstance(value, int) or int(value) <= 0:
            raise ConfigError(f"{prefix}.camouflage_edges_per_node must be an integer >= 1")
    if "camouflage_edge_degree_ratio" in scenario_cfg:
        ratio = _require_non_negative_number(
            scenario_cfg.get("camouflage_edge_degree_ratio"),
            path=f"{prefix}.camouflage_edge_degree_ratio",
        )
        if ratio <= 0.0:
            raise ConfigError(f"{prefix}.camouflage_edge_degree_ratio must be > 0")
        if "camouflage_edges_per_node" in scenario_cfg:
            raise ConfigError(
                f"{prefix} must set only one of camouflage_edges_per_node or camouflage_edge_degree_ratio"
            )
    for key in ("camouflage_min_edges_per_node", "camouflage_max_edges_per_node"):
        if key in scenario_cfg:
            value = scenario_cfg.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or int(value) <= 0:
                raise ConfigError(f"{prefix}.{key} must be an integer >= 1")
    min_edges = int(scenario_cfg.get("camouflage_min_edges_per_node", 1) or 1)
    max_edges = scenario_cfg.get("camouflage_max_edges_per_node")
    if max_edges is not None and int(max_edges) < min_edges:
        raise ConfigError(
            f"{prefix}.camouflage_max_edges_per_node must be >= camouflage_min_edges_per_node"
        )
    if "remove_suspicious_ratio" in scenario_cfg:
        ratio = _require_non_negative_number(
            scenario_cfg.get("remove_suspicious_ratio"),
            path=f"{prefix}.remove_suspicious_ratio",
        )
        if ratio > 1.0:
            raise ConfigError(f"{prefix}.remove_suspicious_ratio must be <= 1")
    if "gamma" in scenario_cfg:
        gamma = _require_non_negative_number(scenario_cfg.get("gamma"), path=f"{prefix}.gamma")
        if gamma > 1.0:
            raise ConfigError(f"{prefix}.gamma must be <= 1")
    if "undirected" in scenario_cfg:
        _require_bool(scenario_cfg.get("undirected"), path=f"{prefix}.undirected")
    if "fixed_feature_partition_across_severity" in scenario_cfg:
        _require_bool(
            scenario_cfg.get("fixed_feature_partition_across_severity"),
            path=f"{prefix}.fixed_feature_partition_across_severity",
        )
    if "relation_allocation" in scenario_cfg:
        relation_allocation = str(scenario_cfg.get("relation_allocation", "")).strip().lower()
        if relation_allocation not in {"uniform", "proportional"}:
            raise ConfigError(f"{prefix}.relation_allocation must be 'uniform' or 'proportional'")

    for key in ("allow_self_loops", "reject_existing", "reject_duplicates"):
        if key in scenario_cfg:
            _require_bool(scenario_cfg.get(key), path=f"{prefix}.{key}")
    if "max_attempt_multiplier" in scenario_cfg:
        value = scenario_cfg.get("max_attempt_multiplier")
        if isinstance(value, bool) or not isinstance(value, int) or int(value) <= 0:
            raise ConfigError(f"{prefix}.max_attempt_multiplier must be an integer >= 1")


def _validate_evaluation_cfg(eval_cfg: dict[str, Any]) -> None:
    metrics = eval_cfg.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ConfigError("evaluation.metrics must be a non-empty list")
    for idx, metric in enumerate(metrics):
        if not isinstance(metric, str) or not metric.strip():
            raise ConfigError(f"evaluation.metrics[{idx}] must be a non-empty string")

    if "export_variant_audit" in eval_cfg:
        _require_bool(eval_cfg.get("export_variant_audit"), path="evaluation.export_variant_audit")
    if "audit_metrics" in eval_cfg:
        _validate_string_list(eval_cfg.get("audit_metrics"), path="evaluation.audit_metrics")


def _validate_datasets(datasets: list[Any]) -> None:
    dataset_ids: list[str] = []
    for idx, dataset_cfg in enumerate(datasets):
        prefix = f"cfg['datasets'][{idx}]"
        if not isinstance(dataset_cfg, dict):
            raise ConfigError(f"{prefix} must be an object")
        dataset_ids.append(_require_non_empty_string(dataset_cfg.get("dataset_id"), path=f"{prefix}.dataset_id"))
        _require_non_empty_string(dataset_cfg.get("source_name"), path=f"{prefix}.source_name")
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ConfigError("cfg['datasets'] must use unique dataset_id values")


def _validate_data_splits(data_splits: list[Any]) -> None:
    split_ids: list[str] = []
    split_seeds: list[int] = []
    for idx, split_cfg in enumerate(data_splits):
        prefix = f"cfg['data_splits'][{idx}]"
        if not isinstance(split_cfg, dict):
            raise ConfigError(f"{prefix} must be an object")
        split_ids.append(_require_non_empty_string(split_cfg.get("split_id"), path=f"{prefix}.split_id"))

        split_seed = split_cfg.get("split_seed")
        if isinstance(split_seed, bool) or not isinstance(split_seed, int):
            raise ConfigError(f"{prefix}.split_seed must be an integer")
        split_seeds.append(int(split_seed))

        train_size = _require_non_negative_number(split_cfg.get("train_size"), path=f"{prefix}.train_size")
        val_size = _require_non_negative_number(split_cfg.get("val_size"), path=f"{prefix}.val_size")
        if train_size <= 0.0 or val_size <= 0.0:
            raise ConfigError(f"{prefix}.train_size and {prefix}.val_size must both be > 0")
        if train_size + val_size >= 1.0:
            raise ConfigError(f"{prefix}.train_size + {prefix}.val_size must be < 1")

    if len(split_ids) != len(set(split_ids)):
        raise ConfigError("cfg['data_splits'] must use unique split_id values")
    if len(split_seeds) != len(set(split_seeds)):
        raise ConfigError("cfg['data_splits'] must use unique split_seed values")


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

    _require_non_empty_string(cfg.get("experiment_name"), path="cfg['experiment_name']")
    _validate_datasets(cfg["datasets"])
    _validate_data_splits(cfg["data_splits"])

    graph_rep = cfg["graph_representation"]
    if not isinstance(graph_rep, dict):
        raise ConfigError("cfg['graph_representation'] must be an object")
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

    model_ids: list[str] = []
    for idx, model_cfg in enumerate(cfg["models"]):
        if not isinstance(model_cfg, dict):
            raise ConfigError(f"cfg['models'][{idx}] must be an object")
        model_ids.append(_require_non_empty_string(model_cfg.get("model_id"), path=f"cfg['models'][{idx}].model_id"))
        _validate_model_hparams(model_cfg)
    if len(model_ids) != len(set(model_ids)):
        raise ConfigError("cfg['models'] must use unique model_id values")

    scenario_ids: list[str] = []
    for idx, scenario_cfg in enumerate(cfg["scenarios"]):
        if not isinstance(scenario_cfg, dict):
            raise ConfigError(f"cfg['scenarios'][{idx}] must be an object")
        _validate_scenario_cfg(scenario_cfg, idx=idx)
        scenario_ids.append(str(scenario_cfg["scenario_id"]).strip())
        severities = [float(value) for value in scenario_cfg["severity_values"]]
        if len(severities) != len(set(severities)):
            raise ConfigError(f"cfg['scenarios'][{idx}].severity_values must not contain duplicates")
        if 0.0 not in severities:
            raise ConfigError(f"cfg['scenarios'][{idx}].severity_values must include 0.0")
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ConfigError("cfg['scenarios'] must use unique scenario_id values")

    eval_cfg = cfg["evaluation"]
    if not isinstance(eval_cfg, dict):
        raise ConfigError("cfg['evaluation'] must be an object")
    _validate_evaluation_cfg(eval_cfg)


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
