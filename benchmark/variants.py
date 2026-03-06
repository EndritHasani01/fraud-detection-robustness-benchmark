from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class VariantRow:
    experiment_name: str
    dataset_id: str
    split_id: str
    graph_seed: int
    scenario_id: str
    severity: float
    oracle_labels: bool
    scenario_applied: bool
    base_graph_path: str
    graph_path: str
    n_nodes: int
    n_edges: int
    mean_in_degree: float
    median_in_degree: float
    mean_out_degree: float
    median_out_degree: float
    heterophily_ratio: float | None
    pos_rate: float | None


def parse_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return bool(x)
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return int(default)


def safe_float(x: Any, default: float | None = None) -> float | None:
    try:
        return float(x)
    except Exception:
        return default


def read_variants_csv(path: Path) -> list[VariantRow]:
    if not path.exists():
        raise FileNotFoundError(f"graph_variants.csv not found: {path}")

    rows: list[VariantRow] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                VariantRow(
                    experiment_name=str(r.get("experiment_name", "")),
                    dataset_id=str(r.get("dataset_id", "")),
                    split_id=str(r.get("split_id", "")),
                    graph_seed=safe_int(r.get("graph_seed", 0)),
                    scenario_id=str(r.get("scenario_id", "")),
                    severity=float(r.get("severity", 0.0) or 0.0),
                    oracle_labels=parse_bool(r.get("oracle_labels", False)),
                    scenario_applied=parse_bool(r.get("scenario_applied", True)),
                    base_graph_path=str(r.get("base_graph_path", "")),
                    graph_path=str(r.get("graph_path", "")),
                    n_nodes=safe_int(r.get("n_nodes", 0)),
                    n_edges=safe_int(r.get("n_edges", 0)),
                    mean_in_degree=float(r.get("mean_in_degree", 0.0) or 0.0),
                    median_in_degree=float(r.get("median_in_degree", 0.0) or 0.0),
                    mean_out_degree=float(r.get("mean_out_degree", 0.0) or 0.0),
                    median_out_degree=float(r.get("median_out_degree", 0.0) or 0.0),
                    heterophily_ratio=safe_float(r.get("heterophily_ratio", ""), None),
                    pos_rate=safe_float(r.get("pos_rate", ""), None),
                )
            )
    return rows


def require_variants_csv_rows(path: Path) -> list[VariantRow]:
    rows = read_variants_csv(path)
    if rows:
        return rows
    raise RuntimeError(
        f"No rows found in {path}. Re-run `py -m benchmark.run --stage graphs ...`; "
        "a previous graphs run may have been interrupted."
    )


def filter_variants(
    variants: Sequence[Any],
    *,
    include_noop: bool,
    only_clean: bool,
    max_variants: int | None,
) -> list[Any]:
    filtered: list[Any] = []
    for variant in variants:
        scenario_id = str(getattr(variant, "scenario_id"))
        scenario_applied = bool(getattr(variant, "scenario_applied"))
        if only_clean and scenario_id != "clean":
            continue
        if (not include_noop) and (not scenario_applied) and scenario_id != "clean":
            continue
        filtered.append(variant)
    if max_variants is not None:
        filtered = filtered[: int(max_variants)]
    return filtered


def load_graph_bin(path: Path):
    try:
        from dgl.data.utils import load_graphs
    except Exception as e:  # pragma: no cover
        raise RuntimeError("DGL is required to load cached graphs.") from e

    graphs, _ = load_graphs(str(path))
    if not graphs:
        raise RuntimeError(f"No graphs found in file: {path}")
    return graphs[0]


def set_seeds(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    try:
        import dgl

        dgl.seed(int(seed))
        dgl.random.seed(int(seed))
    except Exception:
        pass


def class_weights_from_train_labels(y_train):
    import torch

    y_train = y_train.to(torch.int64)
    n_pos = int((y_train == 1).sum().item())
    n_neg = int((y_train == 0).sum().item())
    if n_pos <= 0 or n_neg <= 0:
        return None
    w0 = 1.0
    w1 = float(n_neg) / float(n_pos)
    return torch.tensor([w0, w1], dtype=torch.float32)
