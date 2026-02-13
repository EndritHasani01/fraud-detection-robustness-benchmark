from __future__ import annotations

import csv
from pathlib import Path


RESULTS_COLUMNS = [
    # identifiers
    "experiment_name",
    "dataset_id",
    "split_id",
    "graph_seed",
    "training_seed",
    "scenario_id",
    "severity",
    "model_id",
    # metrics (may be empty until models are integrated)
    "roc_auc",
    "average_precision",
    "f1_macro",
    "threshold",
    "duration_sec",
    # graph stats (always available)
    "n_nodes",
    "n_edges",
    "mean_in_degree",
    "median_in_degree",
    "mean_out_degree",
    "median_out_degree",
    "heterophily_ratio",
    "pos_rate",
    # paths and misc
    "base_graph_path",
    "graph_path",
    "status",
    "error",
]


VARIANTS_COLUMNS = [
    "experiment_name",
    "dataset_id",
    "split_id",
    "graph_seed",
    "scenario_id",
    "severity",
    "oracle_labels",
    "scenario_applied",
    "base_graph_path",
    "graph_path",
    "n_nodes",
    "n_edges",
    "mean_in_degree",
    "median_in_degree",
    "mean_out_degree",
    "median_out_degree",
    "heterophily_ratio",
    "pos_rate",
]


def ensure_csv_header(path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)


def append_csv_row(path: Path, columns: list[str], row: dict) -> None:
    ensure_csv_header(path, columns)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writerow({k: row.get(k, "") for k in columns})

