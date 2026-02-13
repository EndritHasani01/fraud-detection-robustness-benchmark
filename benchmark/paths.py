from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _sev_dirname(severity: float) -> str:
    # Stable directory name for floats, avoids issues like 0.30000000004.
    # 0.1 -> "0.10", 0.05 -> "0.05"
    return f"{severity:.2f}"


@dataclass(frozen=True)
class GraphPath:
    dir_path: Path
    graph_bin_path: Path
    meta_json_path: Path
    ref_json_path: Path


def base_graph_path(graphs_dir: Path, dataset_id: str, split_id: str) -> GraphPath:
    d = graphs_dir / "base" / dataset_id / split_id
    return GraphPath(
        dir_path=d,
        graph_bin_path=d / "graph.bin",
        meta_json_path=d / "meta.json",
        ref_json_path=d / "graph_ref.json",
    )


def variant_graph_path(
    graphs_dir: Path,
    dataset_id: str,
    split_id: str,
    scenario_id: str,
    severity: float,
    graph_seed: int,
) -> GraphPath:
    sev = _sev_dirname(float(severity))
    d = graphs_dir / "variants" / dataset_id / split_id / scenario_id / sev / f"seed_{graph_seed}"
    return GraphPath(
        dir_path=d,
        graph_bin_path=d / "graph.bin",
        meta_json_path=d / "meta.json",
        ref_json_path=d / "graph_ref.json",
    )

