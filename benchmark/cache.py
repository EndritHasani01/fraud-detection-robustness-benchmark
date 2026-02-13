from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import GraphPath


class CacheError(RuntimeError):
    pass


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def save_graph(graph_path: GraphPath, graph, meta: dict[str, Any], *, force: bool) -> None:
    """Save a DGL graph and metadata to disk.

    Imports DGL lazily so that `py -m benchmark.run --help` works without ML deps.
    """
    if graph_path.graph_bin_path.exists() and not force:
        return

    graph_path.dir_path.mkdir(parents=True, exist_ok=True)
    _write_json(graph_path.meta_json_path, meta)

    try:
        from dgl.data.utils import save_graphs
    except Exception as e:  # pragma: no cover
        raise CacheError(
            "DGL is required to save graphs. Install dgl (and torch) in your environment."
        ) from e

    save_graphs(str(graph_path.graph_bin_path), [graph])


def save_graph_ref(graph_path: GraphPath, target_graph_bin: Path, meta: dict[str, Any], *, force: bool) -> None:
    """Create a lightweight variant that references another graph file.

    This avoids duplicating large graph binaries when the variant is a no-op
    (e.g., while stress tests are not implemented yet).
    """
    if graph_path.ref_json_path.exists() and not force:
        return

    graph_path.dir_path.mkdir(parents=True, exist_ok=True)
    _write_json(graph_path.meta_json_path, meta)
    _write_json(graph_path.ref_json_path, {"ref_graph_bin": str(target_graph_bin.resolve())})


def load_graph(graph_path: GraphPath):
    """Load a DGL graph from a cached path or a reference.

    Returns: (graph, meta_dict)
    """
    try:
        from dgl.data.utils import load_graphs
    except Exception as e:  # pragma: no cover
        raise CacheError(
            "DGL is required to load graphs. Install dgl (and torch) in your environment."
        ) from e

    if graph_path.graph_bin_path.exists():
        graphs, _ = load_graphs(str(graph_path.graph_bin_path))
        g = graphs[0]
    elif graph_path.ref_json_path.exists():
        with graph_path.ref_json_path.open("r", encoding="utf-8") as f:
            ref = json.load(f)
        ref_bin = Path(ref["ref_graph_bin"])
        graphs, _ = load_graphs(str(ref_bin))
        g = graphs[0]
    else:
        raise CacheError(f"No cached graph found at: {graph_path.dir_path}")

    meta = {}
    if graph_path.meta_json_path.exists():
        with graph_path.meta_json_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    return g, meta

