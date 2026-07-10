from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .paths import GraphPath


class CacheError(RuntimeError):
    pass


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    temp.replace(path)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_graph_integrity(path: Path, meta: dict[str, Any]) -> None:
    """Reject a cache file whose recorded size or digest no longer matches."""
    if "graph_bytes" in meta and int(meta["graph_bytes"]) != path.stat().st_size:
        raise CacheError(f"Cached graph size does not match metadata: {path}")
    if "graph_sha256" in meta and str(meta["graph_sha256"]) != _sha256_file(path):
        raise CacheError(f"Cached graph SHA-256 does not match metadata: {path}")


def save_graph(graph_path: GraphPath, graph, meta: dict[str, Any], *, force: bool) -> None:
    """Save a DGL graph and metadata to disk.

    Imports DGL lazily so that `py -m benchmark.run --help` works without ML deps.
    """
    graph_path.dir_path.mkdir(parents=True, exist_ok=True)

    try:
        from dgl.data.utils import load_graphs, save_graphs
    except Exception as e:  # pragma: no cover
        raise CacheError(
            "DGL is required to save graphs. Install dgl (and torch) in your environment."
        ) from e

    if graph_path.graph_bin_path.exists() and not force:
        try:
            existing_meta: dict[str, Any] = {}
            if graph_path.meta_json_path.exists():
                with graph_path.meta_json_path.open("r", encoding="utf-8") as handle:
                    existing_meta = json.load(handle)
            _validate_graph_integrity(graph_path.graph_bin_path, existing_meta)
            existing, _ = load_graphs(str(graph_path.graph_bin_path))
            if len(existing) != 1:
                raise CacheError(f"Expected one cached graph, found {len(existing)}")
            cached = existing[0]
            if int(cached.num_nodes()) != int(graph.num_nodes()) or int(cached.num_edges()) != int(graph.num_edges()):
                raise CacheError("Cached graph dimensions do not match the requested graph.")
            meta_with_integrity = {
                **meta,
                "graph_bytes": graph_path.graph_bin_path.stat().st_size,
                "graph_sha256": _sha256_file(graph_path.graph_bin_path),
            }
            _write_json(graph_path.meta_json_path, meta_with_integrity)
            return
        except Exception:
            # A corrupt or mismatched final file is safe to rebuild because the
            # replacement below is published atomically.
            pass

    temp_graph = graph_path.graph_bin_path.with_suffix(graph_path.graph_bin_path.suffix + ".tmp")
    if temp_graph.exists():
        temp_graph.unlink()
    try:
        save_graphs(str(temp_graph), [graph])
        saved, _ = load_graphs(str(temp_graph))
        if len(saved) != 1:
            raise CacheError(f"Expected one graph after save, found {len(saved)}")
        saved_graph = saved[0]
        if int(saved_graph.num_nodes()) != int(graph.num_nodes()) or int(saved_graph.num_edges()) != int(graph.num_edges()):
            raise CacheError("Saved graph dimensions changed during serialization.")
        meta_with_integrity = {
            **meta,
            "graph_bytes": temp_graph.stat().st_size,
            "graph_sha256": _sha256_file(temp_graph),
        }
        temp_graph.replace(graph_path.graph_bin_path)
        _write_json(graph_path.meta_json_path, meta_with_integrity)
        if graph_path.ref_json_path.exists():
            graph_path.ref_json_path.unlink()
    finally:
        if temp_graph.exists():
            temp_graph.unlink()


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
    if force and graph_path.graph_bin_path.exists():
        graph_path.graph_bin_path.unlink()


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

    meta = {}
    if graph_path.meta_json_path.exists():
        with graph_path.meta_json_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

    if graph_path.graph_bin_path.exists():
        _validate_graph_integrity(graph_path.graph_bin_path, meta)
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

    return g, meta

