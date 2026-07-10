from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from benchmark.cache import CacheError, load_graph, save_graph
from benchmark.paths import base_graph_path


class _FakeGraph:
    def __init__(self, nodes: int, edges: int) -> None:
        self._nodes = nodes
        self._edges = edges

    def num_nodes(self) -> int:
        return self._nodes

    def num_edges(self) -> int:
        return self._edges


def _fake_dgl_modules(*, fail_save: bool = False) -> dict[str, types.ModuleType]:
    dgl = types.ModuleType("dgl")
    data = types.ModuleType("dgl.data")
    utils = types.ModuleType("dgl.data.utils")

    def save_graphs(path: str, graphs: list[_FakeGraph]) -> None:
        target = Path(path)
        target.write_text(
            f"{graphs[0].num_nodes()},{graphs[0].num_edges()}",
            encoding="ascii",
        )
        if fail_save:
            raise RuntimeError("simulated interrupted serialization")

    def load_graphs(path: str):
        raw = Path(path).read_text(encoding="ascii")
        nodes, edges = (int(value) for value in raw.split(","))
        return [_FakeGraph(nodes, edges)], {}

    utils.save_graphs = save_graphs
    utils.load_graphs = load_graphs
    data.utils = utils
    dgl.data = data
    return {"dgl": dgl, "dgl.data": data, "dgl.data.utils": utils}


class CacheTests(unittest.TestCase):
    def test_save_graph_publishes_validated_graph_and_integrity_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = base_graph_path(Path(tmpdir), "yelpchi", "s0")
            with mock.patch.dict(sys.modules, _fake_dgl_modules()):
                save_graph(path, _FakeGraph(10, 20), {"scenario_id": "clean"}, force=False)
                graph, meta = load_graph(path)

            self.assertEqual((graph.num_nodes(), graph.num_edges()), (10, 20))
            self.assertGreater(meta["graph_bytes"], 0)
            self.assertEqual(len(meta["graph_sha256"]), 64)
            self.assertFalse(path.graph_bin_path.with_suffix(".bin.tmp").exists())

    def test_load_graph_rejects_integrity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = base_graph_path(Path(tmpdir), "yelpchi", "s0")
            with mock.patch.dict(sys.modules, _fake_dgl_modules()):
                save_graph(path, _FakeGraph(10, 20), {}, force=False)
                path.graph_bin_path.write_text("10,21", encoding="ascii")
                with self.assertRaisesRegex(CacheError, "does not match metadata"):
                    load_graph(path)

    def test_save_graph_rebuilds_an_integrity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = base_graph_path(Path(tmpdir), "yelpchi", "s0")
            with mock.patch.dict(sys.modules, _fake_dgl_modules()):
                save_graph(path, _FakeGraph(10, 20), {}, force=False)
                path.graph_bin_path.write_text("10,21", encoding="ascii")
                save_graph(path, _FakeGraph(10, 20), {}, force=False)
                graph, _ = load_graph(path)

            self.assertEqual((graph.num_nodes(), graph.num_edges()), (10, 20))

    def test_failed_force_save_preserves_previous_final_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = base_graph_path(Path(tmpdir), "yelpchi", "s0")
            with mock.patch.dict(sys.modules, _fake_dgl_modules()):
                save_graph(path, _FakeGraph(10, 20), {}, force=False)
            original_graph = path.graph_bin_path.read_bytes()
            original_meta = json.loads(path.meta_json_path.read_text(encoding="utf-8"))

            with mock.patch.dict(sys.modules, _fake_dgl_modules(fail_save=True)):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    save_graph(path, _FakeGraph(11, 22), {}, force=True)

            self.assertEqual(path.graph_bin_path.read_bytes(), original_graph)
            self.assertEqual(
                json.loads(path.meta_json_path.read_text(encoding="utf-8")),
                original_meta,
            )
            self.assertFalse(path.graph_bin_path.with_suffix(".bin.tmp").exists())


if __name__ == "__main__":
    unittest.main()
