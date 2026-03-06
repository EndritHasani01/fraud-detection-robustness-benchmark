from __future__ import annotations

import sys
import types
import unittest
from typing import Any
from unittest import mock

import torch

from benchmark.scenarios import ScenarioSpec, apply_scenario


def _clone_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.clone()
        else:
            out[key] = value
    return out


class FakeGraph:
    def __init__(self, edges, *, num_nodes: int, ndata: dict[str, Any] | None = None, edata: dict[str, Any] | None = None):
        src, dst = edges
        self._src = src.clone() if isinstance(src, torch.Tensor) else torch.tensor(src, dtype=torch.int64)
        self._dst = dst.clone() if isinstance(dst, torch.Tensor) else torch.tensor(dst, dtype=torch.int64)
        self._num_nodes = int(num_nodes)
        self.ndata = _clone_mapping(ndata or {})
        self.edata = _clone_mapping(edata or {})

    def edges(self):
        return self._src.clone(), self._dst.clone()

    def num_nodes(self):
        return int(self._num_nodes)

    def num_edges(self):
        return int(self._src.shape[0])

    def clone(self):
        return FakeGraph((self._src, self._dst), num_nodes=self._num_nodes, ndata=self.ndata, edata=self.edata)


class FakeHeteroGraph:
    def __init__(self, data_dict, *, num_nodes_dict: dict[str, int], ndata: dict[str, Any] | None = None):
        self._data = {
            key: (
                src.clone() if isinstance(src, torch.Tensor) else torch.tensor(src, dtype=torch.int64),
                dst.clone() if isinstance(dst, torch.Tensor) else torch.tensor(dst, dtype=torch.int64),
            )
            for key, (src, dst) in data_dict.items()
        }
        self._num_nodes_dict = {str(key): int(value) for key, value in num_nodes_dict.items()}
        self.ntypes = list(self._num_nodes_dict)
        self.canonical_etypes = list(self._data)
        self.ndata = _clone_mapping(ndata or {})
        self.edata: dict[str, Any] = {}

    def edges(self, etype=None):
        if etype is None:
            if len(self.canonical_etypes) != 1:
                raise TypeError("etype required for multi-relation graph")
            etype = self.canonical_etypes[0]
        src, dst = self._data[etype]
        return src.clone(), dst.clone()

    def num_nodes(self, ntype=None):
        if ntype is None:
            if len(self._num_nodes_dict) != 1:
                raise TypeError("ntype required")
            return next(iter(self._num_nodes_dict.values()))
        return self._num_nodes_dict[str(ntype)]

    def num_edges(self, etype=None):
        if etype is None:
            return sum(int(src.shape[0]) for src, _dst in self._data.values())
        return int(self._data[etype][0].shape[0])

    def clone(self):
        return FakeHeteroGraph(self._data, num_nodes_dict=self._num_nodes_dict, ndata=self.ndata)


def _fake_dgl_module():
    mod = types.ModuleType("dgl")
    mod.graph = lambda edges, num_nodes: FakeGraph(edges, num_nodes=num_nodes)
    mod.heterograph = lambda data_dict, num_nodes_dict: FakeHeteroGraph(data_dict, num_nodes_dict=num_nodes_dict)
    return mod


def _base_homo_graph() -> FakeGraph:
    return FakeGraph(
        (torch.tensor([0, 1, 2, 3], dtype=torch.int64), torch.tensor([1, 0, 3, 2], dtype=torch.int64)),
        num_nodes=4,
        ndata={
            "label": torch.tensor([1, 1, 0, 0], dtype=torch.int64),
            "feature": torch.tensor(
                [
                    [10.0, 10.0],
                    [9.5, 9.5],
                    [0.0, 0.0],
                    [0.5, 0.5],
                ],
                dtype=torch.float32,
            ),
            "train_mask": torch.tensor([True, True, False, False]),
            "val_mask": torch.tensor([False, False, True, False]),
            "test_mask": torch.tensor([False, False, False, True]),
        },
    )


def _hetero_graph() -> FakeHeteroGraph:
    return FakeHeteroGraph(
        {
            ("node", "r1", "node"): (
                torch.tensor([0, 1, 2], dtype=torch.int64),
                torch.tensor([1, 0, 3], dtype=torch.int64),
            ),
            ("node", "r2", "node"): (
                torch.tensor([0], dtype=torch.int64),
                torch.tensor([1], dtype=torch.int64),
            ),
        },
        num_nodes_dict={"node": 4},
        ndata={
            "label": torch.tensor([1, 1, 0, 0], dtype=torch.int64),
            "feature": torch.tensor(
                [
                    [10.0, 10.0],
                    [9.0, 9.0],
                    [0.0, 0.0],
                    [1.0, 1.0],
                ],
                dtype=torch.float32,
            ),
            "train_mask": torch.tensor([True, True, False, False]),
            "val_mask": torch.tensor([False, False, True, False]),
            "test_mask": torch.tensor([False, False, False, True]),
        },
    )


class ScenarioTests(unittest.TestCase):
    def test_oracle_rewire_increases_heterophily(self) -> None:
        spec = ScenarioSpec(
            scenario_id="heterophily_rewire_oracle",
            family="heterophily",
            method="rewire_edge_dst_to_opposite_label",
            oracle_labels=True,
            severity_param="p_rewire",
            severity=0.5,
            graph_seed=7,
            params={},
        )

        with mock.patch.dict(sys.modules, {"dgl": _fake_dgl_module()}):
            g2, applied, info = apply_scenario(_base_homo_graph(), spec)

        self.assertTrue(applied)
        self.assertGreater(info["n_rewired_edges_actual"], 0)
        self.assertGreater(info["heterophily_ratio_after"], info["heterophily_ratio_before"])
        self.assertEqual(g2.num_nodes(), 4)

    def test_feature_camouflage_changes_only_fraud_nodes(self) -> None:
        spec = ScenarioSpec(
            scenario_id="camouflage_feature_oracle",
            family="camouflage",
            method="replace_fraud_features_with_normal_features",
            oracle_labels=True,
            severity_param="p_cam_feat",
            severity=0.5,
            graph_seed=11,
            params={"gamma": 1.0},
        )

        base = _base_homo_graph()
        with mock.patch.dict(sys.modules, {"dgl": _fake_dgl_module()}):
            g2, applied, info = apply_scenario(base, spec)

        self.assertTrue(applied)
        changed = torch.any(g2.ndata["feature"] != base.ndata["feature"], dim=1)
        changed_idx = torch.nonzero(changed, as_tuple=True)[0].tolist()
        self.assertEqual(len(changed_idx), 1)
        self.assertEqual(changed_idx[0], 0)
        self.assertGreater(info["mean_l2_delta"], 0.0)
        self.assertEqual(info["n_camouflaged_nodes_actual"], 1)

    def test_nonoracle_rewire_is_deterministic_for_same_seed(self) -> None:
        base = FakeGraph(
            (
                torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64),
                torch.tensor([1, 0, 3, 2, 5, 4], dtype=torch.int64),
            ),
            num_nodes=6,
            ndata={
                "label": torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.int64),
                "feature": torch.tensor(
                    [
                        [8.0, 8.0],
                        [7.0, 7.5],
                        [6.5, 7.0],
                        [0.0, 0.0],
                        [0.5, 0.2],
                        [1.0, 0.4],
                    ],
                    dtype=torch.float32,
                ),
                "train_mask": torch.tensor([True, True, False, False, False, False]),
                "val_mask": torch.tensor([False, False, True, False, False, False]),
                "test_mask": torch.tensor([False, False, False, True, True, True]),
            },
        )
        spec_same = ScenarioSpec(
            scenario_id="heterophily_rewire_nonoracle",
            family="heterophily",
            method="rewire_edge_dst_to_feature_pseudo_opposite_label",
            oracle_labels=False,
            severity_param="p_rewire",
            severity=1.0,
            graph_seed=5,
            params={},
        )
        spec_other = ScenarioSpec(
            scenario_id="heterophily_rewire_nonoracle",
            family="heterophily",
            method="rewire_edge_dst_to_feature_pseudo_opposite_label",
            oracle_labels=False,
            severity_param="p_rewire",
            severity=1.0,
            graph_seed=17,
            params={},
        )

        with mock.patch.dict(sys.modules, {"dgl": _fake_dgl_module()}):
            g_a, applied_a, _info_a = apply_scenario(base, spec_same)
            g_b, applied_b, _info_b = apply_scenario(base, spec_same)
            g_c, applied_c, _info_c = apply_scenario(base, spec_other)

        self.assertTrue(applied_a)
        self.assertTrue(applied_b)
        self.assertTrue(applied_c)
        self.assertTrue(torch.equal(g_a.edges()[1], g_b.edges()[1]))
        self.assertFalse(torch.equal(g_a.edges()[1], g_c.edges()[1]))

    def test_relation_camouflage_adds_benign_neighbors_on_selected_relation(self) -> None:
        spec = ScenarioSpec(
            scenario_id="camouflage_relation_oracle",
            family="camouflage_relation",
            method="add_relation_camouflage_edges",
            oracle_labels=True,
            severity_param="p_cam_rel",
            severity=1.0,
            graph_seed=3,
            params={
                "camouflage_edges_per_node": 1,
                "relation_filter": ["r1"],
            },
        )

        base = _hetero_graph()
        with mock.patch.dict(sys.modules, {"dgl": _fake_dgl_module()}):
            g2, applied, info = apply_scenario(base, spec)

        self.assertTrue(applied)
        self.assertGreater(info["n_camouflaged_edges_added"], 0)
        self.assertGreater(info["fraud_to_normal_neighbor_ratio_after"], info["fraud_to_normal_neighbor_ratio_before"])
        self.assertEqual(info["relation_filter_applied"], ["node:r1:node"])
        self.assertGreater(g2.num_edges(("node", "r1", "node")), base.num_edges(("node", "r1", "node")))

    def test_noise_edges_follow_unique_non_self_loop_policy(self) -> None:
        spec = ScenarioSpec(
            scenario_id="noise_edges_uniform",
            family="noise",
            method="add_random_edges",
            oracle_labels=False,
            severity_param="edge_noise_rate",
            severity=2.0,
            graph_seed=19,
            params={},
        )
        base = FakeGraph(
            (torch.tensor([0], dtype=torch.int64), torch.tensor([1], dtype=torch.int64)),
            num_nodes=3,
            ndata={
                "label": torch.tensor([1, 0, 0], dtype=torch.int64),
                "feature": torch.tensor([[1.0], [0.0], [0.5]], dtype=torch.float32),
                "train_mask": torch.tensor([True, False, False]),
                "val_mask": torch.tensor([False, True, False]),
                "test_mask": torch.tensor([False, False, True]),
            },
        )

        with mock.patch.dict(sys.modules, {"dgl": _fake_dgl_module()}):
            g2, applied, info = apply_scenario(base, spec)

        self.assertTrue(applied)
        src, dst = g2.edges()
        edges = list(zip(src.tolist(), dst.tolist()))
        self.assertEqual(len(edges), len(set(edges)))
        self.assertTrue(all(u != v for u, v in edges))
        self.assertEqual(info["n_added_edges_actual_total"], g2.num_edges() - base.num_edges())


if __name__ == "__main__":
    unittest.main()
