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


def _masks(num_nodes: int) -> dict[str, torch.Tensor]:
    return {
        "train_mask": torch.tensor([idx < 2 for idx in range(num_nodes)], dtype=torch.bool),
        "val_mask": torch.tensor([idx == 2 for idx in range(num_nodes)], dtype=torch.bool),
        "test_mask": torch.tensor([idx >= 3 for idx in range(num_nodes)], dtype=torch.bool),
    }


def _rewire_graph() -> FakeGraph:
    labels = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.int64)
    features = torch.tensor(
        [
            [8.0, 0.2],
            [7.5, 0.4],
            [7.0, 0.6],
            [0.2, 8.0],
            [0.4, 7.5],
            [0.6, 7.0],
        ],
        dtype=torch.float32,
    )
    return FakeGraph(
        (
            torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64),
            torch.tensor([1, 2, 0, 4, 5, 3], dtype=torch.int64),
        ),
        num_nodes=6,
        ndata={"label": labels, "feature": features, **_masks(6)},
    )


def _fully_blocked_rewire_graph() -> FakeGraph:
    labels = torch.tensor([1, 1, 0, 0], dtype=torch.int64)
    features = torch.tensor(
        [
            [9.0, 0.0],
            [8.5, 0.5],
            [0.0, 9.0],
            [0.5, 8.5],
        ],
        dtype=torch.float32,
    )
    return FakeGraph(
        (
            torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64),
            torch.tensor([2, 3, 2, 3, 0, 1, 0, 1], dtype=torch.int64),
        ),
        num_nodes=4,
        ndata={"label": labels, "feature": features, **_masks(4)},
    )


def _feature_camouflage_graph() -> FakeGraph:
    labels = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.int64)
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.9, 0.1],
            [0.85, 0.15],
            [0.0, 1.0],
            [0.05, 0.95],
            [0.1, 0.9],
            [0.15, 0.85],
        ],
        dtype=torch.float32,
    )
    return FakeGraph(
        (
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64),
            torch.tensor([1, 2, 3, 0, 5, 6, 7, 4], dtype=torch.int64),
        ),
        num_nodes=8,
        ndata={"label": labels, "feature": features, **_masks(8)},
    )


def _relation_camouflage_graph() -> FakeHeteroGraph:
    labels = torch.tensor([1, 1, 0, 0, 0], dtype=torch.int64)
    features = torch.tensor(
        [
            [8.0, 0.0],
            [7.5, 0.5],
            [0.0, 8.0],
            [0.5, 7.5],
            [1.0, 7.0],
        ],
        dtype=torch.float32,
    )
    return FakeHeteroGraph(
        {
            ("node", "r1", "node"): (
                torch.tensor([0, 1, 2], dtype=torch.int64),
                torch.tensor([1, 0, 3], dtype=torch.int64),
            ),
            ("node", "r2", "node"): (
                torch.tensor([2, 3], dtype=torch.int64),
                torch.tensor([4, 2], dtype=torch.int64),
            ),
        },
        num_nodes_dict={"node": 5},
        ndata={"label": labels, "feature": features, **_masks(5)},
    )


def _noise_graph() -> FakeGraph:
    labels = torch.tensor([1, 0], dtype=torch.int64)
    features = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    return FakeGraph(
        (
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([1], dtype=torch.int64),
        ),
        num_nodes=2,
        ndata={"label": labels, "feature": features, **_masks(2)},
    )


def _spec(
    *,
    scenario_id: str,
    family: str,
    method: str,
    severity_param: str,
    severity: float,
    graph_seed: int,
    oracle_labels: bool,
    params: dict[str, Any] | None = None,
) -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id=scenario_id,
        family=family,
        method=method,
        oracle_labels=oracle_labels,
        severity_param=severity_param,
        severity=float(severity),
        graph_seed=int(graph_seed),
        params=dict(params or {}),
    )


def _edge_pairs(graph, etype=None) -> list[tuple[int, int]]:
    src, dst = graph.edges() if etype is None else graph.edges(etype=etype)
    return list(zip(src.tolist(), dst.tolist()))


def _assert_protected_node_data_unchanged(test_case: unittest.TestCase, base_graph, new_graph) -> None:
    test_case.assertTrue(torch.equal(base_graph.ndata["label"], new_graph.ndata["label"]))
    test_case.assertTrue(torch.equal(base_graph.ndata["train_mask"], new_graph.ndata["train_mask"]))
    test_case.assertTrue(torch.equal(base_graph.ndata["val_mask"], new_graph.ndata["val_mask"]))
    test_case.assertTrue(torch.equal(base_graph.ndata["test_mask"], new_graph.ndata["test_mask"]))


class ScenarioTests(unittest.TestCase):
    def test_zero_severity_is_a_noop_with_explicit_reason(self) -> None:
        base = _rewire_graph()
        spec = _spec(
            scenario_id="heterophily_rewire_oracle",
            family="heterophily",
            method="rewire_edge_dst_to_opposite_label",
            severity_param="p_rewire",
            severity=0.0,
            graph_seed=7,
            oracle_labels=True,
        )

        with mock.patch.dict(sys.modules, {"dgl": _fake_dgl_module()}):
            g2, applied, info = apply_scenario(base, spec)

        self.assertIs(g2, base)
        self.assertFalse(applied)
        self.assertEqual(info, {"reason": "severity==0"})

    def test_oracle_rewire_increases_heterophily_and_preserves_invariants(self) -> None:
        base = _rewire_graph()
        spec = _spec(
            scenario_id="heterophily_rewire_oracle",
            family="heterophily",
            method="rewire_edge_dst_to_opposite_label",
            severity_param="p_rewire",
            severity=1.0,
            graph_seed=7,
            oracle_labels=True,
        )

        with mock.patch.dict(sys.modules, {"dgl": _fake_dgl_module()}):
            g2, applied, info = apply_scenario(base, spec)

        self.assertTrue(applied)
        self.assertEqual(g2.num_nodes(), base.num_nodes())
        self.assertEqual(g2.num_edges(), base.num_edges())
        _assert_protected_node_data_unchanged(self, base, g2)
        self.assertGreater(info["heterophily_ratio_after"], info["heterophily_ratio_before"])
        self.assertEqual(info["n_rewired_edges_selected"], base.num_edges())
        self.assertEqual(info["n_rewired_edges_actual"], base.num_edges())
        self.assertEqual(info["n_rewire_candidate_rejections_existing"], 0)
        self.assertGreater(info["n_rewire_candidate_attempts"], 0)

    def test_rewire_logs_selected_vs_realized_when_policy_blocks_all_changes(self) -> None:
        base = _fully_blocked_rewire_graph()
        spec = _spec(
            scenario_id="heterophily_rewire_oracle",
            family="heterophily",
            method="rewire_edge_dst_to_opposite_label",
            severity_param="p_rewire",
            severity=1.0,
            graph_seed=3,
            oracle_labels=True,
        )

        with mock.patch.dict(sys.modules, {"dgl": _fake_dgl_module()}):
            g2, applied, info = apply_scenario(base, spec)

        self.assertFalse(applied)
        self.assertEqual(_edge_pairs(g2), _edge_pairs(base))
        _assert_protected_node_data_unchanged(self, base, g2)
        self.assertEqual(info["n_rewired_edges_selected"], base.num_edges())
        self.assertEqual(info["n_rewired_edges_actual"], 0)
        self.assertEqual(info["heterophily_ratio_before"], 1.0)
        self.assertEqual(info["heterophily_ratio_after"], 1.0)
        self.assertGreater(info["n_rewire_candidate_rejections_existing"], 0)
        self.assertGreater(info["n_rewire_candidate_rejections_exhausted"], 0)

    def test_nonoracle_rewire_is_deterministic_for_same_seed_and_varies_for_other_seed(self) -> None:
        base = _rewire_graph()
        spec_same = _spec(
            scenario_id="heterophily_rewire_nonoracle",
            family="heterophily",
            method="rewire_edge_dst_to_feature_pseudo_opposite_label",
            severity_param="p_rewire",
            severity=1.0,
            graph_seed=5,
            oracle_labels=False,
        )
        spec_other = _spec(
            scenario_id="heterophily_rewire_nonoracle",
            family="heterophily",
            method="rewire_edge_dst_to_feature_pseudo_opposite_label",
            severity_param="p_rewire",
            severity=1.0,
            graph_seed=17,
            oracle_labels=False,
        )

        with mock.patch.dict(sys.modules, {"dgl": _fake_dgl_module()}):
            g_a, applied_a, info_a = apply_scenario(base, spec_same)
            g_b, applied_b, info_b = apply_scenario(base, spec_same)
            g_c, applied_c, _info_c = apply_scenario(base, spec_other)

        self.assertTrue(applied_a)
        self.assertTrue(applied_b)
        self.assertTrue(applied_c)
        self.assertTrue(torch.equal(g_a.edges()[1], g_b.edges()[1]))
        self.assertEqual(info_a["pseudo_label_pos_rate"], info_b["pseudo_label_pos_rate"])
        self.assertFalse(torch.equal(g_a.edges()[1], g_c.edges()[1]))

    def test_feature_camouflage_changes_expected_fraction_and_increases_similarity(self) -> None:
        base = _feature_camouflage_graph()
        spec = _spec(
            scenario_id="camouflage_feature_oracle",
            family="camouflage_feature",
            method="replace_fraud_features_with_normal_features",
            severity_param="p_cam_feat",
            severity=0.5,
            graph_seed=11,
            oracle_labels=True,
            params={"gamma": 1.0},
        )

        with mock.patch.dict(sys.modules, {"dgl": _fake_dgl_module()}):
            g2, applied, info = apply_scenario(base, spec)

        self.assertTrue(applied)
        changed = torch.any(g2.ndata["feature"] != base.ndata["feature"], dim=1)
        changed_idx = torch.nonzero(changed, as_tuple=True)[0]
        changed_labels = base.ndata["label"][changed_idx]
        self.assertEqual(int(changed_idx.numel()), 2)
        self.assertTrue(torch.all(changed_labels == 1))
        self.assertEqual(info["n_camouflaged_nodes_requested"], 2)
        self.assertEqual(info["n_camouflaged_nodes_actual"], 2)
        self.assertGreater(info["mean_cosine_to_sampled_normal_after"], info["mean_cosine_to_sampled_normal_before"])
        _assert_protected_node_data_unchanged(self, base, g2)

    def test_feature_camouflage_is_deterministic_for_same_seed(self) -> None:
        base = _feature_camouflage_graph()
        spec = _spec(
            scenario_id="camouflage_feature_oracle",
            family="camouflage_feature",
            method="replace_fraud_features_with_normal_features",
            severity_param="p_cam_feat",
            severity=0.5,
            graph_seed=23,
            oracle_labels=True,
            params={"gamma": 1.0},
        )

        with mock.patch.dict(sys.modules, {"dgl": _fake_dgl_module()}):
            g_a, applied_a, _info_a = apply_scenario(base, spec)
            g_b, applied_b, _info_b = apply_scenario(base, spec)

        self.assertTrue(applied_a)
        self.assertTrue(applied_b)
        self.assertTrue(torch.equal(g_a.ndata["feature"], g_b.ndata["feature"]))

    def test_relation_camouflage_add_only_path_updates_neighbor_mix_and_edge_count(self) -> None:
        base = _relation_camouflage_graph()
        spec = _spec(
            scenario_id="camouflage_relation_oracle",
            family="camouflage_relation",
            method="add_relation_camouflage_edges",
            severity_param="p_cam_rel",
            severity=1.0,
            graph_seed=3,
            oracle_labels=True,
            params={"camouflage_edges_per_node": 1, "relation_filter": ["r1"], "remove_suspicious_ratio": 0.0},
        )

        with mock.patch.dict(sys.modules, {"dgl": _fake_dgl_module()}):
            g2, applied, info = apply_scenario(base, spec)

        self.assertTrue(applied)
        self.assertEqual(info["n_camouflaged_nodes"], 2)
        self.assertEqual(info["n_suspicious_edges_removed"], 0)
        self.assertEqual(info["relation_filter_applied"], ["node:r1:node"])
        self.assertGreater(info["fraud_to_normal_neighbor_ratio_after"], info["fraud_to_normal_neighbor_ratio_before"])
        self.assertGreater(info["mean_selected_out_degree_after"], info["mean_selected_out_degree_before"])
        self.assertEqual(
            g2.num_edges(("node", "r1", "node")),
            base.num_edges(("node", "r1", "node")) + int(info["n_camouflaged_edges_added"]),
        )
        self.assertEqual(g2.num_edges(("node", "r2", "node")), base.num_edges(("node", "r2", "node")))
        _assert_protected_node_data_unchanged(self, base, g2)

    def test_relation_camouflage_add_plus_remove_path_matches_logged_net_change(self) -> None:
        base = _relation_camouflage_graph()
        spec = _spec(
            scenario_id="camouflage_relation_oracle",
            family="camouflage_relation",
            method="add_relation_camouflage_edges",
            severity_param="p_cam_rel",
            severity=1.0,
            graph_seed=5,
            oracle_labels=True,
            params={"camouflage_edges_per_node": 1, "relation_filter": ["r1"], "remove_suspicious_ratio": 1.0},
        )

        with mock.patch.dict(sys.modules, {"dgl": _fake_dgl_module()}):
            g2, applied, info = apply_scenario(base, spec)

        self.assertTrue(applied)
        self.assertEqual(info["n_camouflaged_edges_added"], 2)
        self.assertEqual(info["n_suspicious_edges_removed_requested"], 2)
        self.assertEqual(info["n_suspicious_edges_removed"], 2)
        self.assertLess(info["fraud_to_fraud_neighbor_ratio_after"], info["fraud_to_fraud_neighbor_ratio_before"])
        expected_edges = base.num_edges(("node", "r1", "node")) + info["n_camouflaged_edges_added"] - info["n_suspicious_edges_removed"]
        self.assertEqual(g2.num_edges(("node", "r1", "node")), expected_edges)
        _assert_protected_node_data_unchanged(self, base, g2)

    def test_relation_camouflage_is_deterministic_for_same_seed_and_varies_for_other_seed(self) -> None:
        base = _relation_camouflage_graph()
        spec_same = _spec(
            scenario_id="camouflage_relation_oracle",
            family="camouflage_relation",
            method="add_relation_camouflage_edges",
            severity_param="p_cam_rel",
            severity=1.0,
            graph_seed=13,
            oracle_labels=True,
            params={"camouflage_edges_per_node": 1, "relation_filter": ["r1"], "remove_suspicious_ratio": 0.0},
        )
        spec_other = _spec(
            scenario_id="camouflage_relation_oracle",
            family="camouflage_relation",
            method="add_relation_camouflage_edges",
            severity_param="p_cam_rel",
            severity=1.0,
            graph_seed=29,
            oracle_labels=True,
            params={"camouflage_edges_per_node": 1, "relation_filter": ["r1"], "remove_suspicious_ratio": 0.0},
        )

        with mock.patch.dict(sys.modules, {"dgl": _fake_dgl_module()}):
            g_a, applied_a, _info_a = apply_scenario(base, spec_same)
            g_b, applied_b, _info_b = apply_scenario(base, spec_same)
            g_c, applied_c, _info_c = apply_scenario(base, spec_other)

        self.assertTrue(applied_a)
        self.assertTrue(applied_b)
        self.assertTrue(applied_c)
        self.assertEqual(_edge_pairs(g_a, ("node", "r1", "node")), _edge_pairs(g_b, ("node", "r1", "node")))
        self.assertNotEqual(_edge_pairs(g_a, ("node", "r1", "node")), _edge_pairs(g_c, ("node", "r1", "node")))

    def test_noise_default_policy_rejects_existing_edges_duplicates_and_self_loops(self) -> None:
        base = _noise_graph()
        spec = _spec(
            scenario_id="noise_edges_uniform",
            family="noise",
            method="add_random_edges",
            severity_param="edge_noise_rate",
            severity=4.0,
            graph_seed=19,
            oracle_labels=False,
        )

        with mock.patch.dict(sys.modules, {"dgl": _fake_dgl_module()}):
            g2, applied, info = apply_scenario(base, spec)

        self.assertTrue(applied)
        self.assertEqual(info["n_added_edge_pairs_requested"], 4)
        self.assertEqual(info["n_added_edge_pairs_actual"], 1)
        self.assertEqual(info["n_added_edges_actual_total"], 1)
        self.assertEqual(info["n_noise_candidate_rejections_exhausted"], 3)
        self.assertGreater(info["n_noise_candidate_rejections_existing"], 0)
        self.assertGreater(info["n_noise_candidate_rejections_self_loop"], 0)
        edges = _edge_pairs(g2)
        self.assertEqual(len(edges), len(set(edges)))
        self.assertTrue(all(src != dst for src, dst in edges))
        self.assertNotIn((0, 1), edges[1:])
        _assert_protected_node_data_unchanged(self, base, g2)

    def test_noise_same_seed_repeats_the_same_realization(self) -> None:
        base = FakeGraph(
            (
                torch.tensor([0, 1], dtype=torch.int64),
                torch.tensor([1, 2], dtype=torch.int64),
            ),
            num_nodes=4,
            ndata={
                "label": torch.tensor([1, 0, 0, 1], dtype=torch.int64),
                "feature": torch.tensor([[1.0], [0.0], [0.5], [0.8]], dtype=torch.float32),
                **_masks(4),
            },
        )
        spec = _spec(
            scenario_id="noise_edges_uniform",
            family="noise",
            method="add_random_edges",
            severity_param="edge_noise_rate",
            severity=1.0,
            graph_seed=31,
            oracle_labels=False,
        )

        with mock.patch.dict(sys.modules, {"dgl": _fake_dgl_module()}):
            g_a, applied_a, info_a = apply_scenario(base, spec)
            g_b, applied_b, info_b = apply_scenario(base, spec)

        self.assertTrue(applied_a)
        self.assertTrue(applied_b)
        self.assertEqual(_edge_pairs(g_a), _edge_pairs(g_b))
        self.assertEqual(info_a["n_added_edge_pairs_actual"], info_b["n_added_edge_pairs_actual"])


if __name__ == "__main__":
    unittest.main()
