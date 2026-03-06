from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from benchmark.scenario_audit import (
    VARIANT_AUDIT_COLUMNS,
    append_variant_audit_row,
    build_variant_audit_row,
    ensure_variant_audit_csv,
)


class ScenarioAuditTests(unittest.TestCase):
    def test_build_variant_audit_row_normalizes_requested_and_realized_metrics(self) -> None:
        row = build_variant_audit_row(
            experiment_name="exp",
            dataset_id="yelpchi",
            split_id="s0",
            scenario_id="heterophily_rewire_oracle",
            severity=0.3,
            graph_seed=7,
            oracle_labels=True,
            scenario_applied=True,
            scenario_family="heterophily",
            scenario_method="rewire_edge_dst_to_opposite_label",
            severity_param="p_rewire",
            graph_view_mode="canonical",
            base_graph_path="base.bin",
            graph_path="variant.bin",
            base_stats={"n_nodes": 10, "n_edges": 12, "heterophily_ratio": 0.2, "pos_rate": 0.3},
            variant_stats={"n_nodes": 10, "n_edges": 12, "heterophily_ratio": 0.5, "pos_rate": 0.3},
            scenario_info={
                "n_rewired_edges_selected": 4,
                "n_rewired_edges_actual": 3,
                "heterophily_ratio_before": 0.2,
                "heterophily_ratio_after": 0.5,
                "sampling_policy": {"allow_self_loops": False},
                "relation_filter_applied": ["node:r1:node"],
                "n_rewired_edges_actual_by_relation": {"node:r1:node": 3},
                "selection_proxy": "oracle",
            },
        )

        self.assertEqual(row["requested_change"], 4)
        self.assertEqual(row["requested_change_unit"], "edges_selected")
        self.assertEqual(row["realized_change"], 3)
        self.assertEqual(row["realized_change_unit"], "edges_changed")
        self.assertEqual(row["heterophily_ratio_before"], 0.2)
        self.assertEqual(row["heterophily_ratio_after"], 0.5)
        self.assertEqual(row["relation_filter_applied_json"], '["node:r1:node"]')
        self.assertIn('"selection_proxy":"oracle"', row["extra_info_json"])

    def test_build_variant_audit_row_leaves_non_applicable_metrics_blank_and_preserves_extra_logs(self) -> None:
        row = build_variant_audit_row(
            experiment_name="exp",
            dataset_id="yelpchi",
            split_id="s0",
            scenario_id="noise_edges_uniform",
            severity=0.2,
            graph_seed=9,
            oracle_labels=False,
            scenario_applied=True,
            scenario_family="noise",
            scenario_method="add_random_edges",
            severity_param="edge_noise_rate",
            graph_view_mode="heterograph_aware_generation",
            base_graph_path="base.bin",
            graph_path="variant.bin",
            base_stats={"n_nodes": 10, "n_edges": 12, "heterophily_ratio": 0.2, "pos_rate": 0.3},
            variant_stats={"n_nodes": 10, "n_edges": 14, "heterophily_ratio": 0.2, "pos_rate": 0.3},
            scenario_info={
                "n_added_edges_requested_total": 2,
                "n_added_edge_pairs_requested": 2,
                "n_added_edge_pairs_actual": 1,
                "n_added_edges_actual_total": 2,
                "n_noise_candidate_rejections_existing": 3,
                "n_noise_candidate_rejections_self_loop": 1,
            },
        )

        self.assertEqual(row["requested_change"], 2)
        self.assertEqual(row["requested_change_unit"], "edges_requested")
        self.assertEqual(row["realized_change"], 2)
        self.assertEqual(row["realized_change_unit"], "edges_added")
        self.assertEqual(row["mean_cosine_to_sampled_normal_before"], "")
        self.assertEqual(row["fraud_to_normal_neighbor_ratio_before"], "")
        self.assertEqual(row["sampling_policy_json"], "")
        self.assertIn('"n_noise_candidate_rejections_existing":3', row["extra_info_json"])
        self.assertIn('"n_noise_candidate_rejections_self_loop":1', row["extra_info_json"])

    def test_build_variant_audit_row_defaults_unapplied_scenarios_to_zero_realized_change(self) -> None:
        row = build_variant_audit_row(
            experiment_name="exp",
            dataset_id="yelpchi",
            split_id="s0",
            scenario_id="heterophily_rewire_oracle",
            severity=0.0,
            graph_seed=0,
            oracle_labels=True,
            scenario_applied=False,
            scenario_family="heterophily",
            scenario_method="rewire_edge_dst_to_opposite_label",
            severity_param="p_rewire",
            graph_view_mode="canonical",
            base_graph_path="base.bin",
            graph_path="base.bin",
            base_stats={"n_nodes": 10, "n_edges": 12},
            variant_stats={"n_nodes": 10, "n_edges": 12},
            scenario_info={"reason": "severity==0"},
        )

        self.assertEqual(row["requested_change"], 0)
        self.assertEqual(row["requested_change_unit"], "none")
        self.assertEqual(row["realized_change"], 0)
        self.assertEqual(row["realized_change_unit"], "none")
        self.assertIn('"reason":"severity==0"', row["extra_info_json"])

    def test_variant_audit_csv_header_and_append(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "variant_audit.csv"
            ensure_variant_audit_csv(path, overwrite=True)
            append_variant_audit_row(
                path,
                build_variant_audit_row(
                    experiment_name="exp",
                    dataset_id="yelpchi",
                    split_id="s0",
                    scenario_id="clean",
                    severity=0.0,
                    graph_seed=717,
                    oracle_labels=False,
                    scenario_applied=True,
                    scenario_family="clean",
                    scenario_method="clean",
                    severity_param="severity",
                    graph_view_mode="canonical",
                    base_graph_path="base.bin",
                    graph_path="base.bin",
                    base_stats={"n_nodes": 10, "n_edges": 12},
                    variant_stats={"n_nodes": 10, "n_edges": 12},
                    scenario_info={},
                ),
            )

            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(reader.fieldnames, VARIANT_AUDIT_COLUMNS)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["scenario_id"], "clean")
            self.assertEqual(rows[0]["requested_change"], "0")
            self.assertEqual(rows[0]["realized_change"], "0")


if __name__ == "__main__":
    unittest.main()
