from __future__ import annotations

import contextlib
import io
import unittest

from benchmark.config import (
    ConfigError,
    get_graph_seeds,
    get_training_seeds,
    scenario_graph_view_mode,
    scenario_oracle_labels,
    should_export_variant_audit,
    validate_config,
)


def _base_cfg() -> dict:
    return {
        "experiment_name": "exp",
        "datasets": [{"dataset_id": "yelpchi", "source_name": "yelp"}],
        "data_splits": [{"split_id": "s0", "split_seed": 0, "train_size": 0.4, "val_size": 0.2}],
        "graph_representation": {"canonical_view": "homogeneous"},
        "models": [{"model_id": "mlp"}],
        "scenarios": [
            {
                "scenario_id": "noise_edges",
                "family": "noise",
                "severity_param": "edge_noise_rate",
                "severity_values": [0.0, 0.1],
                "method": "add_random_edges",
            }
        ],
        "seeds": {"training_seeds": [0, 1, 2]},
        "evaluation": {"metrics": ["roc_auc"]},
    }


class ConfigSeedTests(unittest.TestCase):
    def test_validate_config_accepts_v1_seed_layout(self) -> None:
        cfg = _base_cfg()
        validate_config(cfg)
        self.assertEqual(get_graph_seeds(cfg), [0, 1, 2])
        self.assertEqual(get_training_seeds(cfg), [0, 1, 2])

    def test_get_graph_seeds_prefers_graph_seed_list(self) -> None:
        cfg = _base_cfg()
        cfg["seeds"]["graph_seeds"] = [7, 8]

        validate_config(cfg)

        self.assertEqual(get_graph_seeds(cfg), [7, 8])
        self.assertEqual(get_training_seeds(cfg), [0, 1, 2])

    def test_validate_config_rejects_invalid_graph_seed_entries(self) -> None:
        cfg = _base_cfg()
        cfg["seeds"]["graph_seeds"] = [0, True]

        with self.assertRaisesRegex(ConfigError, r"seeds\.graph_seeds\[1\] must be an integer"):
            validate_config(cfg)

    def test_validate_config_rejects_duplicate_seed_values(self) -> None:
        cfg = _base_cfg()
        cfg["seeds"]["training_seeds"] = [0, 0]

        with self.assertRaisesRegex(ConfigError, r"seeds\.training_seeds must not contain duplicate values"):
            validate_config(cfg)

    def test_validate_config_rejects_duplicate_or_invalid_splits(self) -> None:
        cfg = _base_cfg()
        cfg["data_splits"].append(
            {"split_id": "s1", "split_seed": 0, "train_size": 0.4, "val_size": 0.2}
        )
        with self.assertRaisesRegex(ConfigError, r"unique split_seed"):
            validate_config(cfg)

        cfg = _base_cfg()
        cfg["data_splits"][0]["val_size"] = 0.7
        with self.assertRaisesRegex(ConfigError, r"train_size \+ .*val_size must be < 1"):
            validate_config(cfg)

    def test_validate_config_rejects_duplicate_identifiers_and_severities(self) -> None:
        cfg = _base_cfg()
        cfg["models"].append({"model_id": "mlp"})
        with self.assertRaisesRegex(ConfigError, r"unique model_id"):
            validate_config(cfg)

        cfg = _base_cfg()
        cfg["scenarios"][0]["severity_values"] = [0.0, 0.1, 0.1]
        with self.assertRaisesRegex(ConfigError, r"severity_values must not contain duplicates"):
            validate_config(cfg)

        cfg = _base_cfg()
        cfg["scenarios"][0]["severity_values"] = [0.1]
        with self.assertRaisesRegex(ConfigError, r"severity_values must include 0.0"):
            validate_config(cfg)

    def test_validate_config_warns_then_errors_when_both_seed_lists_missing(self) -> None:
        cfg = _base_cfg()
        cfg["seeds"] = {}

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaisesRegex(ConfigError, r"seeds\.training_seeds must be a non-empty list of integers"):
                validate_config(cfg)

        self.assertIn("missing both seeds.training_seeds and seeds.graph_seeds", stderr.getvalue())

    def test_validate_config_accepts_secgfd_hparams(self) -> None:
        cfg = _base_cfg()
        cfg["models"] = [
            {
                "model_id": "secgfd",
                "repo_path": "Repos/SEC-GFD-main",
                "hparams": {
                    "hid_dim": 32,
                    "order_d": 2,
                    "high_order": 1,
                    "lemda": 0.2,
                    "lr": 0.01,
                    "weight_decay": 0.0,
                },
            }
        ]

        validate_config(cfg)

    def test_validate_config_rejects_invalid_secgfd_hparams(self) -> None:
        cfg = _base_cfg()
        cfg["models"] = [
            {
                "model_id": "secgfd",
                "repo_path": "Repos/SEC-GFD-main",
                "hparams": {"high_order": "fast"},
            }
        ]

        with self.assertRaisesRegex(ConfigError, r"secgfd\.hparams\.high_order must be an integer"):
            validate_config(cfg)


class ConfigV3SchemaTests(unittest.TestCase):
    def test_validate_config_accepts_v3_scenario_metadata(self) -> None:
        cfg = _base_cfg()
        cfg["scenarios"] = [
            {
                "scenario_id": "heterophily_rewire_nonoracle",
                "family": "heterophily",
                "scenario_group": "heterophily",
                "severity_param": "p_rewire",
                "severity_values": [0.0, 0.15, 0.3],
                "method": "rewire_edge_dst_to_feature_pseudo_opposite_label",
                "oracle_mode": "non_oracle",
                "graph_view_mode": "heterograph_aware_generation",
                "relation_filter": ["net_rsr", "net_rtr", "net_rur"],
                "allow_self_loops": False,
                "reject_existing": True,
                "reject_duplicates": True,
                "max_attempt_multiplier": 20,
            },
            {
                "scenario_id": "camouflage_relation_oracle",
                "family": "camouflage_relation",
                "scenario_group": "camouflage",
                "severity_param": "p_cam_rel",
                "severity_values": [0.0, 0.3],
                "method": "add_relation_camouflage_edges",
                "oracle_mode": "oracle",
                "oracle_labels": True,
                "graph_view_mode": "heterograph",
                "relation_filter": ["net_rsr"],
                "camouflage_edges_per_node": 2,
                "remove_suspicious_ratio": 0.5,
            },
        ]
        cfg["evaluation"]["export_variant_audit"] = True
        cfg["evaluation"]["audit_metrics"] = [
            "n_rewired_edges_actual",
            "heterophily_ratio_after",
            "fraud_to_normal_neighbor_ratio_after",
        ]

        validate_config(cfg)

        self.assertFalse(scenario_oracle_labels(cfg["scenarios"][0]))
        self.assertTrue(scenario_oracle_labels(cfg["scenarios"][1]))
        self.assertEqual(scenario_graph_view_mode(cfg["scenarios"][1]), "heterograph_aware_generation")
        self.assertTrue(should_export_variant_audit(cfg))

    def test_validate_config_rejects_invalid_oracle_mode(self) -> None:
        cfg = _base_cfg()
        cfg["scenarios"][0]["oracle_mode"] = "mystery"

        with self.assertRaisesRegex(ConfigError, r"cfg\['scenarios'\]\[0\]\.oracle_mode must be 'oracle' or 'non_oracle'"):
            validate_config(cfg)

    def test_validate_config_rejects_conflicting_oracle_fields(self) -> None:
        cfg = _base_cfg()
        cfg["scenarios"][0]["oracle_mode"] = "oracle"
        cfg["scenarios"][0]["oracle_labels"] = False

        with self.assertRaisesRegex(
            ConfigError,
            r"cfg\['scenarios'\]\[0\]\.oracle_mode conflicts with cfg\['scenarios'\]\[0\]\.oracle_labels",
        ):
            validate_config(cfg)

    def test_validate_config_rejects_invalid_graph_view_mode(self) -> None:
        cfg = _base_cfg()
        cfg["scenarios"][0]["graph_view_mode"] = "invalid"

        with self.assertRaisesRegex(
            ConfigError,
            r"scenario\.graph_view_mode must be 'canonical' or 'heterograph_aware_generation'",
        ):
            validate_config(cfg)

    def test_validate_config_rejects_invalid_relation_filter(self) -> None:
        cfg = _base_cfg()
        cfg["scenarios"][0]["relation_filter"] = ["net_rsr", ""]

        with self.assertRaisesRegex(ConfigError, r"cfg\['scenarios'\]\[0\]\.relation_filter\[1\] must be a non-empty string"):
            validate_config(cfg)

    def test_validate_config_rejects_invalid_relation_camouflage_parameters(self) -> None:
        cfg = _base_cfg()
        cfg["scenarios"][0]["camouflage_edges_per_node"] = 0

        with self.assertRaisesRegex(
            ConfigError,
            r"cfg\['scenarios'\]\[0\]\.camouflage_edges_per_node must be an integer >= 1",
        ):
            validate_config(cfg)

    def test_validate_config_accepts_degree_relative_relation_camouflage(self) -> None:
        cfg = _base_cfg()
        cfg["scenarios"][0].update(
            {
                "camouflage_edge_degree_ratio": 0.25,
                "camouflage_min_edges_per_node": 2,
                "camouflage_max_edges_per_node": 64,
            }
        )
        validate_config(cfg)

        cfg["scenarios"][0]["camouflage_edges_per_node"] = 2
        with self.assertRaisesRegex(ConfigError, r"must set only one"):
            validate_config(cfg)

    def test_validate_config_accepts_prospective_v4_scenario_controls(self) -> None:
        cfg = _base_cfg()
        cfg["scenarios"][0]["fixed_feature_partition_across_severity"] = True
        cfg["scenarios"][0]["relation_allocation"] = "proportional"
        validate_config(cfg)

        cfg["scenarios"][0]["relation_allocation"] = "largest_first"
        with self.assertRaisesRegex(ConfigError, r"relation_allocation must be 'uniform' or 'proportional'"):
            validate_config(cfg)

    def test_validate_config_rejects_invalid_audit_export_controls(self) -> None:
        cfg = _base_cfg()
        cfg["evaluation"]["export_variant_audit"] = "yes"

        with self.assertRaisesRegex(ConfigError, r"evaluation\.export_variant_audit must be a boolean"):
            validate_config(cfg)

    def test_should_export_variant_audit_defaults_to_true(self) -> None:
        cfg = _base_cfg()

        self.assertTrue(should_export_variant_audit(cfg))
        self.assertTrue(should_export_variant_audit({"experiment_name": "exp"}))


if __name__ == "__main__":
    unittest.main()
