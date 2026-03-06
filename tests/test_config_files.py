from __future__ import annotations

import unittest
from pathlib import Path

from benchmark.config import (
    get_graph_seeds,
    get_training_seeds,
    load_json,
    scenario_graph_view_mode,
    scenario_oracle_labels,
    should_export_variant_audit,
    validate_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"


def _load_config(name: str) -> dict:
    cfg = load_json(CONFIGS_DIR / name)
    validate_config(cfg)
    return cfg


def _scenario_values(cfg: dict[str, object]) -> dict[str, list[float]]:
    scenarios = cfg["scenarios"]
    assert isinstance(scenarios, list)
    return {
        str(scenario["scenario_id"]): list(scenario["severity_values"])
        for scenario in scenarios
        if isinstance(scenario, dict)
    }


def _model_map(cfg: dict[str, object]) -> dict[str, dict[str, object]]:
    models = cfg["models"]
    assert isinstance(models, list)
    return {
        str(model["model_id"]): model
        for model in models
        if isinstance(model, dict)
    }


def _scenario_map(cfg: dict[str, object]) -> dict[str, dict[str, object]]:
    scenarios = cfg["scenarios"]
    assert isinstance(scenarios, list)
    return {
        str(scenario["scenario_id"]): scenario
        for scenario in scenarios
        if isinstance(scenario, dict)
    }


class FrozenConfigFileTests(unittest.TestCase):
    def test_all_frozen_configs_validate(self) -> None:
        for name in (
            "exp_yelpchi_v1.json",
            "exp_yelpchi_v2.json",
            "exp_yelpchi_v2_fast.json",
            "exp_yelpchi_v2_full.json",
            "exp_yelpchi_v3_fast.json",
            "exp_yelpchi_v3.json",
            "exp_yelpchi_v3_full.json",
        ):
            with self.subTest(config=name):
                _load_config(name)

    def test_v1_config_remains_backward_compatible(self) -> None:
        cfg = _load_config("exp_yelpchi_v1.json")

        self.assertEqual(cfg["experiment_name"], "gfd_robustness_benchmark_v1")
        self.assertEqual(get_training_seeds(cfg), [0, 1, 2, 3, 4])
        self.assertEqual(get_graph_seeds(cfg), [0, 1, 2, 3, 4])

    def test_v2_config_matches_primary_publishable_matrix(self) -> None:
        cfg = _load_config("exp_yelpchi_v2.json")
        models = _model_map(cfg)

        self.assertEqual(cfg["experiment_name"], "gfd_robustness_benchmark_v2")
        self.assertEqual(get_graph_seeds(cfg), [0, 1])
        self.assertEqual(get_training_seeds(cfg), [0, 1, 2])
        self.assertEqual(list(models), ["mlp", "sage", "pmp", "secgfd"])
        self.assertEqual(
            _scenario_values(cfg),
            {
                "heterophily_rewire_oracle": [0.0, 0.3],
                "camouflage_feature_oracle": [0.0, 0.3],
                "noise_edges": [0.0, 0.2],
            },
        )
        self.assertEqual(
            models["pmp"]["hparams"],
            {
                "full_neighbors": False,
                "sampled_neighbors": [10],
                "batch_size": 512,
                "num_workers": 0,
                "epochs": 100,
                "patience": 10,
            },
        )
        self.assertEqual(
            models["secgfd"]["hparams"],
            {
                "hid_dim": 32,
                "order_d": 2,
                "high_order": 1,
                "lemda": 0.2,
                "lr": 0.01,
                "weight_decay": 0.0,
            },
        )

    def test_v2_fast_config_is_small_baseline_only_matrix(self) -> None:
        cfg = _load_config("exp_yelpchi_v2_fast.json")
        models = _model_map(cfg)

        self.assertEqual(cfg["experiment_name"], "gfd_robustness_dev")
        self.assertEqual(get_graph_seeds(cfg), [0])
        self.assertEqual(get_training_seeds(cfg), [0])
        self.assertEqual(list(models), ["mlp", "sage"])
        self.assertEqual(
            _scenario_values(cfg),
            {
                "heterophily_rewire_oracle": [0.0, 0.3],
                "camouflage_feature_oracle": [0.0, 0.3],
                "noise_edges": [0.0, 0.2],
            },
        )

    def test_v2_full_config_restores_full_curves(self) -> None:
        cfg = _load_config("exp_yelpchi_v2_full.json")
        models = _model_map(cfg)

        self.assertEqual(cfg["experiment_name"], "gfd_robustness_benchmark_v2_full")
        self.assertEqual(get_graph_seeds(cfg), [0, 1])
        self.assertEqual(get_training_seeds(cfg), [0, 1, 2, 3, 4])
        self.assertEqual(list(models), ["mlp", "sage", "pmp", "secgfd"])
        self.assertEqual(
            _scenario_values(cfg),
            {
                "heterophily_rewire_oracle": [0.0, 0.1, 0.2, 0.3],
                "camouflage_feature_oracle": [0.0, 0.1, 0.2, 0.3],
                "noise_edges": [0.0, 0.05, 0.1, 0.2],
            },
        )

    def test_v3_fast_config_uses_full_v3_scenario_surface_in_smoke_test_form(self) -> None:
        cfg = _load_config("exp_yelpchi_v3_fast.json")
        models = _model_map(cfg)
        scenarios = _scenario_map(cfg)

        self.assertEqual(cfg["experiment_name"], "gfd_robustness_v3_dev")
        self.assertEqual(get_graph_seeds(cfg), [0])
        self.assertEqual(get_training_seeds(cfg), [0])
        self.assertEqual(list(models), ["mlp", "sage"])
        self.assertTrue(should_export_variant_audit(cfg))
        self.assertEqual(
            _scenario_values(cfg),
            {
                "heterophily_rewire_oracle": [0.0, 0.3],
                "heterophily_rewire_nonoracle": [0.0, 0.3],
                "camouflage_feature_oracle": [0.0, 0.3],
                "camouflage_relation_oracle": [0.0, 0.3],
                "noise_edges_uniform": [0.0, 0.2],
            },
        )
        self.assertEqual(scenarios["camouflage_feature_oracle"]["scenario_group"], "camouflage")
        self.assertEqual(scenarios["camouflage_relation_oracle"]["scenario_group"], "camouflage")
        self.assertEqual(scenario_graph_view_mode(scenarios["heterophily_rewire_oracle"]), "heterograph_aware_generation")
        self.assertEqual(scenario_graph_view_mode(scenarios["camouflage_feature_oracle"]), "canonical")
        self.assertTrue(scenario_oracle_labels(scenarios["heterophily_rewire_oracle"]))
        self.assertFalse(scenario_oracle_labels(scenarios["heterophily_rewire_nonoracle"]))
        self.assertEqual(
            scenarios["camouflage_relation_oracle"]["relation_filter"],
            ["net_rsr", "net_rtr", "net_rur"],
        )
        self.assertEqual(scenarios["camouflage_relation_oracle"]["camouflage_edges_per_node"], 2)
        self.assertEqual(scenarios["camouflage_relation_oracle"]["remove_suspicious_ratio"], 0.5)

    def test_v3_main_config_matches_report_facing_matrix(self) -> None:
        cfg = _load_config("exp_yelpchi_v3.json")
        models = _model_map(cfg)
        scenarios = _scenario_map(cfg)

        self.assertEqual(cfg["experiment_name"], "gfd_robustness_benchmark_v3")
        self.assertEqual(get_graph_seeds(cfg), [0, 1])
        self.assertEqual(get_training_seeds(cfg), [0, 1, 2])
        self.assertEqual(list(models), ["mlp", "sage", "pmp", "secgfd"])
        self.assertTrue(should_export_variant_audit(cfg))
        self.assertEqual(
            _scenario_values(cfg),
            {
                "heterophily_rewire_oracle": [0.0, 0.15, 0.3],
                "heterophily_rewire_nonoracle": [0.0, 0.15, 0.3],
                "camouflage_feature_oracle": [0.0, 0.15, 0.3],
                "camouflage_relation_oracle": [0.0, 0.15, 0.3],
                "noise_edges_uniform": [0.0, 0.1, 0.2],
            },
        )
        self.assertTrue(scenario_oracle_labels(scenarios["camouflage_relation_oracle"]))
        self.assertFalse(scenario_oracle_labels(scenarios["noise_edges_uniform"]))
        self.assertEqual(scenario_graph_view_mode(scenarios["noise_edges_uniform"]), "heterograph_aware_generation")
        self.assertEqual(
            cfg["evaluation"]["audit_metrics"],
            [
                "n_rewired_edges_actual",
                "heterophily_ratio_before",
                "heterophily_ratio_after",
                "mean_cosine_to_sampled_normal_before",
                "mean_cosine_to_sampled_normal_after",
                "fraud_to_normal_neighbor_ratio_before",
                "fraud_to_normal_neighbor_ratio_after",
                "n_added_edge_pairs_actual",
            ],
        )

    def test_v3_full_config_restores_denser_curves_and_broader_seed_coverage(self) -> None:
        cfg = _load_config("exp_yelpchi_v3_full.json")
        models = _model_map(cfg)

        self.assertEqual(cfg["experiment_name"], "gfd_robustness_benchmark_v3_full")
        self.assertEqual(get_graph_seeds(cfg), [0, 1, 2])
        self.assertEqual(get_training_seeds(cfg), [0, 1, 2, 3, 4])
        self.assertEqual(list(models), ["mlp", "sage", "pmp", "secgfd"])
        self.assertEqual(
            _scenario_values(cfg),
            {
                "heterophily_rewire_oracle": [0.0, 0.1, 0.2, 0.3],
                "heterophily_rewire_nonoracle": [0.0, 0.1, 0.2, 0.3],
                "camouflage_feature_oracle": [0.0, 0.1, 0.2, 0.3],
                "camouflage_relation_oracle": [0.0, 0.1, 0.2, 0.3],
                "noise_edges_uniform": [0.0, 0.05, 0.1, 0.15, 0.2],
            },
        )


if __name__ == "__main__":
    unittest.main()
