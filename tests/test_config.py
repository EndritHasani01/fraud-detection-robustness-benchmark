from __future__ import annotations

import contextlib
import io
import unittest

from benchmark.config import ConfigError, get_graph_seeds, get_training_seeds, validate_config


def _base_cfg() -> dict:
    return {
        "experiment_name": "exp",
        "datasets": [{"dataset_id": "yelpchi"}],
        "data_splits": [{"split_id": "s0"}],
        "graph_representation": {"canonical_view": "homogeneous"},
        "models": [{"model_id": "mlp"}],
        "scenarios": [{"scenario_id": "noise_edges", "severity_values": [0.0, 0.1]}],
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


if __name__ == "__main__":
    unittest.main()
