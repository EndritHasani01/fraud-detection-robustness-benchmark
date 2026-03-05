from __future__ import annotations

import csv
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from benchmark.run import main
from benchmark.secgfd_stage import (
    _SECGFD_THETA_CACHE,
    _install_secgfd_theta_cache,
    run_secgfd_stage,
)


def _write_variant_csv(path: Path) -> None:
    columns = [
        "experiment_name",
        "dataset_id",
        "split_id",
        "graph_seed",
        "scenario_id",
        "severity",
        "oracle_labels",
        "scenario_applied",
        "base_graph_path",
        "graph_path",
        "n_nodes",
        "n_edges",
        "mean_in_degree",
        "median_in_degree",
        "mean_out_degree",
        "median_out_degree",
        "heterophily_ratio",
        "pos_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerow(
            {
                "experiment_name": "exp",
                "dataset_id": "yelpchi",
                "split_id": "split_0",
                "graph_seed": 0,
                "scenario_id": "clean",
                "severity": 0.0,
                "oracle_labels": False,
                "scenario_applied": True,
                "base_graph_path": "base.bin",
                "graph_path": "graph.bin",
                "n_nodes": 10,
                "n_edges": 12,
                "mean_in_degree": 1.2,
                "median_in_degree": 1.0,
                "mean_out_degree": 1.2,
                "median_out_degree": 1.0,
                "heterophily_ratio": 0.3,
                "pos_rate": 0.2,
            }
        )


class SecgfdCacheTests(unittest.TestCase):
    def test_theta_cache_avoids_recomputing_same_degree(self) -> None:
        _SECGFD_THETA_CACHE.clear()
        calls = {"count": 0}

        def calculate_theta2(d):
            calls["count"] += 1
            return [[float(d), float(calls["count"])]]

        module = types.SimpleNamespace(
            __file__="dummy_secgfd.py",
            calculate_theta2=calculate_theta2,
        )

        _install_secgfd_theta_cache(module)

        first = module.calculate_theta2(2)
        second = module.calculate_theta2(d=2)
        third = module.calculate_theta2(3)

        self.assertEqual(calls["count"], 2)
        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        self.assertEqual(third, [[3.0, 2.0]])


class SecgfdStageTests(unittest.TestCase):
    def test_run_secgfd_stage_uses_config_hparams_and_v2_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            repo_root = out_dir / "secgfd_repo"
            repo_root.mkdir(parents=True, exist_ok=True)
            _write_variant_csv(out_dir / "graph_variants.csv")

            cfg = {
                "seeds": {"training_seeds": [7]},
                "models": [
                    {
                        "model_id": "secgfd",
                        "repo_path": str(repo_root),
                        "hparams": {
                            "hid_dim": 24,
                            "order_d": 3,
                            "high_order": 2,
                            "lemda": 0.4,
                            "lr": 0.005,
                            "weight_decay": 0.01,
                        },
                    }
                ],
            }

            with mock.patch("benchmark.secgfd_stage.load_graph_bin", return_value=object()):
                with mock.patch(
                    "benchmark.secgfd_stage.train_eval_secgfd",
                    return_value={
                        "roc_auc": 0.9,
                        "average_precision": 0.8,
                        "f1_macro": 0.7,
                        "threshold": 0.5,
                        "duration_sec": 1.2,
                    },
                ) as train_mock:
                    with mock.patch("benchmark.secgfd_stage.summarize_results_by_training_seed"):
                        run_secgfd_stage(
                            cfg,
                            out_dir=out_dir,
                            force=False,
                            skip_existing=True,
                            retry_errors=False,
                            device="cpu",
                            include_noop=True,
                            only_clean=False,
                            max_variants=None,
                            max_training_seeds=None,
                            max_epochs=None,
                            patience=None,
                        )

            train_kwargs = train_mock.call_args.kwargs
            self.assertEqual(train_kwargs["hid_dim"], 24)
            self.assertEqual(train_kwargs["order_d"], 3)
            self.assertEqual(train_kwargs["high_order"], 2)
            self.assertEqual(train_kwargs["lemda"], 0.4)
            self.assertEqual(train_kwargs["lr"], 0.005)
            self.assertEqual(train_kwargs["weight_decay"], 0.01)
            self.assertEqual(train_kwargs["max_epochs"], 50)
            self.assertEqual(train_kwargs["patience"], 10)

    def test_run_secgfd_stage_cli_overrides_win_over_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            repo_root = out_dir / "secgfd_repo"
            repo_root.mkdir(parents=True, exist_ok=True)
            _write_variant_csv(out_dir / "graph_variants.csv")

            cfg = {
                "seeds": {"training_seeds": [7]},
                "models": [
                    {
                        "model_id": "secgfd",
                        "repo_path": str(repo_root),
                        "hparams": {"hid_dim": 24, "order_d": 3, "high_order": 2},
                    }
                ],
            }

            with mock.patch("benchmark.secgfd_stage.load_graph_bin", return_value=object()):
                with mock.patch(
                    "benchmark.secgfd_stage.train_eval_secgfd",
                    return_value={
                        "roc_auc": 0.9,
                        "average_precision": 0.8,
                        "f1_macro": 0.7,
                        "threshold": 0.5,
                        "duration_sec": 1.2,
                    },
                ) as train_mock:
                    with mock.patch("benchmark.secgfd_stage.summarize_results_by_training_seed"):
                        run_secgfd_stage(
                            cfg,
                            out_dir=out_dir,
                            force=False,
                            skip_existing=True,
                            retry_errors=False,
                            device="cpu",
                            include_noop=True,
                            only_clean=False,
                            max_variants=None,
                            max_training_seeds=None,
                            max_epochs=12,
                            patience=3,
                            secgfd_hid_dim=16,
                            secgfd_order_d=4,
                            secgfd_high_order=1,
                        )

            train_kwargs = train_mock.call_args.kwargs
            self.assertEqual(train_kwargs["hid_dim"], 16)
            self.assertEqual(train_kwargs["order_d"], 4)
            self.assertEqual(train_kwargs["high_order"], 1)
            self.assertEqual(train_kwargs["max_epochs"], 12)
            self.assertEqual(train_kwargs["patience"], 3)


class RunCliTests(unittest.TestCase):
    def test_run_main_passes_secgfd_override_flags(self) -> None:
        cfg = {"experiment_name": "exp"}

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("benchmark.run.load_json", return_value=cfg):
                with mock.patch("benchmark.run.validate_config"):
                    with mock.patch("benchmark.secgfd_stage.run_secgfd_stage") as stage_mock:
                        rc = main(
                            [
                                "--config",
                                "dummy.json",
                                "--out",
                                tmpdir,
                                "--stage",
                                "secgfd",
                                "--secgfd-hid-dim",
                                "16",
                                "--secgfd-high-order",
                                "1",
                                "--secgfd-order-d",
                                "4",
                            ]
                        )

        self.assertEqual(rc, 0)
        self.assertEqual(stage_mock.call_args.kwargs["secgfd_hid_dim"], 16)
        self.assertEqual(stage_mock.call_args.kwargs["secgfd_high_order"], 1)
        self.assertEqual(stage_mock.call_args.kwargs["secgfd_order_d"], 4)


if __name__ == "__main__":
    unittest.main()
