from __future__ import annotations

import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmark.matrix_stage import run_matrix_stage
from benchmark.preflight import build_expected_run_keys, summarize_training_preflight
from benchmark.results import PROTOCOL_TRAIN_ON_VARIANT, append_result_row, ensure_results_csv
from benchmark.run import main


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


class RunModelsCliTests(unittest.TestCase):
    def test_run_main_passes_model_filter_to_stage(self) -> None:
        cfg = {"experiment_name": "exp"}

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("benchmark.run.load_json", return_value=cfg):
                with mock.patch("benchmark.run.validate_config"):
                    with mock.patch("benchmark.baselines_stage.run_baselines_stage") as stage_mock:
                        rc = main(
                            [
                                "--config",
                                "dummy.json",
                                "--out",
                                tmpdir,
                                "--stage",
                                "baselines",
                                "--models",
                                "mlp,sage",
                            ]
                        )

        self.assertEqual(rc, 0)
        self.assertEqual(stage_mock.call_args.kwargs["selected_model_ids"], ["mlp", "sage"])


class StageModelFilterTests(unittest.TestCase):
    def test_pmp_stage_warns_and_exits_when_models_filter_excludes_pmp(self) -> None:
        from benchmark.pmp_stage import run_pmp_stage

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            _write_variant_csv(out_dir / "graph_variants.csv")
            cfg = {
                "datasets": [{"dataset_id": "yelpchi", "source_name": "yelp"}],
                "seeds": {"training_seeds": [7]},
                "models": [{"model_id": "pmp", "repo_path": "Repos/PMP-master"}],
            }

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                with mock.patch("benchmark.pmp_stage.load_graph_bin", side_effect=AssertionError("should not load")):
                    with mock.patch("benchmark.pmp_stage.train_eval_pmp") as train_mock:
                        run_pmp_stage(
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
                            selected_model_ids=["sage"],
                        )

            train_mock.assert_not_called()
            self.assertIn("[warn] pmp", stdout.getvalue())

    def test_matrix_stage_filters_standard_dispatch_by_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            _write_variant_csv(out_dir / "graph_variants.csv")
            cfg = {
                "datasets": [{"dataset_id": "yelpchi"}],
                "seeds": {"training_seeds": [0]},
                "models": [
                    {"model_id": "mlp"},
                    {"model_id": "sage"},
                    {"model_id": "pmp"},
                    {"model_id": "secgfd"},
                ],
            }

            with mock.patch("benchmark.baselines_stage.run_baselines_stage") as baselines_mock:
                with mock.patch("benchmark.pmp_stage.run_pmp_stage") as pmp_mock:
                    with mock.patch("benchmark.secgfd_stage.run_secgfd_stage") as secgfd_mock:
                        run_matrix_stage(
                            cfg,
                            out_dir=out_dir,
                            force=False,
                            protocol=PROTOCOL_TRAIN_ON_VARIANT,
                            skip_existing=True,
                            retry_errors=False,
                            device="cpu",
                            include_noop=True,
                            only_clean=False,
                            max_variants=None,
                            max_training_seeds=None,
                            max_epochs=None,
                            patience=None,
                            selected_model_ids=["pmp"],
                        )

            baselines_mock.assert_not_called()
            pmp_mock.assert_called_once()
            secgfd_mock.assert_not_called()


class PreflightEstimatorTests(unittest.TestCase):
    def test_summarize_training_preflight_counts_done_and_estimates_remaining_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            results_csv = Path(tmpdir) / "results.csv"
            ensure_results_csv(results_csv)
            append_result_row(
                results_csv,
                {
                    "experiment_name": "exp",
                    "dataset_id": "yelpchi",
                    "split_id": "split_0",
                    "graph_seed": 0,
                    "training_seed": 0,
                    "scenario_id": "clean",
                    "severity": 0.0,
                    "model_id": "mlp",
                    "protocol": PROTOCOL_TRAIN_ON_VARIANT,
                    "duration_sec": 5.0,
                    "status": "ok",
                    "error": "",
                },
            )
            append_result_row(
                results_csv,
                {
                    "experiment_name": "exp",
                    "dataset_id": "yelpchi",
                    "split_id": "split_0",
                    "graph_seed": 1,
                    "training_seed": 0,
                    "scenario_id": "clean",
                    "severity": 0.0,
                    "model_id": "mlp",
                    "protocol": PROTOCOL_TRAIN_ON_VARIANT,
                    "duration_sec": 7.0,
                    "status": "ok",
                    "error": "",
                },
            )

            variants = [
                mock.Mock(
                    dataset_id="yelpchi",
                    split_id="split_0",
                    scenario_id="clean",
                    severity=0.0,
                    graph_seed=0,
                ),
                mock.Mock(
                    dataset_id="yelpchi",
                    split_id="split_0",
                    scenario_id="clean",
                    severity=0.0,
                    graph_seed=1,
                ),
                mock.Mock(
                    dataset_id="yelpchi",
                    split_id="split_0",
                    scenario_id="clean",
                    severity=0.0,
                    graph_seed=2,
                ),
            ]
            expected_keys = build_expected_run_keys(
                variants,
                training_seeds=[0],
                model_ids=["mlp"],
                protocol=PROTOCOL_TRAIN_ON_VARIANT,
            )
            completed_keys = {key for key in expected_keys if key[4] == 0}

            summary = summarize_training_preflight(
                results_csv,
                expected_keys=expected_keys,
                completed_keys=completed_keys,
                skip_existing=True,
                model_ids=["mlp"],
                protocol=PROTOCOL_TRAIN_ON_VARIANT,
            )

        self.assertEqual(summary.total_runs, 3)
        self.assertEqual(summary.already_done, 1)
        self.assertEqual(summary.remaining_runs, 2)
        self.assertEqual(summary.medians_by_model["mlp"], 6.0)
        self.assertEqual(summary.estimated_remaining_sec, 12.0)


if __name__ == "__main__":
    unittest.main()
