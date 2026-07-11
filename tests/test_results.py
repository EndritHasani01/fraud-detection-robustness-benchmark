from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from benchmark.baselines_stage import run_baselines_stage
from benchmark.metrics import best_f1_macro_threshold
from benchmark.results import (
    PROTOCOL_TRAIN_ON_VARIANT,
    RESULTS_COLUMN_DEFAULTS,
    RESULTS_COLUMNS,
    ensure_csv_header,
    ensure_results_csv,
    load_completed_keys,
    make_run_key,
    write_result_row,
)


class ResultsCsvTests(unittest.TestCase):
    def test_ensure_csv_header_migrates_v1_results_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            old_columns = [col for col in RESULTS_COLUMNS if col not in {"protocol", "train_graph_ref"}]

            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=old_columns)
                writer.writeheader()
                writer.writerow(
                    {
                        "experiment_name": "exp",
                        "dataset_id": "yelpchi",
                        "split_id": "split_0",
                        "graph_seed": 0,
                        "training_seed": 42,
                        "scenario_id": "clean",
                        "severity": 0.0,
                        "model_id": "mlp",
                        "roc_auc": 0.91,
                        "average_precision": 0.87,
                        "f1_macro": 0.66,
                        "threshold": 0.5,
                        "duration_sec": 1.25,
                        "n_nodes": 10,
                        "n_edges": 12,
                        "mean_in_degree": 1.2,
                        "median_in_degree": 1.0,
                        "mean_out_degree": 1.2,
                        "median_out_degree": 1.0,
                        "heterophily_ratio": 0.3,
                        "pos_rate": 0.2,
                        "base_graph_path": "base.bin",
                        "graph_path": "graph.bin",
                        "status": "ok",
                        "error": "",
                    }
                )

            ensure_csv_header(path, RESULTS_COLUMNS, row_defaults=RESULTS_COLUMN_DEFAULTS)

            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(reader.fieldnames, RESULTS_COLUMNS)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["protocol"], PROTOCOL_TRAIN_ON_VARIANT)
            self.assertEqual(rows[0]["train_graph_ref"], "")
            self.assertEqual(rows[0]["model_id"], "mlp")
            self.assertEqual(rows[0]["status"], "ok")

    def test_load_completed_keys_respects_retry_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            old_columns = [col for col in RESULTS_COLUMNS if col not in {"protocol", "train_graph_ref"}]

            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=old_columns)
                writer.writeheader()
                writer.writerow(
                    {
                        "experiment_name": "exp",
                        "dataset_id": "yelpchi",
                        "split_id": "split_0",
                        "graph_seed": 0,
                        "training_seed": 42,
                        "scenario_id": "clean",
                        "severity": 0.0,
                        "model_id": "mlp",
                        "status": "ok",
                        "error": "",
                    }
                )
                writer.writerow(
                    {
                        "experiment_name": "exp",
                        "dataset_id": "yelpchi",
                        "split_id": "split_0",
                        "graph_seed": 0,
                        "training_seed": 42,
                        "scenario_id": "clean",
                        "severity": 0.0,
                        "model_id": "sage",
                        "status": "error",
                        "error": "boom",
                    }
                )

            completed_default = load_completed_keys(path, retry_errors=False)
            completed_retry = load_completed_keys(path, retry_errors=True)

            ok_key = make_run_key(
                dataset_id="yelpchi",
                split_id="split_0",
                scenario_id="clean",
                severity=0.0,
                graph_seed=0,
                training_seed=42,
                model_id="mlp",
                protocol=PROTOCOL_TRAIN_ON_VARIANT,
            )
            error_key = make_run_key(
                dataset_id="yelpchi",
                split_id="split_0",
                scenario_id="clean",
                severity=0.0,
                graph_seed=0,
                training_seed=42,
                model_id="sage",
                protocol=PROTOCOL_TRAIN_ON_VARIANT,
            )

            self.assertIn(ok_key, completed_default)
            self.assertIn(ok_key, completed_retry)
            self.assertIn(error_key, completed_default)
            self.assertNotIn(error_key, completed_retry)

    def test_write_result_row_builds_consistent_variant_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            ensure_results_csv(path)

            variant = SimpleNamespace(
                experiment_name="exp",
                dataset_id="yelpchi",
                split_id="split_0",
                graph_seed=7,
                scenario_id="noise_edges",
                severity=0.2,
                n_nodes=10,
                n_edges=12,
                mean_in_degree=1.2,
                median_in_degree=1.0,
                mean_out_degree=1.2,
                median_out_degree=1.0,
                heterophily_ratio=0.3,
                pos_rate=0.2,
                base_graph_path="base.bin",
                graph_path="noise.bin",
            )

            write_result_row(
                path,
                variant,
                model_id="mlp",
                training_seed=42,
                protocol=PROTOCOL_TRAIN_ON_VARIANT,
                metrics={
                    "roc_auc": 0.91,
                    "average_precision": 0.87,
                    "f1_macro": 0.66,
                    "threshold": 0.5,
                    "duration_sec": 1.25,
                },
                train_graph_ref="clean.bin",
            )

            rows = self._read_csv_rows(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["dataset_id"], "yelpchi")
            self.assertEqual(rows[0]["graph_path"], "noise.bin")
            self.assertEqual(rows[0]["train_graph_ref"], "clean.bin")
            self.assertEqual(rows[0]["protocol"], PROTOCOL_TRAIN_ON_VARIANT)
            self.assertEqual(rows[0]["status"], "ok")

    def test_write_result_row_rejects_non_finite_success_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            ensure_results_csv(path)
            variant = SimpleNamespace(
                experiment_name="exp",
                dataset_id="yelpchi",
                split_id="s0",
                graph_seed=0,
                scenario_id="clean",
                severity=0.0,
                n_nodes=10,
                n_edges=12,
                mean_in_degree=1.2,
                median_in_degree=1.0,
                mean_out_degree=1.2,
                median_out_degree=1.0,
                heterophily_ratio=0.3,
                pos_rate=0.2,
                base_graph_path="base.bin",
                graph_path="base.bin",
            )

            with self.assertRaisesRegex(ValueError, "average_precision"):
                write_result_row(
                    path,
                    variant,
                    model_id="mlp",
                    training_seed=0,
                    protocol=PROTOCOL_TRAIN_ON_VARIANT,
                    metrics={
                        "roc_auc": 0.9,
                        "average_precision": math.nan,
                        "f1_macro": 0.7,
                        "threshold": 0.5,
                        "duration_sec": 1.0,
                    },
                )

            with self.assertRaisesRegex(ValueError, "threshold.*finite"):
                write_result_row(
                    path,
                    variant,
                    model_id="mlp",
                    training_seed=0,
                    protocol=PROTOCOL_TRAIN_ON_VARIANT,
                    metrics={
                        "roc_auc": 0.9,
                        "average_precision": 0.8,
                        "f1_macro": 0.7,
                        "threshold": math.inf,
                        "duration_sec": 1.0,
                    },
                )

            self.assertEqual(self._read_csv_rows(path), [])

    def test_all_negative_threshold_above_one_round_trips_to_results_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            ensure_results_csv(path)
            variant = SimpleNamespace(
                experiment_name="exp",
                dataset_id="yelpchi",
                split_id="s0",
                graph_seed=0,
                scenario_id="clean",
                severity=0.0,
                n_nodes=4,
                n_edges=0,
                mean_in_degree=0.0,
                median_in_degree=0.0,
                mean_out_degree=0.0,
                median_out_degree=0.0,
                heterophily_ratio=0.0,
                pos_rate=0.25,
                base_graph_path="base.bin",
                graph_path="base.bin",
            )
            threshold_result = best_f1_macro_threshold(
                [0, 0, 0, 1],
                [1.0, 0.8, 0.7, 0.1],
            )

            self.assertGreater(threshold_result.threshold, 1.0)
            write_result_row(
                path,
                variant,
                model_id="mlp",
                training_seed=0,
                protocol=PROTOCOL_TRAIN_ON_VARIANT,
                metrics={
                    "roc_auc": 0.0,
                    "average_precision": 0.25,
                    "f1_macro": threshold_result.f1_macro,
                    "threshold": threshold_result.threshold,
                    "duration_sec": 1.0,
                },
            )

            rows = self._read_csv_rows(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "ok")
            self.assertEqual(float(rows[0]["threshold"]), threshold_result.threshold)

    def test_baselines_stage_skips_completed_rows_before_graph_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            self._write_variant_csv(out_dir / "graph_variants.csv")
            self._write_results_csv(
                out_dir / "results.csv",
                [
                    {
                        "experiment_name": "exp",
                        "dataset_id": "yelpchi",
                        "split_id": "split_0",
                        "graph_seed": 0,
                        "training_seed": 42,
                        "scenario_id": "clean",
                        "severity": 0.0,
                        "model_id": "mlp",
                        "status": "ok",
                        "error": "",
                    }
                ],
            )

            cfg = {
                "seeds": {"training_seeds": [42]},
                "models": [{"model_id": "mlp"}],
            }

            with mock.patch("benchmark.baselines_stage.load_graph_bin", side_effect=AssertionError("should skip")):
                with mock.patch("benchmark.baselines_stage.train_eval_baseline") as train_mock:
                    with mock.patch("benchmark.baselines_stage.summarize_results_by_training_seed"):
                        run_baselines_stage(
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
                        train_mock.assert_not_called()

            rows = self._read_csv_rows(out_dir / "results.csv")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "ok")
            self.assertEqual(rows[0]["protocol"], PROTOCOL_TRAIN_ON_VARIANT)

    def test_baselines_stage_retries_error_rows_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            self._write_variant_csv(out_dir / "graph_variants.csv")
            self._write_results_csv(
                out_dir / "results.csv",
                [
                    {
                        "experiment_name": "exp",
                        "dataset_id": "yelpchi",
                        "split_id": "split_0",
                        "graph_seed": 0,
                        "training_seed": 42,
                        "scenario_id": "clean",
                        "severity": 0.0,
                        "model_id": "mlp",
                        "status": "error",
                        "error": "boom",
                    }
                ],
            )

            cfg = {
                "seeds": {"training_seeds": [42]},
                "models": [{"model_id": "mlp"}],
            }

            with mock.patch("benchmark.baselines_stage.load_graph_bin", return_value=object()) as load_mock:
                with mock.patch(
                    "benchmark.baselines_stage.train_eval_baseline",
                    return_value={
                        "roc_auc": 0.9,
                        "average_precision": 0.8,
                        "f1_macro": 0.7,
                        "threshold": 0.5,
                        "duration_sec": 1.2,
                    },
                ) as train_mock:
                    with mock.patch("benchmark.baselines_stage.summarize_results_by_training_seed"):
                        run_baselines_stage(
                            cfg,
                            out_dir=out_dir,
                            force=False,
                            skip_existing=True,
                            retry_errors=True,
                            device="cpu",
                            include_noop=True,
                            only_clean=False,
                            max_variants=None,
                            max_training_seeds=None,
                            max_epochs=None,
                            patience=None,
                        )

            load_mock.assert_called_once()
            train_mock.assert_called_once()

            rows = self._read_csv_rows(out_dir / "results.csv")
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[-1]["status"], "ok")
            self.assertEqual(rows[-1]["protocol"], PROTOCOL_TRAIN_ON_VARIANT)

    @staticmethod
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

    @staticmethod
    def _write_results_csv(path: Path, rows: list[dict[str, object]]) -> None:
        old_columns = [col for col in RESULTS_COLUMNS if col not in {"protocol", "train_graph_ref"}]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=old_columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    @staticmethod
    def _read_csv_rows(path: Path) -> list[dict[str, str]]:
        with path.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))


if __name__ == "__main__":
    unittest.main()
