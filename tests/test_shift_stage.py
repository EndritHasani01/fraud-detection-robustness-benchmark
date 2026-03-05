from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmark.matrix_stage import run_matrix_stage
from benchmark.plots_stage import _completeness_report
from benchmark.results import (
    PROTOCOL_TRAIN_CLEAN_EVAL_ALL,
    append_result_row,
    ensure_results_csv,
)
from benchmark.shift_stage import run_shift_stage


def _variant_row(
    *,
    graph_path: str,
    base_graph_path: str,
    graph_seed: int = 0,
    scenario_id: str,
    severity: float = 0.0,
) -> dict[str, object]:
    return {
        "experiment_name": "exp",
        "dataset_id": "yelpchi",
        "split_id": "split_0",
        "graph_seed": graph_seed,
        "scenario_id": scenario_id,
        "severity": severity,
        "oracle_labels": False,
        "scenario_applied": True,
        "base_graph_path": base_graph_path,
        "graph_path": graph_path,
        "n_nodes": 10,
        "n_edges": 12,
        "mean_in_degree": 1.2,
        "median_in_degree": 1.0,
        "mean_out_degree": 1.2,
        "median_out_degree": 1.0,
        "heterophily_ratio": 0.3,
        "pos_rate": 0.2,
    }


def _write_variant_csv(path: Path, *, rows: list[dict[str, object]]) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class ShiftStageTests(unittest.TestCase):
    def test_shift_stage_trains_once_and_evaluates_all_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            clean_path = out_dir / "graphs" / "clean.bin"
            pert_path = out_dir / "graphs" / "noise.bin"
            _write_variant_csv(
                out_dir / "graph_variants.csv",
                rows=[
                    _variant_row(graph_path=str(clean_path), base_graph_path=str(clean_path), scenario_id="clean", severity=0.0),
                    _variant_row(
                        graph_path=str(pert_path),
                        base_graph_path=str(clean_path),
                        graph_seed=7,
                        scenario_id="noise_edges",
                        severity=0.2,
                    ),
                ],
            )

            cfg = {
                "experiment_name": "exp",
                "datasets": [{"dataset_id": "yelpchi", "source_name": "yelp"}],
                "models": [{"model_id": "mlp"}],
                "seeds": {"training_seeds": [42]},
            }

            def load_graph_side_effect(path: Path) -> str:
                return f"graph::{Path(path).name}"

            eval_outputs = [
                {
                    "roc_auc": 0.91,
                    "average_precision": 0.81,
                    "f1_macro": 0.71,
                    "threshold": 0.5,
                    "duration_sec": 0.25,
                },
                {
                    "roc_auc": 0.73,
                    "average_precision": 0.63,
                    "f1_macro": 0.53,
                    "threshold": 0.5,
                    "duration_sec": 0.15,
                },
            ]

            with mock.patch("benchmark.shift_stage.load_graph_bin", side_effect=load_graph_side_effect) as load_mock:
                with mock.patch("benchmark.shift_stage.train_baseline_model", return_value="artifact") as train_mock:
                    with mock.patch("benchmark.shift_stage.eval_baseline_model", side_effect=eval_outputs) as eval_mock:
                        with mock.patch("benchmark.shift_stage.summarize_results_by_training_seed") as summary_mock:
                            run_shift_stage(
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

            self.assertEqual(load_mock.call_count, 3)
            train_mock.assert_called_once()
            eval_mock.assert_has_calls(
                [
                    mock.call("artifact", "graph::clean.bin"),
                    mock.call("artifact", "graph::noise.bin"),
                ]
            )
            summary_mock.assert_called_once()
            summary_args, summary_kwargs = summary_mock.call_args
            self.assertEqual(Path(summary_args[0]).resolve(), (out_dir / "results.csv").resolve())
            self.assertEqual(Path(summary_kwargs["out_csv_path"]).resolve(), (out_dir / "results_summary_shift.csv").resolve())
            self.assertEqual(summary_kwargs["model_ids"], {"mlp"})

            rows = _read_csv_rows(out_dir / "results.csv")
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["protocol"] for row in rows}, {PROTOCOL_TRAIN_CLEAN_EVAL_ALL})
            self.assertEqual({row["train_graph_ref"] for row in rows}, {str(clean_path)})
            self.assertEqual({row["status"] for row in rows}, {"ok"})
            self.assertEqual({row["graph_path"] for row in rows}, {str(clean_path), str(pert_path)})


class MatrixProtocolDispatchTests(unittest.TestCase):
    def test_matrix_stage_runs_shift_stage_for_clean_train_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            (out_dir / "graph_variants.csv").write_text("dataset_id,split_id\n", encoding="utf-8")
            cfg = {
                "seeds": {"training_seeds": [0]},
                "models": [{"model_id": "mlp"}],
            }

            with mock.patch("benchmark.shift_stage.run_shift_stage") as shift_mock:
                with mock.patch("benchmark.baselines_stage.run_baselines_stage") as baselines_mock:
                    with mock.patch("benchmark.pmp_stage.run_pmp_stage") as pmp_mock:
                        with mock.patch("benchmark.secgfd_stage.run_secgfd_stage") as secgfd_mock:
                            run_matrix_stage(
                                cfg,
                                out_dir=out_dir,
                                force=False,
                                protocol=PROTOCOL_TRAIN_CLEAN_EVAL_ALL,
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

            shift_mock.assert_called_once()
            baselines_mock.assert_not_called()
            pmp_mock.assert_not_called()
            secgfd_mock.assert_not_called()


class PlotsProtocolCompletenessTests(unittest.TestCase):
    def test_completeness_report_uses_shift_protocol_when_only_shift_rows_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            clean_path = out_dir / "graphs" / "clean.bin"
            _write_variant_csv(
                out_dir / "graph_variants.csv",
                rows=[_variant_row(graph_path=str(clean_path), base_graph_path=str(clean_path), scenario_id="clean")],
            )

            results_csv = out_dir / "results.csv"
            ensure_results_csv(results_csv)
            append_result_row(
                results_csv,
                {
                    "experiment_name": "exp",
                    "dataset_id": "yelpchi",
                    "split_id": "split_0",
                    "graph_seed": 0,
                    "training_seed": 42,
                    "scenario_id": "clean",
                    "severity": 0.0,
                    "model_id": "mlp",
                    "protocol": PROTOCOL_TRAIN_CLEAN_EVAL_ALL,
                    "roc_auc": 0.9,
                    "average_precision": 0.8,
                    "f1_macro": 0.7,
                    "threshold": 0.5,
                    "duration_sec": 1.0,
                    "n_nodes": 10,
                    "n_edges": 12,
                    "mean_in_degree": 1.2,
                    "median_in_degree": 1.0,
                    "mean_out_degree": 1.2,
                    "median_out_degree": 1.0,
                    "heterophily_ratio": 0.3,
                    "pos_rate": 0.2,
                    "base_graph_path": str(clean_path),
                    "graph_path": str(clean_path),
                    "train_graph_ref": str(clean_path),
                    "status": "ok",
                    "error": "",
                },
            )

            cfg = {
                "models": [{"model_id": "mlp"}],
                "seeds": {"training_seeds": [42]},
            }

            report = _completeness_report(
                cfg,
                out_dir=out_dir,
                include_noop=True,
                only_clean=False,
                max_variants=None,
                max_training_seeds=None,
            )

            self.assertEqual(report.total_expected, 1)
            self.assertEqual(report.present_ok, 1)
            self.assertEqual(report.missing_count, 0)
            self.assertEqual(report.error_count, 0)
            report_rows = _read_csv_rows(out_dir / "plots" / "missing_or_error_runs.csv")
            self.assertEqual(report_rows, [])


if __name__ == "__main__":
    unittest.main()
