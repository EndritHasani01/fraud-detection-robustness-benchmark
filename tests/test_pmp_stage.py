from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmark.pmp_stage import _resolve_pmp_num_workers, resolve_pmp_config, run_pmp_stage
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


class PmpConfigTests(unittest.TestCase):
    def test_resolve_pmp_config_shallow_merges_benchmark_hparams(self) -> None:
        base_cfg = {
            "full_neighbors": True,
            "sampled_neighbors": [-1],
            "batch_size": 512,
            "num_workers": 8,
            "epochs": 500,
            "patience": 100,
        }
        model_cfg = {
            "hparams": {
                "full_neighbors": False,
                "sampled_neighbors": [10, 5],
                "batch_size": 256,
                "num_workers": 2,
                "epochs": 25,
                "patience": 8,
            }
        }

        with mock.patch("benchmark.pmp_stage._load_pmp_yaml_config", return_value=base_cfg):
            resolved = resolve_pmp_config(
                Path("Repos/PMP-master"),
                dataset_source_name="yelp",
                model_cfg=model_cfg,
            )

        self.assertTrue(base_cfg["full_neighbors"])
        self.assertEqual(base_cfg["batch_size"], 512)
        self.assertFalse(resolved["full_neighbors"])
        self.assertEqual(resolved["sampled_neighbors"], [10, 5])
        self.assertEqual(resolved["batch_size"], 256)
        self.assertEqual(resolved["num_workers"], 2)
        self.assertEqual(resolved["epochs"], 25)
        self.assertEqual(resolved["patience"], 8)

    def test_resolve_pmp_num_workers_is_platform_aware(self) -> None:
        with mock.patch("benchmark.pmp_stage.sys.platform", "win32"):
            self.assertEqual(_resolve_pmp_num_workers({"num_workers": 4}), 0)

        with mock.patch("benchmark.pmp_stage.sys.platform", "linux"):
            self.assertEqual(_resolve_pmp_num_workers({"num_workers": 4}), 4)
            self.assertEqual(_resolve_pmp_num_workers({}), 0)


class PmpStageTests(unittest.TestCase):
    def test_run_pmp_stage_merges_model_hparams_into_yaml_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            repo_root = out_dir / "pmp_repo"
            repo_root.mkdir(parents=True, exist_ok=True)
            _write_variant_csv(
                out_dir / "graph_variants.csv",
                rows=[_variant_row(graph_path="graph.bin", base_graph_path="base.bin", scenario_id="clean")],
            )

            cfg = {
                "datasets": [{"dataset_id": "yelpchi", "source_name": "yelp"}],
                "seeds": {"training_seeds": [7]},
                "models": [
                    {
                        "model_id": "pmp",
                        "repo_path": str(repo_root),
                        "hparams": {
                            "full_neighbors": False,
                            "sampled_neighbors": [10, 5],
                            "batch_size": 256,
                            "num_workers": 2,
                            "epochs": 25,
                            "patience": 8,
                        },
                    }
                ],
            }
            yaml_cfg = {
                "full_neighbors": True,
                "sampled_neighbors": [-1],
                "batch_size": 512,
                "num_workers": 8,
                "epochs": 500,
                "patience": 100,
            }

            with mock.patch("benchmark.pmp_stage._load_pmp_yaml_config", return_value=yaml_cfg):
                with mock.patch("benchmark.pmp_stage.load_graph_bin", return_value=object()):
                    with mock.patch(
                        "benchmark.pmp_stage.train_eval_pmp",
                        return_value={
                            "roc_auc": 0.9,
                            "average_precision": 0.8,
                            "f1_macro": 0.7,
                            "threshold": 0.5,
                            "duration_sec": 1.2,
                        },
                    ) as train_mock:
                        with mock.patch("benchmark.pmp_stage.summarize_results_by_training_seed"):
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
                            )

            train_kwargs = train_mock.call_args.kwargs
            cfg_pmp = train_kwargs["cfg_pmp"]
            self.assertFalse(cfg_pmp["full_neighbors"])
            self.assertEqual(cfg_pmp["sampled_neighbors"], [10, 5])
            self.assertEqual(cfg_pmp["batch_size"], 256)
            self.assertEqual(cfg_pmp["num_workers"], 2)
            self.assertEqual(cfg_pmp["epochs"], 25)
            self.assertEqual(cfg_pmp["patience"], 8)
            self.assertIsNone(train_kwargs["max_epochs"])
            self.assertIsNone(train_kwargs["patience"])

    def test_run_pmp_stage_cli_epoch_overrides_still_pass_through(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            repo_root = out_dir / "pmp_repo"
            repo_root.mkdir(parents=True, exist_ok=True)
            _write_variant_csv(
                out_dir / "graph_variants.csv",
                rows=[_variant_row(graph_path="graph.bin", base_graph_path="base.bin", scenario_id="clean")],
            )

            cfg = {
                "datasets": [{"dataset_id": "yelpchi", "source_name": "yelp"}],
                "seeds": {"training_seeds": [7]},
                "models": [
                    {
                        "model_id": "pmp",
                        "repo_path": str(repo_root),
                        "hparams": {"epochs": 25, "patience": 8},
                    }
                ],
            }

            with mock.patch("benchmark.pmp_stage._load_pmp_yaml_config", return_value={"epochs": 500, "patience": 100}):
                with mock.patch("benchmark.pmp_stage.load_graph_bin", return_value=object()):
                    with mock.patch(
                        "benchmark.pmp_stage.train_eval_pmp",
                        return_value={
                            "roc_auc": 0.9,
                            "average_precision": 0.8,
                            "f1_macro": 0.7,
                            "threshold": 0.5,
                            "duration_sec": 1.2,
                        },
                    ) as train_mock:
                        with mock.patch("benchmark.pmp_stage.summarize_results_by_training_seed"):
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
                                max_epochs=12,
                                patience=3,
                            )

            train_kwargs = train_mock.call_args.kwargs
            self.assertEqual(train_kwargs["cfg_pmp"]["epochs"], 25)
            self.assertEqual(train_kwargs["cfg_pmp"]["patience"], 8)
            self.assertEqual(train_kwargs["max_epochs"], 12)
            self.assertEqual(train_kwargs["patience"], 3)


class PmpShiftStageTests(unittest.TestCase):
    def test_shift_stage_uses_resolved_pmp_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            repo_root = out_dir / "pmp_repo"
            repo_root.mkdir(parents=True, exist_ok=True)
            clean_path = out_dir / "graphs" / "clean.bin"
            _write_variant_csv(
                out_dir / "graph_variants.csv",
                rows=[_variant_row(graph_path=str(clean_path), base_graph_path=str(clean_path), scenario_id="clean")],
            )

            cfg = {
                "datasets": [{"dataset_id": "yelpchi", "source_name": "yelp"}],
                "seeds": {"training_seeds": [7]},
                "models": [{"model_id": "pmp", "repo_path": str(repo_root), "hparams": {"full_neighbors": False}}],
            }
            resolved_cfg = {"full_neighbors": False, "sampled_neighbors": [10], "epochs": 25}

            with mock.patch("benchmark.shift_stage.load_graph_bin", return_value=object()):
                with mock.patch("benchmark.shift_stage.resolve_pmp_config", return_value=resolved_cfg) as cfg_mock:
                    with mock.patch("benchmark.shift_stage.train_pmp_model", return_value="artifact") as train_mock:
                        with mock.patch(
                            "benchmark.shift_stage.eval_pmp_model",
                            return_value={
                                "roc_auc": 0.9,
                                "average_precision": 0.8,
                                "f1_macro": 0.7,
                                "threshold": 0.5,
                                "duration_sec": 1.2,
                            },
                        ):
                            with mock.patch("benchmark.shift_stage.summarize_results_by_training_seed"):
                                run_shift_stage(
                                    cfg,
                                    out_dir=out_dir,
                                    force=False,
                                    skip_existing=True,
                                    retry_errors=False,
                                    device="cpu",
                                    include_noop=True,
                                    only_clean=True,
                                    max_variants=None,
                                    max_training_seeds=None,
                                    max_epochs=None,
                                    patience=None,
                                )

            cfg_mock.assert_called_once()
            self.assertEqual(train_mock.call_args.kwargs["cfg_pmp"], resolved_cfg)


if __name__ == "__main__":
    unittest.main()
