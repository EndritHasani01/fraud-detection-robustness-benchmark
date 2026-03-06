from __future__ import annotations

import contextlib
import csv
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from benchmark.plots_stage import run_plots_stage
from benchmark.results import append_result_row, ensure_results_csv


def _variant_row(
    *,
    split_id: str,
    graph_path: str,
    base_graph_path: str,
    scenario_id: str,
    severity: float,
    oracle_labels: bool,
    graph_seed: int = 0,
) -> dict[str, object]:
    return {
        "experiment_name": "exp",
        "dataset_id": "yelpchi",
        "split_id": split_id,
        "graph_seed": graph_seed,
        "scenario_id": scenario_id,
        "severity": severity,
        "oracle_labels": oracle_labels,
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


def _write_variant_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


class _FakeFigure:
    def tight_layout(self) -> None:
        return None

    def savefig(self, path: Path | str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("fake plot", encoding="utf-8")


class _FakeAxes:
    def __init__(self, title_sink: list[str]) -> None:
        self._title_sink = title_sink

    def errorbar(self, *args, **kwargs) -> None:
        return None

    def set_title(self, title: str) -> None:
        self._title_sink.append(str(title))

    def set_xlabel(self, _label: str) -> None:
        return None

    def set_ylabel(self, _label: str) -> None:
        return None

    def set_xticks(self, _ticks) -> None:
        return None

    def grid(self, *args, **kwargs) -> None:
        return None

    def legend(self, *args, **kwargs) -> None:
        return None


class _FakePyplot:
    def __init__(self) -> None:
        self.titles: list[str] = []

    def subplots(self, *args, **kwargs):
        return _FakeFigure(), _FakeAxes(self.titles)

    def close(self, _fig) -> None:
        return None


def _fake_matplotlib_modules() -> tuple[types.ModuleType, types.ModuleType, _FakePyplot]:
    fake_pyplot = _FakePyplot()
    matplotlib_mod = types.ModuleType("matplotlib")
    matplotlib_mod.use = lambda *_args, **_kwargs: None
    pyplot_mod = types.ModuleType("matplotlib.pyplot")
    pyplot_mod.subplots = fake_pyplot.subplots
    pyplot_mod.close = fake_pyplot.close
    setattr(matplotlib_mod, "pyplot", pyplot_mod)
    return matplotlib_mod, pyplot_mod, fake_pyplot


class PlotsStageTests(unittest.TestCase):
    def test_run_plots_stage_writes_oracle_and_cross_split_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            clean_0 = out_dir / "graphs" / "s0_clean.bin"
            noisy_0 = out_dir / "graphs" / "s0_hetero.bin"
            clean_1 = out_dir / "graphs" / "s1_clean.bin"
            noisy_1 = out_dir / "graphs" / "s1_hetero.bin"
            _write_variant_csv(
                out_dir / "graph_variants.csv",
                [
                    _variant_row(
                        split_id="s0",
                        graph_path=str(clean_0),
                        base_graph_path=str(clean_0),
                        scenario_id="clean",
                        severity=0.0,
                        oracle_labels=False,
                    ),
                    _variant_row(
                        split_id="s0",
                        graph_path=str(noisy_0),
                        base_graph_path=str(clean_0),
                        scenario_id="heterophily_rewire_oracle",
                        severity=0.3,
                        oracle_labels=True,
                    ),
                    _variant_row(
                        split_id="s1",
                        graph_path=str(clean_1),
                        base_graph_path=str(clean_1),
                        scenario_id="clean",
                        severity=0.0,
                        oracle_labels=False,
                    ),
                    _variant_row(
                        split_id="s1",
                        graph_path=str(noisy_1),
                        base_graph_path=str(clean_1),
                        scenario_id="heterophily_rewire_oracle",
                        severity=0.3,
                        oracle_labels=True,
                    ),
                ],
            )

            results_csv = out_dir / "results.csv"
            ensure_results_csv(results_csv)
            for split_id, clean_score, stressed_score in (("s0", 0.90, 0.70), ("s1", 0.80, 0.60)):
                append_result_row(
                    results_csv,
                    {
                        "experiment_name": "exp",
                        "dataset_id": "yelpchi",
                        "split_id": split_id,
                        "graph_seed": 0,
                        "training_seed": 0,
                        "scenario_id": "clean",
                        "severity": 0.0,
                        "model_id": "mlp",
                        "protocol": "train_on_variant",
                        "roc_auc": clean_score,
                        "average_precision": clean_score - 0.1,
                        "f1_macro": clean_score - 0.2,
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
                        "base_graph_path": str(clean_0 if split_id == "s0" else clean_1),
                        "graph_path": str(clean_0 if split_id == "s0" else clean_1),
                        "train_graph_ref": str(clean_0 if split_id == "s0" else clean_1),
                        "status": "ok",
                        "error": "",
                    },
                )
                append_result_row(
                    results_csv,
                    {
                        "experiment_name": "exp",
                        "dataset_id": "yelpchi",
                        "split_id": split_id,
                        "graph_seed": 0,
                        "training_seed": 0,
                        "scenario_id": "heterophily_rewire_oracle",
                        "severity": 0.3,
                        "model_id": "mlp",
                        "protocol": "train_on_variant",
                        "roc_auc": stressed_score,
                        "average_precision": stressed_score - 0.1,
                        "f1_macro": stressed_score - 0.2,
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
                        "base_graph_path": str(clean_0 if split_id == "s0" else clean_1),
                        "graph_path": str(noisy_0 if split_id == "s0" else noisy_1),
                        "train_graph_ref": str(noisy_0 if split_id == "s0" else noisy_1),
                        "status": "ok",
                        "error": "",
                    },
                )

            cfg = {
                "models": [{"model_id": "mlp"}],
                "seeds": {"training_seeds": [0]},
                "scenarios": [
                    {
                        "scenario_id": "heterophily_rewire_oracle",
                        "severity_values": [0.0, 0.3],
                        "oracle_labels": True,
                    }
                ],
            }

            fake_matplotlib, fake_pyplot_mod, fake_pyplot = _fake_matplotlib_modules()
            with mock.patch.dict(
                sys.modules,
                {"matplotlib": fake_matplotlib, "matplotlib.pyplot": fake_pyplot_mod},
            ):
                run_plots_stage(
                    cfg,
                    out_dir=out_dir,
                    include_noop=True,
                    only_clean=False,
                    max_variants=None,
                    max_training_seeds=None,
                    ci=True,
                )

            summary_rows = _read_csv_rows(out_dir / "plots" / "summary_curves.csv")
            oracle_rows = [row for row in summary_rows if row["scenario_id"] == "heterophily_rewire_oracle" and row["metric"] == "roc_auc"]
            self.assertTrue(oracle_rows)
            self.assertEqual({row["oracle_labels"] for row in oracle_rows}, {"True"})
            self.assertTrue(all("ci_lower" in row and "ci_upper" in row for row in oracle_rows))

            cross_rows = _read_csv_rows(out_dir / "plots" / "summary_curves_cross_split.csv")
            cross_row = next(
                row
                for row in cross_rows
                if row["scenario_id"] == "heterophily_rewire_oracle"
                and row["severity"] == "0.3"
                and row["metric"] == "roc_auc"
                and row["model_id"] == "mlp"
            )
            self.assertEqual(cross_row["oracle_labels"], "True")
            self.assertEqual(cross_row["n_splits"], "2")
            self.assertAlmostEqual(float(cross_row["mean"]), 0.65, places=6)

            drop_rows = _read_csv_rows(out_dir / "plots" / "performance_drop_max_stress.csv")
            self.assertEqual({row["oracle_labels"] for row in drop_rows}, {"True"})
            robust_rows = _read_csv_rows(out_dir / "plots" / "robustness_scores.csv")
            self.assertEqual({row["oracle_labels"] for row in robust_rows}, {"True"})
            self.assertTrue(any("(oracle)" in title for title in fake_pyplot.titles))

    def test_run_plots_stage_prints_completeness_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            clean_path = out_dir / "graphs" / "clean.bin"
            noise_path = out_dir / "graphs" / "noise.bin"
            camo_path = out_dir / "graphs" / "camo.bin"
            _write_variant_csv(
                out_dir / "graph_variants.csv",
                [
                    _variant_row(
                        split_id="s0",
                        graph_path=str(clean_path),
                        base_graph_path=str(clean_path),
                        scenario_id="clean",
                        severity=0.0,
                        oracle_labels=False,
                    ),
                    _variant_row(
                        split_id="s0",
                        graph_path=str(noise_path),
                        base_graph_path=str(clean_path),
                        scenario_id="noise_edges",
                        severity=0.2,
                        oracle_labels=False,
                    ),
                    _variant_row(
                        split_id="s0",
                        graph_path=str(camo_path),
                        base_graph_path=str(clean_path),
                        scenario_id="camouflage_feature_oracle",
                        severity=0.3,
                        oracle_labels=True,
                    ),
                ],
            )

            results_csv = out_dir / "results.csv"
            ensure_results_csv(results_csv)
            append_result_row(
                results_csv,
                {
                    "experiment_name": "exp",
                    "dataset_id": "yelpchi",
                    "split_id": "s0",
                    "graph_seed": 0,
                    "training_seed": 42,
                    "scenario_id": "clean",
                    "severity": 0.0,
                    "model_id": "mlp",
                    "protocol": "train_on_variant",
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
            append_result_row(
                results_csv,
                {
                    "experiment_name": "exp",
                    "dataset_id": "yelpchi",
                    "split_id": "s0",
                    "graph_seed": 0,
                    "training_seed": 42,
                    "scenario_id": "noise_edges",
                    "severity": 0.2,
                    "model_id": "mlp",
                    "protocol": "train_on_variant",
                    "roc_auc": "",
                    "average_precision": "",
                    "f1_macro": "",
                    "threshold": "",
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
                    "graph_path": str(noise_path),
                    "train_graph_ref": str(noise_path),
                    "status": "error",
                    "error": "boom",
                },
            )

            cfg = {
                "models": [{"model_id": "mlp"}],
                "seeds": {"training_seeds": [42]},
                "scenarios": [
                    {"scenario_id": "noise_edges", "severity_values": [0.0, 0.2], "oracle_labels": False},
                    {
                        "scenario_id": "camouflage_feature_oracle",
                        "severity_values": [0.0, 0.3],
                        "oracle_labels": True,
                    },
                ],
            }

            fake_matplotlib, fake_pyplot_mod, fake_pyplot = _fake_matplotlib_modules()
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                with mock.patch.dict(
                    sys.modules,
                    {"matplotlib": fake_matplotlib, "matplotlib.pyplot": fake_pyplot_mod},
                ):
                    run_plots_stage(
                        cfg,
                        out_dir=out_dir,
                        include_noop=True,
                        only_clean=False,
                        max_variants=None,
                        max_training_seeds=None,
                        ci=False,
                    )

            output = stdout.getvalue()
            self.assertIn("[plots] Completeness: 1/3 expected runs present (33%), 1 errors, 1 missing", output)
            self.assertIn("Missing runs: mlp/camouflage_feature_oracle/sev=0.3/gs=0/ts=42", output)
            self.assertIn("Error runs: mlp/noise_edges/sev=0.2/gs=0/ts=42", output)


if __name__ == "__main__":
    unittest.main()
