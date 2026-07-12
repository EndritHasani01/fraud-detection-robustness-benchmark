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
from benchmark.scenario_audit import append_variant_audit_row, build_variant_audit_row, ensure_variant_audit_csv


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


def _audit_row(
    *,
    split_id: str,
    scenario_id: str,
    severity: float,
    graph_seed: int,
    oracle_labels: bool,
    base_graph_path: str,
    graph_path: str,
    scenario_family: str,
    scenario_method: str,
    severity_param: str,
    graph_view_mode: str = "canonical",
    base_stats: dict[str, object] | None = None,
    variant_stats: dict[str, object] | None = None,
    scenario_info: dict[str, object] | None = None,
) -> dict[str, object]:
    return build_variant_audit_row(
        experiment_name="exp",
        dataset_id="yelpchi",
        split_id=split_id,
        scenario_id=scenario_id,
        severity=severity,
        graph_seed=graph_seed,
        oracle_labels=oracle_labels,
        scenario_applied=(scenario_id == "clean" or float(severity) > 0.0),
        scenario_family=scenario_family,
        scenario_method=scenario_method,
        severity_param=severity_param,
        graph_view_mode=graph_view_mode,
        base_graph_path=base_graph_path,
        graph_path=graph_path,
        base_stats=base_stats or {"n_nodes": 10, "n_edges": 12, "heterophily_ratio": 0.2, "pos_rate": 0.2},
        variant_stats=variant_stats or {"n_nodes": 10, "n_edges": 12, "heterophily_ratio": 0.2, "pos_rate": 0.2},
        scenario_info=scenario_info or {},
    )


def _write_variant_audit_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_variant_audit_csv(path, overwrite=True)
    for row in rows:
        append_variant_audit_row(path, row)


def _append_result(
    path: Path,
    *,
    scenario_id: str,
    severity: float,
    graph_seed: int,
    training_seed: int,
    protocol: str,
    roc_auc: float,
    average_precision: float,
    f1_macro: float,
) -> None:
    append_result_row(
        path,
        {
            "experiment_name": "exp",
            "dataset_id": "yelpchi",
            "split_id": "s0",
            "graph_seed": graph_seed,
            "training_seed": training_seed,
            "scenario_id": scenario_id,
            "severity": severity,
            "model_id": "mlp",
            "protocol": protocol,
            "roc_auc": roc_auc,
            "average_precision": average_precision,
            "f1_macro": f1_macro,
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
            "base_graph_path": "clean.bin",
            "graph_path": "clean.bin" if scenario_id == "clean" else "stress.bin",
            "train_graph_ref": "clean.bin" if protocol == "train_clean_eval_all" else "stress.bin",
            "status": "ok",
            "error": "",
        },
    )


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
            _write_variant_audit_csv(
                out_dir / "variant_audit.csv",
                [
                    _audit_row(
                        split_id="s0",
                        scenario_id="clean",
                        severity=0.0,
                        graph_seed=0,
                        oracle_labels=False,
                        base_graph_path=str(clean_0),
                        graph_path=str(clean_0),
                        scenario_family="clean",
                        scenario_method="clean",
                        severity_param="severity",
                    ),
                    _audit_row(
                        split_id="s0",
                        scenario_id="heterophily_rewire_oracle",
                        severity=0.3,
                        graph_seed=0,
                        oracle_labels=True,
                        base_graph_path=str(clean_0),
                        graph_path=str(noisy_0),
                        scenario_family="heterophily",
                        scenario_method="rewire_edge_dst_to_opposite_label",
                        severity_param="p_rewire",
                        graph_view_mode="heterograph_aware_generation",
                        base_stats={"n_nodes": 10, "n_edges": 12, "heterophily_ratio": 0.2, "pos_rate": 0.2},
                        variant_stats={"n_nodes": 10, "n_edges": 12, "heterophily_ratio": 0.5, "pos_rate": 0.2},
                        scenario_info={
                            "n_rewired_edges_selected": 4,
                            "n_rewired_edges_actual": 4,
                            "heterophily_ratio_before": 0.2,
                            "heterophily_ratio_after": 0.5,
                        },
                    ),
                    _audit_row(
                        split_id="s1",
                        scenario_id="clean",
                        severity=0.0,
                        graph_seed=0,
                        oracle_labels=False,
                        base_graph_path=str(clean_1),
                        graph_path=str(clean_1),
                        scenario_family="clean",
                        scenario_method="clean",
                        severity_param="severity",
                    ),
                    _audit_row(
                        split_id="s1",
                        scenario_id="heterophily_rewire_oracle",
                        severity=0.3,
                        graph_seed=0,
                        oracle_labels=True,
                        base_graph_path=str(clean_1),
                        graph_path=str(noisy_1),
                        scenario_family="heterophily",
                        scenario_method="rewire_edge_dst_to_opposite_label",
                        severity_param="p_rewire",
                        graph_view_mode="heterograph_aware_generation",
                        base_stats={"n_nodes": 10, "n_edges": 12, "heterophily_ratio": 0.25, "pos_rate": 0.2},
                        variant_stats={"n_nodes": 10, "n_edges": 12, "heterophily_ratio": 0.55, "pos_rate": 0.2},
                        scenario_info={
                            "n_rewired_edges_selected": 4,
                            "n_rewired_edges_actual": 4,
                            "heterophily_ratio_before": 0.25,
                            "heterophily_ratio_after": 0.55,
                        },
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
                        "oracle_mode": "oracle",
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
            self.assertEqual(
                {row["claim_scope"] for row in oracle_rows},
                {"oracle_privileged_training_diagnostic"},
            )
            self.assertEqual({row["operational_ranking_eligible"] for row in oracle_rows}, {"False"})
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
            joined_rows = _read_csv_rows(out_dir / "plots" / "performance_audit_join.csv")
            joined_protocols = {row["protocol"] for row in joined_rows if row["scenario_id"] == "heterophily_rewire_oracle"}
            self.assertEqual(joined_protocols, {"train_on_variant"})
            joined_row = next(
                row
                for row in joined_rows
                if row["split_id"] == "s0"
                and row["scenario_id"] == "heterophily_rewire_oracle"
                and row["status"] == "ok"
            )
            self.assertEqual(joined_row["graph_view_mode"], "heterograph_aware_generation")
            self.assertEqual(float(joined_row["requested_change"]), 4.0)
            self.assertEqual(float(joined_row["realized_change"]), 4.0)
            audit_rows = _read_csv_rows(out_dir / "plots" / "audit_curves.csv")
            hetero_audit = next(
                row
                for row in audit_rows
                if row["split_id"] == "s0"
                and row["scenario_id"] == "heterophily_rewire_oracle"
                and row["protocol"] == "train_on_variant"
                and row["audit_metric"] == "heterophily_ratio_after"
            )
            self.assertEqual(hetero_audit["oracle_labels"], "True")
            self.assertEqual(hetero_audit["graph_view_mode"], "heterograph_aware_generation")
            self.assertAlmostEqual(float(hetero_audit["mean"]), 0.5, places=6)
            audit_cross_rows = _read_csv_rows(out_dir / "plots" / "audit_curves_cross_split.csv")
            audit_cross_row = next(
                row
                for row in audit_cross_rows
                if row["scenario_id"] == "heterophily_rewire_oracle"
                and row["protocol"] == "train_on_variant"
                and row["audit_metric"] == "heterophily_ratio_after"
                and row["severity"] == "0.3"
            )
            self.assertAlmostEqual(float(audit_cross_row["mean"]), 0.525, places=6)
            self.assertTrue((out_dir / "plots" / "train_on_variant" / "yelpchi" / "s0" / "audit__heterophily_rewire_oracle__heterophily_ratio_after.png").exists())
            self.assertTrue(any("(oracle)" in title for title in fake_pyplot.titles))

    def test_seed_aware_reporting_preserves_pairing_and_exports_research_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            clean_path = out_dir / "graphs" / "clean.bin"
            variant_rows = [
                _variant_row(
                    split_id="s0",
                    graph_path=str(clean_path),
                    base_graph_path=str(clean_path),
                    scenario_id="clean",
                    severity=0.0,
                    oracle_labels=False,
                    graph_seed=717,
                )
            ]
            audit_rows = [
                _audit_row(
                    split_id="s0",
                    scenario_id="clean",
                    severity=0.0,
                    graph_seed=717,
                    oracle_labels=False,
                    base_graph_path=str(clean_path),
                    graph_path=str(clean_path),
                    scenario_family="clean",
                    scenario_method="clean",
                    severity_param="severity",
                )
            ]
            for severity in (0.5, 1.0):
                for graph_seed in (10, 20):
                    stress_path = out_dir / "graphs" / f"stress_{severity}_{graph_seed}.bin"
                    variant_rows.append(
                        _variant_row(
                            split_id="s0",
                            graph_path=str(stress_path),
                            base_graph_path=str(clean_path),
                            scenario_id="noise_edges_uniform",
                            severity=severity,
                            oracle_labels=False,
                            graph_seed=graph_seed,
                        )
                    )
                    audit_rows.append(
                        _audit_row(
                            split_id="s0",
                            scenario_id="noise_edges_uniform",
                            severity=severity,
                            graph_seed=graph_seed,
                            oracle_labels=False,
                            base_graph_path=str(clean_path),
                            graph_path=str(stress_path),
                            scenario_family="noise",
                            scenario_method="add_random_edges",
                            severity_param="edge_noise_rate",
                            graph_view_mode="heterograph_aware_generation",
                            scenario_info={"n_added_edge_pairs_actual": int(10 * severity)},
                        )
                    )

            _write_variant_csv(out_dir / "graph_variants.csv", variant_rows)
            _write_variant_audit_csv(out_dir / "variant_audit.csv", audit_rows)

            results_csv = out_dir / "results.csv"
            ensure_results_csv(results_csv)
            clean_metrics = {
                1: (0.2, 0.3, 0.4),
                2: (0.8, 0.7, 0.6),
            }
            protocols = ("train_on_variant", "train_clean_eval_all")
            for protocol in protocols:
                for training_seed, (roc_auc, average_precision, f1_macro) in clean_metrics.items():
                    _append_result(
                        results_csv,
                        scenario_id="clean",
                        severity=0.0,
                        graph_seed=717,
                        training_seed=training_seed,
                        protocol=protocol,
                        roc_auc=roc_auc,
                        average_precision=average_precision,
                        f1_macro=f1_macro,
                    )

                for severity in (0.5, 1.0):
                    for graph_seed in (10, 20):
                        for training_seed, (_clean_roc, clean_ap, clean_f1) in clean_metrics.items():
                            # Each paired ROC trajectory has AUC 0.5 even though
                            # individual seed levels differ substantially.
                            roc_auc = 0.5 if severity == 0.5 else (0.8 if training_seed == 1 else 0.2)
                            f1_macro = clean_f1 + (0.1 if protocol == "train_on_variant" else 0.0)
                            _append_result(
                                results_csv,
                                scenario_id="noise_edges_uniform",
                                severity=severity,
                                graph_seed=graph_seed,
                                training_seed=training_seed,
                                protocol=protocol,
                                roc_auc=roc_auc,
                                # Exact per-training-seed graph invariant.
                                average_precision=clean_ap,
                                f1_macro=f1_macro,
                            )

            cfg = {
                "models": [{"model_id": "mlp"}],
                "seeds": {"training_seeds": [1, 2]},
                "scenarios": [
                    {
                        "scenario_id": "noise_edges_uniform",
                        "severity_values": [0.0, 0.5, 1.0],
                        "oracle_mode": "non_oracle",
                    }
                ],
                "evaluation": {"audit_metrics": ["n_added_edge_pairs_actual"]},
            }

            fake_matplotlib, fake_pyplot_mod, _fake_pyplot = _fake_matplotlib_modules()
            with mock.patch.dict(
                sys.modules,
                {"matplotlib": fake_matplotlib, "matplotlib.pyplot": fake_pyplot_mod},
            ):
                run_plots_stage(
                    cfg,
                    out_dir=out_dir,
                    include_noop=False,
                    only_clean=False,
                    max_variants=None,
                    max_training_seeds=None,
                    ci=True,
                )

            summary_rows = _read_csv_rows(out_dir / "plots" / "summary_curves.csv")
            stress_summary = next(
                row
                for row in summary_rows
                if row["protocol"] == "train_clean_eval_all"
                and row["metric"] == "average_precision"
                and row["severity"] == "1.0"
            )
            self.assertEqual(stress_summary["n_runs"], "4")
            self.assertEqual(stress_summary["n_training_seeds"], "2")
            self.assertEqual(stress_summary["n_graph_seeds"], "2")
            self.assertEqual(stress_summary["n_seed_cells"], "4")
            self.assertEqual(stress_summary["ci_method"], "crossed_training_graph_seed_bootstrap")

            drop_rows = _read_csv_rows(out_dir / "plots" / "performance_drop_max_stress.csv")
            invariant_drop = next(
                row
                for row in drop_rows
                if row["protocol"] == "train_clean_eval_all"
                and row["metric"] == "average_precision"
            )
            for field in ("drop_mean", "drop_std", "drop_ci_lower", "drop_ci_upper"):
                self.assertAlmostEqual(float(invariant_drop[field]), 0.0, places=12)
            self.assertEqual(invariant_drop["n_paired_training_seeds"], "2")
            self.assertEqual(invariant_drop["n_graph_seeds"], "2")
            self.assertEqual(
                invariant_drop["ci_method"],
                "paired_drop_crossed_training_graph_seed_bootstrap",
            )

            robust_rows = _read_csv_rows(out_dir / "plots" / "robustness_scores.csv")
            paired_curve = next(
                row
                for row in robust_rows
                if row["protocol"] == "train_clean_eval_all" and row["metric"] == "roc_auc"
            )
            self.assertAlmostEqual(float(paired_curve["robustness_auc_mean"]), 0.5, places=12)
            self.assertAlmostEqual(float(paired_curve["robustness_auc_std"]), 0.0, places=12)
            self.assertAlmostEqual(float(paired_curve["robustness_auc_ci_lower"]), 0.5, places=12)
            self.assertAlmostEqual(float(paired_curve["robustness_auc_ci_upper"]), 0.5, places=12)
            self.assertEqual(paired_curve["n_paired_seed_cells"], "4")
            self.assertEqual(
                paired_curve["ci_method"],
                "paired_curve_crossed_training_graph_seed_bootstrap",
            )

            contrast_rows = _read_csv_rows(out_dir / "plots" / "protocol_contrasts.csv")
            protocol_contrast = next(
                row
                for row in contrast_rows
                if row["metric"] == "f1_macro" and row["severity"] == "1.0"
            )
            self.assertAlmostEqual(float(protocol_contrast["contrast_mean"]), 0.1, places=12)
            self.assertAlmostEqual(float(protocol_contrast["contrast_ci_lower"]), 0.1, places=12)
            self.assertAlmostEqual(float(protocol_contrast["contrast_ci_upper"]), 0.1, places=12)
            self.assertEqual(protocol_contrast["n_pairs"], "4")
            self.assertEqual(protocol_contrast["n_training_seeds"], "2")
            self.assertEqual(protocol_contrast["n_graph_seeds"], "2")
            self.assertEqual(protocol_contrast["claim_scope"], "non_oracle_controlled_stress")
            self.assertEqual(protocol_contrast["operational_ranking_eligible"], "True")

            worst_rows = _read_csv_rows(out_dir / "plots" / "worst_case_performance.csv")
            self.assertIn("retention_fraction_mean", worst_rows[0])
            self.assertIn("retention_ci_method", worst_rows[0])
            self.assertTrue(worst_rows)
            self.assertTrue(
                all(
                    row["selection_method"] == "minimum_configured_nonzero_point_mean"
                    for row in worst_rows
                )
            )

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
            _write_variant_audit_csv(
                out_dir / "variant_audit.csv",
                [
                    _audit_row(
                        split_id="s0",
                        scenario_id="clean",
                        severity=0.0,
                        graph_seed=0,
                        oracle_labels=False,
                        base_graph_path=str(clean_path),
                        graph_path=str(clean_path),
                        scenario_family="clean",
                        scenario_method="clean",
                        severity_param="severity",
                    ),
                    _audit_row(
                        split_id="s0",
                        scenario_id="noise_edges",
                        severity=0.2,
                        graph_seed=0,
                        oracle_labels=False,
                        base_graph_path=str(clean_path),
                        graph_path=str(noise_path),
                        scenario_family="noise",
                        scenario_method="add_random_edges",
                        severity_param="edge_noise_rate",
                        graph_view_mode="heterograph_aware_generation",
                        base_stats={"n_nodes": 10, "n_edges": 12, "heterophily_ratio": 0.3, "pos_rate": 0.2},
                        variant_stats={"n_nodes": 10, "n_edges": 14, "heterophily_ratio": 0.3, "pos_rate": 0.2},
                        scenario_info={
                            "n_added_edges_requested_total": 2,
                            "n_added_edges_actual_total": 2,
                            "n_added_edge_pairs_requested": 2,
                            "n_added_edge_pairs_actual": 2,
                        },
                    ),
                    _audit_row(
                        split_id="s0",
                        scenario_id="camouflage_feature_oracle",
                        severity=0.3,
                        graph_seed=0,
                        oracle_labels=True,
                        base_graph_path=str(clean_path),
                        graph_path=str(camo_path),
                        scenario_family="camouflage_feature",
                        scenario_method="replace_fraud_features_with_normal_features",
                        severity_param="p_cam_feat",
                        base_stats={"n_nodes": 10, "n_edges": 12, "heterophily_ratio": 0.3, "pos_rate": 0.2},
                        variant_stats={"n_nodes": 10, "n_edges": 12, "heterophily_ratio": 0.3, "pos_rate": 0.2},
                        scenario_info={
                            "n_camouflaged_nodes_requested": 2,
                            "n_camouflaged_nodes_actual": 2,
                            "mean_cosine_to_sampled_normal_before": 0.1,
                            "mean_cosine_to_sampled_normal_after": 0.8,
                        },
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
                    {"scenario_id": "noise_edges", "severity_values": [0.0, 0.2], "oracle_mode": "non_oracle"},
                    {
                        "scenario_id": "camouflage_feature_oracle",
                        "severity_values": [0.0, 0.3],
                        "oracle_mode": "oracle",
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

    def test_run_plots_stage_joins_audit_rows_per_protocol_and_reads_by_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            clean_path = out_dir / "graphs" / "clean.bin"
            stressed_path = out_dir / "graphs" / "relcam.bin"
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
                        graph_seed=7,
                    ),
                    _variant_row(
                        split_id="s0",
                        graph_path=str(stressed_path),
                        base_graph_path=str(clean_path),
                        scenario_id="camouflage_relation_oracle",
                        severity=0.3,
                        oracle_labels=True,
                        graph_seed=7,
                    ),
                ],
            )
            _write_variant_audit_csv(
                out_dir / "variant_audit.csv",
                [
                    _audit_row(
                        split_id="s0",
                        scenario_id="camouflage_relation_oracle",
                        severity=0.3,
                        graph_seed=7,
                        oracle_labels=True,
                        base_graph_path=str(clean_path),
                        graph_path=str(stressed_path),
                        scenario_family="camouflage_relation",
                        scenario_method="add_relation_camouflage_edges",
                        severity_param="p_cam_rel",
                        graph_view_mode="heterograph_aware_generation",
                        base_stats={"n_nodes": 10, "n_edges": 12, "heterophily_ratio": 0.3, "pos_rate": 0.2},
                        variant_stats={"n_nodes": 10, "n_edges": 13, "heterophily_ratio": 0.3, "pos_rate": 0.2},
                        scenario_info={
                            "n_camouflaged_edges_requested": 1,
                            "n_camouflaged_edges_added": 1,
                            "fraud_to_normal_neighbor_ratio_before": 0.2,
                            "fraud_to_normal_neighbor_ratio_after": 0.8,
                        },
                    ),
                    _audit_row(
                        split_id="s0",
                        scenario_id="clean",
                        severity=0.0,
                        graph_seed=7,
                        oracle_labels=False,
                        base_graph_path=str(clean_path),
                        graph_path=str(clean_path),
                        scenario_family="clean",
                        scenario_method="clean",
                        severity_param="severity",
                    ),
                ],
            )

            results_csv = out_dir / "results.csv"
            ensure_results_csv(results_csv)
            for protocol, score in (("train_on_variant", 0.72), ("train_clean_eval_all", 0.68)):
                append_result_row(
                    results_csv,
                    {
                        "experiment_name": "exp",
                        "dataset_id": "yelpchi",
                        "split_id": "s0",
                        "graph_seed": 7,
                        "training_seed": 0,
                        "scenario_id": "camouflage_relation_oracle",
                        "severity": 0.3,
                        "model_id": "mlp",
                        "protocol": protocol,
                        "roc_auc": score,
                        "average_precision": score - 0.1,
                        "f1_macro": score - 0.2,
                        "threshold": 0.5,
                        "duration_sec": 1.0,
                        "n_nodes": 10,
                        "n_edges": 13,
                        "mean_in_degree": 1.3,
                        "median_in_degree": 1.0,
                        "mean_out_degree": 1.3,
                        "median_out_degree": 1.0,
                        "heterophily_ratio": 0.3,
                        "pos_rate": 0.2,
                        "base_graph_path": str(clean_path),
                        "graph_path": str(stressed_path),
                        "train_graph_ref": str(clean_path if protocol == "train_clean_eval_all" else stressed_path),
                        "status": "ok",
                        "error": "",
                    },
                )

            cfg = {
                "models": [{"model_id": "mlp"}],
                "seeds": {"training_seeds": [0]},
                "scenarios": [
                    {
                        "scenario_id": "camouflage_relation_oracle",
                        "severity_values": [0.0, 0.3],
                        "oracle_mode": "oracle",
                    }
                ],
            }

            fake_matplotlib, fake_pyplot_mod, _fake_pyplot = _fake_matplotlib_modules()
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

            joined_rows = _read_csv_rows(out_dir / "plots" / "performance_audit_join.csv")
            protocol_rows = [row for row in joined_rows if row["scenario_id"] == "camouflage_relation_oracle"]
            self.assertEqual({row["protocol"] for row in protocol_rows}, {"train_on_variant", "train_clean_eval_all"})
            self.assertEqual({row["graph_seed"] for row in protocol_rows}, {"7"})
            self.assertEqual({row["graph_view_mode"] for row in protocol_rows}, {"heterograph_aware_generation"})
            self.assertEqual({row["fraud_to_normal_neighbor_ratio_after"] for row in protocol_rows}, {"0.8"})

            audit_rows = _read_csv_rows(out_dir / "plots" / "audit_curves.csv")
            shift_rows = [
                row
                for row in audit_rows
                if row["scenario_id"] == "camouflage_relation_oracle"
                and row["audit_metric"] == "fraud_to_normal_neighbor_ratio_shift"
            ]
            self.assertEqual({row["protocol"] for row in shift_rows}, {"train_on_variant", "train_clean_eval_all"})
            self.assertEqual({row["oracle_labels"] for row in shift_rows}, {"True"})
            self.assertEqual(
                {(row["protocol"], row["claim_scope"]) for row in shift_rows},
                {
                    ("train_on_variant", "oracle_privileged_training_diagnostic"),
                    ("train_clean_eval_all", "oracle_shift_sensitivity_diagnostic"),
                },
            )
            self.assertEqual({row["operational_ranking_eligible"] for row in shift_rows}, {"False"})
            self.assertEqual({row["mean"] for row in shift_rows}, {"0.6000000000000001"})
            self.assertTrue((out_dir / "plots" / "train_clean_eval_all" / "yelpchi" / "s0" / "audit__camouflage_relation_oracle__fraud_to_normal_neighbor_ratio_shift.png").exists())

    def test_run_plots_stage_fails_clearly_when_variant_audit_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            clean_path = out_dir / "graphs" / "clean.bin"
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
                    )
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
                    "training_seed": 0,
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

            cfg = {
                "models": [{"model_id": "mlp"}],
                "seeds": {"training_seeds": [0]},
                "scenarios": [{"scenario_id": "noise_edges", "severity_values": [0.0, 0.2], "oracle_mode": "non_oracle"}],
            }

            fake_matplotlib, fake_pyplot_mod, _fake_pyplot = _fake_matplotlib_modules()
            with mock.patch.dict(
                sys.modules,
                {"matplotlib": fake_matplotlib, "matplotlib.pyplot": fake_pyplot_mod},
            ):
                with self.assertRaisesRegex(FileNotFoundError, r"variant_audit\.csv not found"):
                    run_plots_stage(
                        cfg,
                        out_dir=out_dir,
                        include_noop=True,
                        only_clean=False,
                        max_variants=None,
                        max_training_seeds=None,
                        ci=False,
                    )

    def test_run_plots_stage_fails_clearly_when_variant_audit_row_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            clean_path = out_dir / "graphs" / "clean.bin"
            stressed_path = out_dir / "graphs" / "noise.bin"
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
                        graph_path=str(stressed_path),
                        base_graph_path=str(clean_path),
                        scenario_id="noise_edges",
                        severity=0.2,
                        oracle_labels=False,
                    ),
                ],
            )
            _write_variant_audit_csv(
                out_dir / "variant_audit.csv",
                [
                    _audit_row(
                        split_id="s0",
                        scenario_id="clean",
                        severity=0.0,
                        graph_seed=0,
                        oracle_labels=False,
                        base_graph_path=str(clean_path),
                        graph_path=str(clean_path),
                        scenario_family="clean",
                        scenario_method="clean",
                        severity_param="severity",
                    )
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
                    "training_seed": 0,
                    "scenario_id": "noise_edges",
                    "severity": 0.2,
                    "model_id": "mlp",
                    "protocol": "train_on_variant",
                    "roc_auc": 0.75,
                    "average_precision": 0.65,
                    "f1_macro": 0.55,
                    "threshold": 0.5,
                    "duration_sec": 1.0,
                    "n_nodes": 10,
                    "n_edges": 14,
                    "mean_in_degree": 1.4,
                    "median_in_degree": 1.0,
                    "mean_out_degree": 1.4,
                    "median_out_degree": 1.0,
                    "heterophily_ratio": 0.3,
                    "pos_rate": 0.2,
                    "base_graph_path": str(clean_path),
                    "graph_path": str(stressed_path),
                    "train_graph_ref": str(stressed_path),
                    "status": "ok",
                    "error": "",
                },
            )

            cfg = {
                "models": [{"model_id": "mlp"}],
                "seeds": {"training_seeds": [0]},
                "scenarios": [{"scenario_id": "noise_edges", "severity_values": [0.0, 0.2], "oracle_mode": "non_oracle"}],
            }

            fake_matplotlib, fake_pyplot_mod, _fake_pyplot = _fake_matplotlib_modules()
            with mock.patch.dict(
                sys.modules,
                {"matplotlib": fake_matplotlib, "matplotlib.pyplot": fake_pyplot_mod},
            ):
                with self.assertRaisesRegex(RuntimeError, r"variant_audit\.csv is missing rows for graph variants"):
                    run_plots_stage(
                        cfg,
                        out_dir=out_dir,
                        include_noop=True,
                        only_clean=False,
                        max_variants=None,
                        max_training_seeds=None,
                        ci=False,
                    )


if __name__ == "__main__":
    unittest.main()
