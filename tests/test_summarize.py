from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from benchmark.results import (
    PROTOCOL_TRAIN_CLEAN_EVAL_ALL,
    PROTOCOL_TRAIN_ON_VARIANT,
    append_result_row,
    ensure_results_csv,
)
from benchmark.summarize import summarize_results_by_training_seed


def _result_row(protocol: str, score: float) -> dict[str, object]:
    return {
        "experiment_name": "exp",
        "dataset_id": "yelpchi",
        "split_id": "s0",
        "scenario_id": "clean",
        "severity": 0.0,
        "graph_seed": 717,
        "training_seed": 0,
        "model_id": "mlp",
        "protocol": protocol,
        "roc_auc": score,
        "average_precision": score,
        "f1_macro": score,
        "status": "ok",
    }


class SummaryProtocolFilterTests(unittest.TestCase):
    def test_protocol_filter_prevents_cross_protocol_summary_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            results = root / "results.csv"
            summary = root / "summary.csv"
            ensure_results_csv(results)
            append_result_row(results, _result_row(PROTOCOL_TRAIN_ON_VARIANT, 0.9))
            append_result_row(results, _result_row(PROTOCOL_TRAIN_CLEAN_EVAL_ALL, 0.6))

            summarize_results_by_training_seed(
                results,
                out_csv_path=summary,
                model_ids={"mlp"},
                protocols={PROTOCOL_TRAIN_CLEAN_EVAL_ALL},
            )

            with summary.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["protocol"], PROTOCOL_TRAIN_CLEAN_EVAL_ALL)
            self.assertAlmostEqual(float(rows[0]["average_precision_mean"]), 0.6)


if __name__ == "__main__":
    unittest.main()
