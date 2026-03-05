from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from benchmark.variants import VariantRow, filter_variants, read_variants_csv


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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class VariantHelpersTests(unittest.TestCase):
    def test_read_variants_csv_parses_typed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph_variants.csv"
            _write_variant_csv(
                path,
                [
                    {
                        "experiment_name": "exp",
                        "dataset_id": "yelpchi",
                        "split_id": "split_0",
                        "graph_seed": "7",
                        "scenario_id": "noise_edges",
                        "severity": "0.2",
                        "oracle_labels": "false",
                        "scenario_applied": "true",
                        "base_graph_path": "base.bin",
                        "graph_path": "noise.bin",
                        "n_nodes": "10",
                        "n_edges": "12",
                        "mean_in_degree": "1.2",
                        "median_in_degree": "1.0",
                        "mean_out_degree": "1.2",
                        "median_out_degree": "1.0",
                        "heterophily_ratio": "0.3",
                        "pos_rate": "0.2",
                    }
                ],
            )

            rows = read_variants_csv(path)

            self.assertEqual(len(rows), 1)
            self.assertIsInstance(rows[0], VariantRow)
            self.assertEqual(rows[0].graph_seed, 7)
            self.assertEqual(rows[0].severity, 0.2)
            self.assertFalse(rows[0].oracle_labels)
            self.assertTrue(rows[0].scenario_applied)
            self.assertEqual(rows[0].heterophily_ratio, 0.3)
            self.assertEqual(rows[0].pos_rate, 0.2)

    def test_filter_variants_matches_stage_selection_rules(self) -> None:
        variants = [
            VariantRow(
                experiment_name="exp",
                dataset_id="yelpchi",
                split_id="split_0",
                graph_seed=0,
                scenario_id="clean",
                severity=0.0,
                oracle_labels=False,
                scenario_applied=True,
                base_graph_path="clean.bin",
                graph_path="clean.bin",
                n_nodes=10,
                n_edges=12,
                mean_in_degree=1.2,
                median_in_degree=1.0,
                mean_out_degree=1.2,
                median_out_degree=1.0,
                heterophily_ratio=0.3,
                pos_rate=0.2,
            ),
            VariantRow(
                experiment_name="exp",
                dataset_id="yelpchi",
                split_id="split_0",
                graph_seed=0,
                scenario_id="noise_edges",
                severity=0.0,
                oracle_labels=False,
                scenario_applied=False,
                base_graph_path="clean.bin",
                graph_path="noop.bin",
                n_nodes=10,
                n_edges=12,
                mean_in_degree=1.2,
                median_in_degree=1.0,
                mean_out_degree=1.2,
                median_out_degree=1.0,
                heterophily_ratio=0.3,
                pos_rate=0.2,
            ),
            VariantRow(
                experiment_name="exp",
                dataset_id="yelpchi",
                split_id="split_0",
                graph_seed=0,
                scenario_id="noise_edges",
                severity=0.2,
                oracle_labels=False,
                scenario_applied=True,
                base_graph_path="clean.bin",
                graph_path="noise.bin",
                n_nodes=10,
                n_edges=12,
                mean_in_degree=1.2,
                median_in_degree=1.0,
                mean_out_degree=1.2,
                median_out_degree=1.0,
                heterophily_ratio=0.3,
                pos_rate=0.2,
            ),
        ]

        self.assertEqual(
            [v.graph_path for v in filter_variants(variants, include_noop=False, only_clean=False, max_variants=None)],
            ["clean.bin", "noise.bin"],
        )
        self.assertEqual(
            [v.graph_path for v in filter_variants(variants, include_noop=True, only_clean=True, max_variants=None)],
            ["clean.bin"],
        )
        self.assertEqual(
            [v.graph_path for v in filter_variants(variants, include_noop=True, only_clean=False, max_variants=2)],
            ["clean.bin", "noop.bin"],
        )


if __name__ == "__main__":
    unittest.main()
