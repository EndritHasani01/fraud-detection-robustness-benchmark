from __future__ import annotations

import math
import unittest

from benchmark.metrics import (
    average_precision_binary,
    best_f1_macro_threshold,
    f1_macro_at_threshold,
    roc_auc_binary,
)


class MetricsTests(unittest.TestCase):
    def test_average_precision_groups_tied_scores(self) -> None:
        self.assertAlmostEqual(average_precision_binary([1, 0], [0.5, 0.5]), 0.5)
        self.assertAlmostEqual(average_precision_binary([0, 1], [0.5, 0.5]), 0.5)

    def test_average_precision_is_invariant_to_order_within_ties(self) -> None:
        first = average_precision_binary([1, 0, 1, 0], [0.9, 0.9, 0.2, 0.2])
        second = average_precision_binary([0, 1, 0, 1], [0.9, 0.9, 0.2, 0.2])
        self.assertAlmostEqual(first, second)
        self.assertAlmostEqual(first, 0.5)

    def test_perfect_ranking_metrics_are_one(self) -> None:
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        self.assertEqual(roc_auc_binary(labels, scores), 1.0)
        self.assertEqual(average_precision_binary(labels, scores), 1.0)

    def test_score_metrics_reject_nonfinite_values(self) -> None:
        for function in (roc_auc_binary, average_precision_binary, best_f1_macro_threshold):
            with self.subTest(function=function.__name__):
                with self.assertRaisesRegex(ValueError, "finite"):
                    function([0, 1], [0.1, math.nan])

    def test_threshold_metric_rejects_nonfinite_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "threshold must be finite"):
            f1_macro_at_threshold([0, 1], [0.1, 0.9], math.inf)

    def test_threshold_search_includes_all_negative_candidate(self) -> None:
        labels = [0, 0, 0, 1]
        scores = [0.9, 0.8, 0.7, 0.1]

        result = best_f1_macro_threshold(labels, scores)

        self.assertGreater(result.threshold, max(scores))
        self.assertAlmostEqual(result.f1_macro, 3.0 / 7.0)
        self.assertAlmostEqual(
            result.f1_macro,
            f1_macro_at_threshold(labels, scores, result.threshold),
        )

    def test_threshold_search_tie_breaks_toward_all_negative_candidate(self) -> None:
        labels = [0, 1]
        scores = [0.9, 0.1]

        result = best_f1_macro_threshold(labels, scores)

        self.assertGreater(result.threshold, max(scores))
        self.assertAlmostEqual(result.f1_macro, 1.0 / 3.0)

    def test_metrics_reject_nonbinary_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "binary labels"):
            average_precision_binary([0, 2], [0.1, 0.9])


if __name__ == "__main__":
    unittest.main()
