from __future__ import annotations

from dataclasses import dataclass


def _as_float_array(x):
    import numpy as np

    a = np.asarray(x, dtype=np.float64).reshape(-1)
    return a


def _binary_score_inputs(y_true, y_score):
    """Return validated one-dimensional binary labels and finite scores."""
    import numpy as np

    raw_y = np.asarray(y_true).reshape(-1)
    s = _as_float_array(y_score)
    if raw_y.shape[0] != s.shape[0]:
        raise ValueError("y_true and y_score must have the same length")
    try:
        finite_labels = np.isfinite(raw_y.astype(np.float64))
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true must contain only binary labels {0,1}") from exc
    if not np.all(finite_labels) or not np.all((raw_y == 0) | (raw_y == 1)):
        raise ValueError("y_true must contain only binary labels {0,1}")
    if not np.all(np.isfinite(s)):
        raise ValueError("y_score must contain only finite values")
    return raw_y.astype(np.int64), s


def _binary_prediction_inputs(y_true, y_pred):
    """Return validated binary labels and hard binary predictions."""
    import numpy as np

    raw_y = np.asarray(y_true).reshape(-1)
    raw_p = np.asarray(y_pred).reshape(-1)
    if raw_y.shape[0] != raw_p.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    for name, values in (("y_true", raw_y), ("y_pred", raw_p)):
        try:
            finite = np.isfinite(values.astype(np.float64))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain only binary labels {{0,1}}") from exc
        if not np.all(finite) or not np.all((values == 0) | (values == 1)):
            raise ValueError(f"{name} must contain only binary labels {{0,1}}")
    return raw_y.astype(np.int64), raw_p.astype(np.int64)


def roc_auc_binary(y_true, y_score) -> float:
    """ROC-AUC for binary labels {0,1} without sklearn.

    Returns NaN if AUC is undefined (all labels the same).
    """
    import math
    import numpy as np

    y, s = _binary_score_inputs(y_true, y_score)

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Rank-based AUC (Mann-Whitney U) with tie-aware average ranks.
    order = np.argsort(s, kind="mergesort")  # ascending
    s_sorted = s[order]
    y_sorted = y[order]

    ranks = np.empty_like(s_sorted, dtype=np.float64)
    i = 0
    n = s_sorted.shape[0]
    # Assign average rank for ties; ranks are 1..n.
    while i < n:
        j = i + 1
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)  # i is 0-based, j is exclusive
        ranks[i:j] = avg_rank
        i = j

    sum_ranks_pos = float(np.sum(ranks[y_sorted == 1]))
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)

    # Numerical safety.
    if not math.isfinite(auc):
        return float("nan")
    return float(max(0.0, min(1.0, auc)))


def average_precision_binary(y_true, y_score) -> float:
    """Average precision (AP) for binary labels {0,1} without sklearn.

    Score ties are evaluated as one threshold group. This makes AP independent
    of the input order inside a tied group and matches the step-wise
    precision-recall definition used by standard ML libraries.

    Returns NaN if AP is undefined (no positive labels).
    """
    import math
    import numpy as np

    y, s = _binary_score_inputs(y_true, y_score)

    n_pos = int(np.sum(y == 1))
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-s, kind="mergesort")  # descending
    y_sorted = y[order]
    s_sorted = s[order]

    tp_total = 0
    seen_total = 0
    previous_recall = 0.0
    ap = 0.0
    i = 0
    n = y_sorted.shape[0]
    while i < n:
        j = i + 1
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1
        tp_total += int(np.sum(y_sorted[i:j] == 1))
        seen_total += j - i
        recall = tp_total / n_pos
        precision = tp_total / seen_total
        ap += (recall - previous_recall) * precision
        previous_recall = recall
        i = j

    ap = float(ap)
    if not math.isfinite(ap):
        return float("nan")
    return float(max(0.0, min(1.0, ap)))


def f1_macro_from_predictions(y_true, y_pred) -> float:
    """Macro-F1 for binary labels, computed from predicted hard labels {0,1}."""
    import math
    import numpy as np

    y, p = _binary_prediction_inputs(y_true, y_pred)

    tp = int(np.sum((y == 1) & (p == 1)))
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))

    def _f1(num_tp: int, num_fp: int, num_fn: int) -> float:
        denom = (2 * num_tp + num_fp + num_fn)
        if denom <= 0:
            return 0.0
        return float((2 * num_tp) / denom)

    f1_pos = _f1(tp, fp, fn)
    # Treat class 0 as the positive label for the "negative class" F1:
    # TP0=TN, FP0=FN, FN0=FP.
    f1_neg = _f1(tn, fn, fp)
    f1m = 0.5 * (f1_pos + f1_neg)
    if not math.isfinite(f1m):
        return 0.0
    return float(max(0.0, min(1.0, f1m)))


@dataclass(frozen=True)
class ThresholdSearchResult:
    threshold: float
    f1_macro: float


def best_f1_macro_threshold(y_true, y_score) -> ThresholdSearchResult:
    """Pick a threshold using ONLY the provided labels/scores.

    Search strategy:
    - Evaluate thresholds at each unique score value (descending), with rule: predict 1 iff score >= threshold.
    - Deterministic tie-breaking: keeps the first (highest threshold) that achieves the best score.
    """
    import numpy as np

    y, s = _binary_score_inputs(y_true, y_score)

    # Sort by score descending, stable for determinism.
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]

    n = y_sorted.shape[0]
    n_pos = int(np.sum(y_sorted == 1))
    n_neg = n - n_pos
    if n == 0 or n_pos == 0 or n_neg == 0:
        # Degenerate; default threshold 0.5.
        thr = 0.5
        pred = (s >= thr).astype(np.int64)
        return ThresholdSearchResult(threshold=float(thr), f1_macro=f1_macro_from_predictions(y, pred))

    # Cumulative positives in the prefix (predicted positive set for a given threshold).
    tp_prefix = np.cumsum((y_sorted == 1).astype(np.int64))

    best_thr = float("inf")
    best_f1 = -1.0

    i = 0
    while i < n:
        j = i + 1
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1

        # Threshold = this score value includes all items in [0, j).
        k = j  # number predicted positive
        tp = int(tp_prefix[j - 1])
        fp = k - tp
        fn = n_pos - tp
        tn = n_neg - fp

        # Compute macro-F1 from confusion matrix.
        denom_pos = (2 * tp + fp + fn)
        f1_pos = (2 * tp / denom_pos) if denom_pos > 0 else 0.0
        denom_neg = (2 * tn + fn + fp)
        f1_neg = (2 * tn / denom_neg) if denom_neg > 0 else 0.0
        f1m = 0.5 * (f1_pos + f1_neg)

        thr = float(s_sorted[i])
        if f1m > best_f1:
            best_f1 = float(f1m)
            best_thr = thr

        i = j

    if best_thr == float("inf"):
        best_thr = 0.5
    return ThresholdSearchResult(threshold=float(best_thr), f1_macro=float(best_f1))


def f1_macro_at_threshold(y_true, y_score, threshold: float) -> float:
    import math
    import numpy as np

    y, s = _binary_score_inputs(y_true, y_score)
    use_threshold = float(threshold)
    if not math.isfinite(use_threshold):
        raise ValueError("threshold must be finite")
    pred = (s >= use_threshold).astype(np.int64)
    return f1_macro_from_predictions(y, pred)

