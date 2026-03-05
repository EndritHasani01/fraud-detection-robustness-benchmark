from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .baselines import BaselineHParams, build_baseline
from .metrics import (
    average_precision_binary,
    best_f1_macro_threshold,
    f1_macro_at_threshold,
    roc_auc_binary,
)
from .results import (
    PROTOCOL_TRAIN_ON_VARIANT,
    append_result_row,
    ensure_results_csv,
    load_completed_keys,
    make_run_key,
    truncate_error_message,
)
from .summarize import summarize_results_by_training_seed


@dataclass(frozen=True)
class VariantRow:
    experiment_name: str
    dataset_id: str
    split_id: str
    graph_seed: int
    scenario_id: str
    severity: float
    oracle_labels: bool
    scenario_applied: bool
    base_graph_path: str
    graph_path: str
    # graph stats copied from graph_variants.csv
    n_nodes: int
    n_edges: int
    mean_in_degree: float
    median_in_degree: float
    mean_out_degree: float
    median_out_degree: float
    heterophily_ratio: float | None
    pos_rate: float | None


def _parse_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return bool(x)
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return int(default)


def _safe_float(x: Any, default: float | None = None) -> float | None:
    try:
        return float(x)
    except Exception:
        return default


def _read_variants_csv(path: Path) -> list[VariantRow]:
    if not path.exists():
        raise FileNotFoundError(f"graph_variants.csv not found: {path}")

    rows: list[VariantRow] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                VariantRow(
                    experiment_name=str(r.get("experiment_name", "")),
                    dataset_id=str(r.get("dataset_id", "")),
                    split_id=str(r.get("split_id", "")),
                    graph_seed=_safe_int(r.get("graph_seed", 0)),
                    scenario_id=str(r.get("scenario_id", "")),
                    severity=float(r.get("severity", 0.0) or 0.0),
                    oracle_labels=_parse_bool(r.get("oracle_labels", False)),
                    scenario_applied=_parse_bool(r.get("scenario_applied", True)),
                    base_graph_path=str(r.get("base_graph_path", "")),
                    graph_path=str(r.get("graph_path", "")),
                    n_nodes=_safe_int(r.get("n_nodes", 0)),
                    n_edges=_safe_int(r.get("n_edges", 0)),
                    mean_in_degree=float(r.get("mean_in_degree", 0.0) or 0.0),
                    median_in_degree=float(r.get("median_in_degree", 0.0) or 0.0),
                    mean_out_degree=float(r.get("mean_out_degree", 0.0) or 0.0),
                    median_out_degree=float(r.get("median_out_degree", 0.0) or 0.0),
                    heterophily_ratio=_safe_float(r.get("heterophily_ratio", ""), None),
                    pos_rate=_safe_float(r.get("pos_rate", ""), None),
                )
            )
    return rows


def _set_seeds(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_graph_bin(path: Path):
    try:
        from dgl.data.utils import load_graphs
    except Exception as e:  # pragma: no cover
        raise RuntimeError("DGL is required to load cached graphs for baseline evaluation.") from e

    graphs, _ = load_graphs(str(path))
    if not graphs:
        raise RuntimeError(f"No graphs found in file: {path}")
    return graphs[0]


def _class_weights_from_train_labels(y_train):
    import torch

    y_train = y_train.to(torch.int64)
    n_pos = int((y_train == 1).sum().item())
    n_neg = int((y_train == 0).sum().item())
    if n_pos <= 0 or n_neg <= 0:
        return None
    # Weight the positive class up by imbalance ratio.
    w0 = 1.0
    w1 = float(n_neg) / float(n_pos)
    return torch.tensor([w0, w1], dtype=torch.float32)


def _forward_logits(model_id: str, model, g, x):
    if model_id == "mlp":
        return model(x)
    if model_id == "sage":
        return model(g, x)
    raise ValueError(f"Unsupported baseline model_id: {model_id}")


def train_eval_baseline(
    model_id: str,
    g,
    *,
    training_seed: int,
    device: str,
    hparams: BaselineHParams,
    feature_key: str = "feature",
    label_key: str = "label",
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    t0 = time.perf_counter()
    _set_seeds(int(training_seed))

    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # If the caller asked for CUDA but the DGL wheel/graph is CPU-only, fall back to CPU.
    if model_id == "sage" and device == "cuda":
        try:
            g = g.to("cuda")
        except Exception:
            device = "cpu"

    x = g.ndata.get(feature_key)
    y = g.ndata.get(label_key)
    if x is None or y is None:
        raise RuntimeError(f"Graph missing required ndata['{feature_key}'] or ndata['{label_key}']")
    x = x.to(torch.float32)
    y = y.squeeze().to(torch.int64)

    train_mask = g.ndata.get("train_mask")
    val_mask = g.ndata.get("val_mask")
    test_mask = g.ndata.get("test_mask")
    if train_mask is None or val_mask is None or test_mask is None:
        raise RuntimeError("Graph is missing train/val/test masks in ndata.")

    train_idx = torch.nonzero(train_mask, as_tuple=True)[0]
    val_idx = torch.nonzero(val_mask, as_tuple=True)[0]
    test_idx = torch.nonzero(test_mask, as_tuple=True)[0]
    if train_idx.numel() == 0 or val_idx.numel() == 0 or test_idx.numel() == 0:
        raise RuntimeError("One of the splits is empty; cannot train/evaluate.")

    in_dim = int(x.shape[1])
    model = build_baseline(model_id, in_dim=in_dim, hparams=hparams)
    model = model.to(device)

    # Move data to device (graph structure stays on CPU for CPU DGL wheels).
    x_dev = x.to(device)
    y_dev = y.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)

    class_w = _class_weights_from_train_labels(y_dev[train_idx])
    if class_w is not None:
        class_w = class_w.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_w)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(hparams.lr),
        weight_decay=float(hparams.weight_decay),
    )

    best_monitor = -float("inf")
    best_state = None
    bad_epochs = 0

    for _epoch in range(int(hparams.max_epochs)):
        model.train()
        logits_train = _forward_logits(model_id, model, g, x_dev)
        loss = loss_fn(logits_train[train_idx], y_dev[train_idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits_eval = _forward_logits(model_id, model, g, x_dev)
            probs = F.softmax(logits_eval, dim=1)[:, 1]
            val_scores = probs[val_idx].detach().cpu().numpy()
            val_labels = y_dev[val_idx].detach().cpu().numpy()

            val_auc = roc_auc_binary(val_labels, val_scores)
            if math.isfinite(val_auc):
                monitor = float(val_auc)
            else:
                # Fallback: minimize validation loss if AUC is undefined.
                val_loss = loss_fn(logits_eval[val_idx], y_dev[val_idx]).item()
                monitor = -float(val_loss)

        if monitor > best_monitor:
            best_monitor = float(monitor)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(hparams.patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = _forward_logits(model_id, model, g, x_dev)
        probs = F.softmax(logits, dim=1)[:, 1]

        val_scores = probs[val_idx].detach().cpu().numpy()
        val_labels = y_dev[val_idx].detach().cpu().numpy()
        th = best_f1_macro_threshold(val_labels, val_scores)

        test_scores = probs[test_idx].detach().cpu().numpy()
        test_labels = y_dev[test_idx].detach().cpu().numpy()

        roc_auc = roc_auc_binary(test_labels, test_scores)
        ap = average_precision_binary(test_labels, test_scores)
        f1m = f1_macro_at_threshold(test_labels, test_scores, th.threshold)

    dt = time.perf_counter() - t0
    return {
        "roc_auc": float(roc_auc) if roc_auc is not None else float("nan"),
        "average_precision": float(ap) if ap is not None else float("nan"),
        "f1_macro": float(f1m),
        "threshold": float(th.threshold),
        "duration_sec": float(dt),
    }


def run_baselines_stage(
    cfg: dict[str, Any],
    *,
    out_dir: Path,
    force: bool,
    skip_existing: bool,
    retry_errors: bool,
    device: str,
    include_noop: bool,
    only_clean: bool,
    max_variants: int | None,
    max_training_seeds: int | None,
    max_epochs: int | None,
    patience: int | None,
) -> None:
    """Train/evaluate baseline models and append rows into results.csv."""
    out_dir = out_dir.resolve()
    results_csv = out_dir / "results.csv"
    variants_csv = out_dir / "graph_variants.csv"

    ensure_results_csv(results_csv, overwrite=bool(force))
    completed_keys = load_completed_keys(results_csv, retry_errors=bool(retry_errors)) if skip_existing else set()

    variants = _read_variants_csv(variants_csv)
    if not variants:
        raise RuntimeError(f"No rows found in {variants_csv}")

    training_seeds = [int(s) for s in cfg["seeds"]["training_seeds"]]
    if max_training_seeds is not None:
        training_seeds = training_seeds[: int(max_training_seeds)]

    baseline_model_ids = [m["model_id"] for m in cfg["models"] if str(m.get("model_id")) in {"mlp", "sage"}]
    if not baseline_model_ids:
        baseline_model_ids = ["mlp", "sage"]

    filtered: list[VariantRow] = []
    for v in variants:
        if only_clean and v.scenario_id != "clean":
            continue
        if (not include_noop) and (not v.scenario_applied) and v.scenario_id != "clean":
            continue
        filtered.append(v)

    if max_variants is not None:
        filtered = filtered[: int(max_variants)]

    protocol = PROTOCOL_TRAIN_ON_VARIANT

    for v in filtered:
        pending_runs: list[tuple[str, int, tuple[str, str, str, float, int, int, str, str], dict[str, Any]]] = []

        for training_seed in training_seeds:
            for model_id in baseline_model_ids:
                row_common = {
                    "experiment_name": v.experiment_name,
                    "dataset_id": v.dataset_id,
                    "split_id": v.split_id,
                    "graph_seed": int(v.graph_seed),
                    "training_seed": int(training_seed),
                    "scenario_id": v.scenario_id,
                    "severity": float(v.severity),
                    "model_id": str(model_id),
                    "protocol": protocol,
                    "n_nodes": int(v.n_nodes),
                    "n_edges": int(v.n_edges),
                    "mean_in_degree": float(v.mean_in_degree),
                    "median_in_degree": float(v.median_in_degree),
                    "mean_out_degree": float(v.mean_out_degree),
                    "median_out_degree": float(v.median_out_degree),
                    "heterophily_ratio": v.heterophily_ratio,
                    "pos_rate": v.pos_rate,
                    "base_graph_path": v.base_graph_path,
                    "graph_path": v.graph_path,
                }
                run_key = make_run_key(
                    dataset_id=v.dataset_id,
                    split_id=v.split_id,
                    scenario_id=v.scenario_id,
                    severity=v.severity,
                    graph_seed=v.graph_seed,
                    training_seed=training_seed,
                    model_id=model_id,
                    protocol=protocol,
                )
                if run_key in completed_keys:
                    print(
                        f"[skip] {model_id} / {v.scenario_id} / sev={float(v.severity):g} / "
                        f"gs={int(v.graph_seed)} / ts={int(training_seed)} already done"
                    )
                    continue
                pending_runs.append((str(model_id), int(training_seed), run_key, row_common))

        if not pending_runs:
            continue

        graph_t0 = time.perf_counter()
        try:
            g = _load_graph_bin(Path(v.graph_path))
        except Exception as e:
            graph_error = truncate_error_message(e)
            dt = time.perf_counter() - graph_t0
            for _model_id, _training_seed, run_key, row_common in pending_runs:
                append_result_row(
                    results_csv,
                    {
                        **row_common,
                        "duration_sec": float(dt),
                        "status": "error",
                        "error": graph_error,
                    },
                )
                completed_keys.add(run_key)
            continue

        for model_id, training_seed, run_key, row_common in pending_runs:
            run_t0 = time.perf_counter()
            try:
                if model_id == "mlp":
                    hp = BaselineHParams(lr=1e-3, max_epochs=100, patience=10, hidden_dim=128, dropout=0.5)
                else:
                    hp = BaselineHParams(lr=1e-2, max_epochs=100, patience=10, hidden_dim=64, dropout=0.5)

                if max_epochs is not None:
                    hp = BaselineHParams(
                        hidden_dim=hp.hidden_dim,
                        dropout=hp.dropout,
                        lr=hp.lr,
                        weight_decay=hp.weight_decay,
                        max_epochs=int(max_epochs),
                        patience=hp.patience,
                    )
                if patience is not None:
                    hp = BaselineHParams(
                        hidden_dim=hp.hidden_dim,
                        dropout=hp.dropout,
                        lr=hp.lr,
                        weight_decay=hp.weight_decay,
                        max_epochs=hp.max_epochs,
                        patience=int(patience),
                    )

                out = train_eval_baseline(
                    str(model_id),
                    g,
                    training_seed=int(training_seed),
                    device=str(device),
                    hparams=hp,
                )
                append_result_row(
                    results_csv,
                    {
                        **row_common,
                        "roc_auc": out["roc_auc"],
                        "average_precision": out["average_precision"],
                        "f1_macro": out["f1_macro"],
                        "threshold": out["threshold"],
                        "duration_sec": out["duration_sec"],
                        "status": "ok",
                        "error": "",
                    },
                )
            except Exception as e:
                append_result_row(
                    results_csv,
                    {
                        **row_common,
                        "duration_sec": time.perf_counter() - run_t0,
                        "status": "error",
                        "error": truncate_error_message(e),
                    },
                )
            finally:
                completed_keys.add(run_key)

    summarize_results_by_training_seed(
        results_csv,
        out_csv_path=out_dir / "results_summary_baselines.csv",
        model_ids=set(baseline_model_ids),
    )
