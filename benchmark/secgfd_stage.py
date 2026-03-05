from __future__ import annotations

import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import get_training_seeds
from .metrics import average_precision_binary, best_f1_macro_threshold, f1_macro_at_threshold, roc_auc_binary
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

    try:
        import dgl

        dgl.seed(int(seed))
        dgl.random.seed(int(seed))
    except Exception:
        pass


def _load_graph_bin(path: Path):
    try:
        from dgl.data.utils import load_graphs
    except Exception as e:  # pragma: no cover
        raise RuntimeError("DGL is required to load cached graphs for SEC-GFD evaluation.") from e

    graphs, _ = load_graphs(str(path))
    if not graphs:
        raise RuntimeError(f"No graphs found in file: {path}")
    return graphs[0]


def _import_secgfd_class(repo_root: Path):
    import importlib.util

    repo_root = repo_root.resolve()
    secgfd_py = repo_root / "model" / "SECGFD.py"
    if not secgfd_py.exists():
        raise FileNotFoundError(f"SEC-GFD model file not found: {secgfd_py}")

    # Load under a unique module name to avoid collisions with other repos that
    # use a top-level package name like 'model'.
    mod_name = "secgfd_repo_model_SECGFD"
    spec = importlib.util.spec_from_file_location(mod_name, secgfd_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for: {secgfd_py}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)

    if not hasattr(mod, "SECGFD"):
        raise RuntimeError("SEC-GFD module did not define SECGFD")
    return getattr(mod, "SECGFD")


def _class_weight_pos(labels_train):
    import torch

    labels_train = labels_train.to(torch.int64)
    n_pos = int((labels_train == 1).sum().item())
    n_neg = int((labels_train == 0).sum().item())
    if n_pos <= 0:
        return 1.0
    return float(n_neg) / float(n_pos)


def _nce_loss_fixed(emb, features, labels, train_idx, *, eps: float = 1e-8):
    """SEC-GFD's contrastive-like loss, fixed to index the *training nodes*.

    The original repo indexes similarities with positions-in-train rather than
    global node IDs. Here we use the global node indices in train_idx.
    """
    import torch
    import torch.nn.functional as F

    if train_idx.numel() == 0:
        return torch.tensor(0.0, device=features.device)

    y_tr = labels[train_idx].to(torch.int64)
    normal_mask = (y_tr == 0)
    anomaly_mask = (y_tr == 1)
    if int(normal_mask.sum().item()) == 0 or int(anomaly_mask.sum().item()) == 0:
        return torch.tensor(0.0, device=features.device)

    sim = F.cosine_similarity(features, emb, dim=1)
    nor = sim[train_idx[normal_mask]].mean()
    abn = sim[train_idx[anomaly_mask]].mean()
    return -torch.log((nor + float(eps)) / (abn + float(eps)))


def train_eval_secgfd(
    g,
    *,
    repo_root: Path,
    training_seed: int,
    device: str,
    max_epochs: int | None,
    patience: int | None,
    hid_dim: int = 64,
    order_d: int = 2,
    high_order: int = 2,
    lemda: float = 0.2,
    lr: float = 0.01,
    weight_decay: float = 0.0,
) -> dict[str, Any]:
    import copy

    import torch
    import torch.nn.functional as F

    t0 = time.perf_counter()
    _set_seeds(int(training_seed))

    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if device == "cuda":
        # SEC-GFD is full-graph; if the DGL wheel is CPU-only, this will fail.
        try:
            g = g.to("cuda")
        except Exception:
            device = "cpu"
            g = g.to("cpu")
    else:
        g = g.to("cpu")

    features = g.ndata.get("feature")
    labels = g.ndata.get("label")
    if features is None or labels is None:
        raise RuntimeError("Graph missing ndata['feature'] or ndata['label']")
    features = features.to(torch.float32).to(device)
    labels = labels.squeeze().to(torch.int64).to(device)

    train_mask = g.ndata.get("train_mask")
    val_mask = g.ndata.get("val_mask")
    test_mask = g.ndata.get("test_mask")
    if train_mask is None or val_mask is None or test_mask is None:
        raise RuntimeError("Graph is missing train/val/test masks.")
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    train_idx = torch.nonzero(train_mask, as_tuple=True)[0]
    val_idx = torch.nonzero(val_mask, as_tuple=True)[0]
    test_idx = torch.nonzero(test_mask, as_tuple=True)[0]
    if train_idx.numel() == 0 or val_idx.numel() == 0 or test_idx.numel() == 0:
        raise RuntimeError("One of the splits is empty; cannot train/evaluate.")

    in_dim = int(features.shape[1])
    out_dim = 2

    SECGFD = _import_secgfd_class(repo_root)
    model = SECGFD(in_dim, int(hid_dim), out_dim, g, d=int(order_d), high_order=int(high_order)).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    # Class imbalance weight on training split.
    pos_w = _class_weight_pos(labels[train_idx])
    ce_weight = torch.tensor([1.0, float(pos_w)], dtype=torch.float32, device=device)

    epochs = 100 if max_epochs is None else int(max_epochs)
    es_patience = 10 if patience is None else int(patience)

    best_monitor = -float("inf")
    best_state = None
    bad_epochs = 0

    for _epoch in range(int(epochs)):
        model.train()
        logits, emb = model(features)

        loss_ce = F.cross_entropy(logits[train_idx], labels[train_idx], weight=ce_weight)
        loss_nce = _nce_loss_fixed(emb, features, labels, train_idx)
        loss = loss_ce + float(lemda) * loss_nce

        opt.zero_grad(set_to_none=True)
        loss.backward()

        # Validation monitor computed from the current forward pass (no extra full-graph forward).
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)[:, 1]
            val_scores = probs[val_idx].detach().cpu().numpy()
            val_labels = labels[val_idx].detach().cpu().numpy()
            val_auc = roc_auc_binary(val_labels, val_scores)
            monitor = float(val_auc) if math.isfinite(val_auc) else -float(loss_ce.item())

        if monitor > best_monitor:
            best_monitor = monitor
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1
            if es_patience > 0 and bad_epochs >= es_patience:
                opt.step()
                break

        opt.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits, _emb = model(features)
        probs = torch.softmax(logits, dim=1)[:, 1]

        val_scores = probs[val_idx].detach().cpu().numpy()
        val_labels = labels[val_idx].detach().cpu().numpy()
        th = best_f1_macro_threshold(val_labels, val_scores)

        test_scores = probs[test_idx].detach().cpu().numpy()
        test_labels = labels[test_idx].detach().cpu().numpy()

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


def run_secgfd_stage(
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
    out_dir = out_dir.resolve()
    results_csv = out_dir / "results.csv"
    variants_csv = out_dir / "graph_variants.csv"

    ensure_results_csv(results_csv, overwrite=bool(force))
    completed_keys = load_completed_keys(results_csv, retry_errors=bool(retry_errors)) if skip_existing else set()

    variants = _read_variants_csv(variants_csv)
    if not variants:
        raise RuntimeError(f"No rows found in {variants_csv}")

    sec_cfg = None
    for m in cfg.get("models", []):
        if str(m.get("model_id", "")) == "secgfd":
            sec_cfg = m
            break
    if not sec_cfg:
        raise RuntimeError("Config has no model entry with model_id='secgfd'")

    repo_root = Path(str(sec_cfg.get("repo_path", "")))
    if not repo_root.exists():
        raise FileNotFoundError(f"Configured SEC-GFD repo_path does not exist: {repo_root}")

    training_seeds = get_training_seeds(cfg)
    if max_training_seeds is not None:
        training_seeds = training_seeds[: int(max_training_seeds)]

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
        pending_runs: list[tuple[int, tuple[str, str, str, float, int, int, str, str], dict[str, Any]]] = []

        for training_seed in training_seeds:
            row_common = {
                "experiment_name": v.experiment_name,
                "dataset_id": v.dataset_id,
                "split_id": v.split_id,
                "graph_seed": int(v.graph_seed),
                "training_seed": int(training_seed),
                "scenario_id": v.scenario_id,
                "severity": float(v.severity),
                "model_id": "secgfd",
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
                model_id="secgfd",
                protocol=protocol,
            )
            if run_key in completed_keys:
                print(
                    f"[skip] secgfd / {v.scenario_id} / sev={float(v.severity):g} / "
                    f"gs={int(v.graph_seed)} / ts={int(training_seed)} already done"
                )
                continue
            pending_runs.append((int(training_seed), run_key, row_common))

        if not pending_runs:
            continue

        graph_t0 = time.perf_counter()
        try:
            g = _load_graph_bin(Path(v.graph_path))
        except Exception as e:
            graph_error = truncate_error_message(e)
            dt = time.perf_counter() - graph_t0
            for _training_seed, run_key, row_common in pending_runs:
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

        for training_seed, run_key, row_common in pending_runs:
            run_t0 = time.perf_counter()
            try:
                out = train_eval_secgfd(
                    g,
                    repo_root=repo_root,
                    training_seed=int(training_seed),
                    device=str(device),
                    max_epochs=max_epochs,
                    patience=patience,
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
        out_csv_path=out_dir / "results_summary_secgfd.csv",
        model_ids={"secgfd"},
    )
