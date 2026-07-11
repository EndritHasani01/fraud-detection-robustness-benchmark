from __future__ import annotations

import math
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .baselines import BaselineHParams, build_baseline
from .config import get_training_seeds
from .metrics import (
    average_precision_binary,
    best_f1_macro_threshold,
    f1_macro_at_threshold,
    roc_auc_binary,
)
from .preflight import (
    ProgressTracker,
    build_expected_run_keys,
    print_training_preflight,
    select_model_ids,
    summarize_training_preflight,
    warn_no_matching_models,
)
from .results import (
    PROTOCOL_TRAIN_ON_VARIANT,
    ensure_results_csv,
    load_completed_keys,
    make_run_key,
    write_result_row,
)
from .summarize import summarize_results_by_training_seed
from .variants import (
    class_weights_from_train_labels,
    filter_variants,
    load_graph_bin,
    require_variants_csv_rows,
    set_seeds,
)


@dataclass(frozen=True)
class BaselineModelArtifact:
    model_id: str
    hparams: BaselineHParams
    state_dict: dict[str, Any]
    threshold: float
    device: str
    feature_key: str = "feature"
    label_key: str = "label"


def _forward_logits(model_id: str, model, g, x):
    if model_id == "mlp":
        return model(x)
    if model_id == "sage":
        return model(g, x)
    raise ValueError(f"Unsupported baseline model_id: {model_id}")


def build_baseline_hparams(model_id: str, *, max_epochs: int | None, patience: int | None) -> BaselineHParams:
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
    return hp


def _resolve_baseline_device(model_id: str, g, device: str):
    import torch

    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for baseline training, but PyTorch reports that CUDA is unavailable."
        )

    if model_id == "sage" and device == "cuda":
        try:
            g = g.to("cuda")
        except Exception as exc:
            raise RuntimeError(
                "CUDA was requested for GraphSAGE, but the graph could not be moved to CUDA."
            ) from exc
    return g, str(device)


def _baseline_split_tensors(g, *, device: str, feature_key: str, label_key: str):
    import torch

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
    x_dev = x.to(device)
    y_dev = y.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    return in_dim, x_dev, y_dev, train_idx, val_idx, test_idx


def train_baseline_model(
    model_id: str,
    g,
    *,
    training_seed: int,
    device: str,
    hparams: BaselineHParams,
    feature_key: str = "feature",
    label_key: str = "label",
) -> BaselineModelArtifact:
    import torch
    import torch.nn.functional as F

    set_seeds(int(training_seed))
    g, effective_device = _resolve_baseline_device(model_id, g, device)
    in_dim, x_dev, y_dev, train_idx, val_idx, _test_idx = _baseline_split_tensors(
        g,
        device=effective_device,
        feature_key=feature_key,
        label_key=label_key,
    )
    model = build_baseline(model_id, in_dim=in_dim, hparams=hparams)
    model = model.to(effective_device)

    class_w = class_weights_from_train_labels(y_dev[train_idx])
    if class_w is not None:
        class_w = class_w.to(effective_device)
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

    return BaselineModelArtifact(
        model_id=str(model_id),
        hparams=hparams,
        state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        threshold=float(th.threshold),
        device=str(effective_device),
        feature_key=feature_key,
        label_key=label_key,
    )


def eval_baseline_model(
    artifact: BaselineModelArtifact,
    g,
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    t0 = time.perf_counter()
    g, effective_device = _resolve_baseline_device(artifact.model_id, g, artifact.device)
    in_dim, x_dev, y_dev, _train_idx, _val_idx, test_idx = _baseline_split_tensors(
        g,
        device=effective_device,
        feature_key=artifact.feature_key,
        label_key=artifact.label_key,
    )
    model = build_baseline(artifact.model_id, in_dim=in_dim, hparams=artifact.hparams)
    model.load_state_dict(artifact.state_dict)
    model = model.to(effective_device)
    model.eval()

    with torch.no_grad():
        logits = _forward_logits(artifact.model_id, model, g, x_dev)
        probs = F.softmax(logits, dim=1)[:, 1]
        test_scores = probs[test_idx].detach().cpu().numpy()
        test_labels = y_dev[test_idx].detach().cpu().numpy()

    use_threshold = float(artifact.threshold if threshold is None else threshold)
    roc_auc = roc_auc_binary(test_labels, test_scores)
    ap = average_precision_binary(test_labels, test_scores)
    f1m = f1_macro_at_threshold(test_labels, test_scores, use_threshold)

    dt = time.perf_counter() - t0
    return {
        "roc_auc": float(roc_auc) if roc_auc is not None else float("nan"),
        "average_precision": float(ap) if ap is not None else float("nan"),
        "f1_macro": float(f1m),
        "threshold": use_threshold,
        "duration_sec": float(dt),
    }


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
    t0 = time.perf_counter()
    artifact = train_baseline_model(
        model_id,
        g,
        training_seed=training_seed,
        device=device,
        hparams=hparams,
        feature_key=feature_key,
        label_key=label_key,
    )
    out = eval_baseline_model(artifact, g)
    out["duration_sec"] = time.perf_counter() - t0
    return out


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
    selected_model_ids: list[str] | None = None,
) -> None:
    """Train/evaluate baseline models and append rows into results.csv."""
    out_dir = out_dir.resolve()
    results_csv = out_dir / "results.csv"
    variants_csv = out_dir / "graph_variants.csv"

    ensure_results_csv(results_csv, overwrite=bool(force))
    completed_keys = load_completed_keys(results_csv, retry_errors=bool(retry_errors)) if skip_existing else set()

    variants = require_variants_csv_rows(variants_csv)

    training_seeds = get_training_seeds(cfg)
    if max_training_seeds is not None:
        training_seeds = training_seeds[: int(max_training_seeds)]

    configured_baseline_model_ids = select_model_ids(
        cfg,
        supported_model_ids={"mlp", "sage"},
        requested_model_ids=None,
    )
    baseline_model_ids = select_model_ids(
        cfg,
        supported_model_ids={"mlp", "sage"},
        requested_model_ids=selected_model_ids,
        default_order=(["mlp", "sage"] if not configured_baseline_model_ids else None),
    )
    if not baseline_model_ids:
        warn_no_matching_models("baselines", selected_model_ids)
        return

    filtered = filter_variants(
        variants,
        include_noop=bool(include_noop),
        only_clean=bool(only_clean),
        max_variants=max_variants,
    )

    protocol = PROTOCOL_TRAIN_ON_VARIANT
    expected_keys = build_expected_run_keys(
        filtered,
        training_seeds=training_seeds,
        model_ids=baseline_model_ids,
        protocol=protocol,
    )
    preflight = summarize_training_preflight(
        results_csv,
        expected_keys=expected_keys,
        completed_keys=completed_keys,
        skip_existing=bool(skip_existing),
        model_ids=baseline_model_ids,
        protocol=protocol,
    )
    print_training_preflight(
        variant_count=len(filtered),
        training_seed_count=len(training_seeds),
        model_ids=baseline_model_ids,
        summary=preflight,
    )
    progress = ProgressTracker(stage_label="baselines", total_runs=preflight.total_runs, already_done=preflight.already_done)

    for v in filtered:
        pending_runs: list[tuple[str, int, tuple[str, str, str, float, int, int, str, str]]] = []

        for training_seed in training_seeds:
            for model_id in baseline_model_ids:
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
                pending_runs.append((str(model_id), int(training_seed), run_key))

        if not pending_runs:
            continue

        graph_t0 = time.perf_counter()
        try:
            g = load_graph_bin(Path(v.graph_path))
        except Exception as e:
            traceback.print_exc()
            dt = time.perf_counter() - graph_t0
            for model_id, training_seed, run_key in pending_runs:
                write_result_row(
                    results_csv,
                    v,
                    model_id=str(model_id),
                    training_seed=int(training_seed),
                    protocol=protocol,
                    metrics=None,
                    error=e,
                    duration_sec=float(dt),
                )
                progress.record(
                    model_id=str(model_id),
                    scenario_id=v.scenario_id,
                    severity=float(v.severity),
                    graph_seed=int(v.graph_seed),
                    training_seed=int(training_seed),
                    status="error",
                    duration_sec=float(dt),
                )
                completed_keys.add(run_key)
            continue

        for model_id, training_seed, run_key in pending_runs:
            run_t0 = time.perf_counter()
            try:
                hp = build_baseline_hparams(str(model_id), max_epochs=max_epochs, patience=patience)

                out = train_eval_baseline(
                    str(model_id),
                    g,
                    training_seed=int(training_seed),
                    device=str(device),
                    hparams=hp,
                )
                write_result_row(
                    results_csv,
                    v,
                    model_id=str(model_id),
                    training_seed=int(training_seed),
                    protocol=protocol,
                    metrics=out,
                )
                progress.record(
                    model_id=str(model_id),
                    scenario_id=v.scenario_id,
                    severity=float(v.severity),
                    graph_seed=int(v.graph_seed),
                    training_seed=int(training_seed),
                    status="ok",
                    duration_sec=float(out["duration_sec"]),
                    roc_auc=float(out["roc_auc"]),
                )
            except Exception as e:
                traceback.print_exc()
                dt = time.perf_counter() - run_t0
                write_result_row(
                    results_csv,
                    v,
                    model_id=str(model_id),
                    training_seed=int(training_seed),
                    protocol=protocol,
                    metrics=None,
                    error=e,
                    duration_sec=float(dt),
                )
                progress.record(
                    model_id=str(model_id),
                    scenario_id=v.scenario_id,
                    severity=float(v.severity),
                    graph_seed=int(v.graph_seed),
                    training_seed=int(training_seed),
                    status="error",
                    duration_sec=float(dt),
                )
            finally:
                completed_keys.add(run_key)

    summarize_results_by_training_seed(
        results_csv,
        out_csv_path=out_dir / "results_summary_baselines.csv",
        model_ids=set(baseline_model_ids),
        protocols={PROTOCOL_TRAIN_ON_VARIANT},
    )
