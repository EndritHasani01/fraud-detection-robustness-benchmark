from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import get_training_seeds
from .metrics import average_precision_binary, best_f1_macro_threshold, f1_macro_at_threshold, roc_auc_binary
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
from .variants import VariantRow, filter_variants, load_graph_bin, read_variants_csv, set_seeds

SECGFD_DEFAULT_HPARAMS = {
    "hid_dim": 32,
    "order_d": 2,
    "high_order": 1,
    "lemda": 0.2,
    "lr": 0.01,
    "weight_decay": 0.0,
}
SECGFD_DEFAULT_MAX_EPOCHS = 50
SECGFD_DEFAULT_PATIENCE = 10

_SECGFD_THETA_CACHE: dict[tuple[str, int], tuple[tuple[float, ...], ...]] = {}
@dataclass(frozen=True)
class SECGFDModelArtifact:
    repo_root: Path
    state_dict: dict[str, Any]
    threshold: float
    device: str
    hid_dim: int
    order_d: int
    high_order: int
    lemda: float
    lr: float
    weight_decay: float


def _coerce_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise RuntimeError(f"{name} must be > 0")
    return value


def _coerce_non_negative_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{name} must be a number")
    value = float(value)
    if value < 0.0:
        raise RuntimeError(f"{name} must be >= 0")
    return value


def _coerce_positive_float(value: Any, *, name: str) -> float:
    value = _coerce_non_negative_float(value, name=name)
    if value <= 0.0:
        raise RuntimeError(f"{name} must be > 0")
    return value


def resolve_secgfd_hparams(
    model_cfg: dict[str, Any],
    *,
    hid_dim_override: int | None = None,
    order_d_override: int | None = None,
    high_order_override: int | None = None,
) -> dict[str, Any]:
    out = dict(SECGFD_DEFAULT_HPARAMS)

    raw_hparams = model_cfg.get("hparams")
    if raw_hparams is not None:
        if not isinstance(raw_hparams, dict):
            raise RuntimeError("secgfd model config hparams must be an object")
        if "hid_dim" in raw_hparams:
            out["hid_dim"] = _coerce_positive_int(raw_hparams["hid_dim"], name="secgfd.hparams.hid_dim")
        if "order_d" in raw_hparams:
            out["order_d"] = _coerce_positive_int(raw_hparams["order_d"], name="secgfd.hparams.order_d")
        if "high_order" in raw_hparams:
            out["high_order"] = _coerce_positive_int(raw_hparams["high_order"], name="secgfd.hparams.high_order")
        if "lemda" in raw_hparams:
            out["lemda"] = _coerce_non_negative_float(raw_hparams["lemda"], name="secgfd.hparams.lemda")
        if "lr" in raw_hparams:
            out["lr"] = _coerce_positive_float(raw_hparams["lr"], name="secgfd.hparams.lr")
        if "weight_decay" in raw_hparams:
            out["weight_decay"] = _coerce_non_negative_float(
                raw_hparams["weight_decay"], name="secgfd.hparams.weight_decay"
            )

    if hid_dim_override is not None:
        out["hid_dim"] = _coerce_positive_int(hid_dim_override, name="--secgfd-hid-dim")
    if order_d_override is not None:
        out["order_d"] = _coerce_positive_int(order_d_override, name="--secgfd-order-d")
    if high_order_override is not None:
        out["high_order"] = _coerce_positive_int(high_order_override, name="--secgfd-high-order")
    return out


def resolve_secgfd_training_controls(*, max_epochs: int | None, patience: int | None) -> tuple[int, int]:
    eff_max_epochs = SECGFD_DEFAULT_MAX_EPOCHS if max_epochs is None else int(max_epochs)
    eff_patience = SECGFD_DEFAULT_PATIENCE if patience is None else int(patience)
    if eff_max_epochs <= 0:
        raise RuntimeError("SEC-GFD max_epochs must be > 0")
    if eff_patience < 0:
        raise RuntimeError("SEC-GFD patience must be >= 0")
    return eff_max_epochs, eff_patience


def _import_secgfd_module(repo_root: Path):
    import importlib.util

    repo_root = repo_root.resolve()
    secgfd_py = repo_root / "model" / "SECGFD.py"
    if not secgfd_py.exists():
        raise FileNotFoundError(f"SEC-GFD model file not found: {secgfd_py}")

    # Load under a unique module name to avoid collisions with other repos that
    # use a top-level package name like 'model'.
    mod_name = "secgfd_repo_model_SECGFD"
    existing = sys.modules.get(mod_name)
    if existing is not None and Path(getattr(existing, "__file__", "")).resolve() == secgfd_py.resolve():
        return existing

    spec = importlib.util.spec_from_file_location(mod_name, secgfd_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for: {secgfd_py}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _install_secgfd_theta_cache(secgfd_module: Any) -> None:
    if getattr(secgfd_module, "_benchmark_theta_cache_enabled", False):
        return
    original_calculate_theta2 = getattr(secgfd_module, "calculate_theta2", None)
    if original_calculate_theta2 is None:
        raise RuntimeError("SEC-GFD module did not define calculate_theta2")
    module_file = str(Path(getattr(secgfd_module, "__file__", "secgfd_module")).resolve())

    def _cached_calculate_theta2(*args, **kwargs):
        if "d" in kwargs:
            order_d = int(kwargs["d"])
        elif args:
            order_d = int(args[0])
        else:
            raise TypeError("calculate_theta2 expects a polynomial degree 'd'")

        cache_key = (module_file, order_d)
        cached = _SECGFD_THETA_CACHE.get(cache_key)
        if cached is None:
            computed = original_calculate_theta2(*args, **kwargs)
            cached = tuple(tuple(float(coeff) for coeff in theta) for theta in computed)
            _SECGFD_THETA_CACHE[cache_key] = cached
        return [list(theta) for theta in cached]

    setattr(secgfd_module, "_benchmark_original_calculate_theta2", original_calculate_theta2)
    setattr(secgfd_module, "calculate_theta2", _cached_calculate_theta2)
    setattr(secgfd_module, "_benchmark_theta_cache_enabled", True)


def _import_secgfd_class(repo_root: Path):
    mod = _import_secgfd_module(repo_root)
    _install_secgfd_theta_cache(mod)
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


def _resolve_secgfd_graph_device(g, *, device: str):
    import torch

    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if device == "cuda":
        try:
            g = g.to("cuda")
        except Exception:
            device = "cpu"
            g = g.to("cpu")
    else:
        g = g.to("cpu")
    return g, str(device)


def _build_secgfd_model(
    g,
    *,
    repo_root: Path,
    device: str,
    hid_dim: int,
    order_d: int,
    high_order: int,
):
    SECGFD = _import_secgfd_class(repo_root)
    in_dim = int(g.ndata["feature"].shape[1])
    model = SECGFD(in_dim, int(hid_dim), 2, g, d=int(order_d), high_order=int(high_order)).to(device)
    return model


def train_secgfd_model(
    g,
    *,
    repo_root: Path,
    training_seed: int,
    device: str,
    max_epochs: int | None,
    patience: int | None,
    hid_dim: int = 32,
    order_d: int = 2,
    high_order: int = 1,
    lemda: float = 0.2,
    lr: float = 0.01,
    weight_decay: float = 0.0,
) -> SECGFDModelArtifact:
    import copy

    import torch
    import torch.nn.functional as F

    set_seeds(int(training_seed))
    g, effective_device = _resolve_secgfd_graph_device(g, device=device)

    features = g.ndata.get("feature")
    labels = g.ndata.get("label")
    if features is None or labels is None:
        raise RuntimeError("Graph missing ndata['feature'] or ndata['label']")
    features = features.to(torch.float32).to(effective_device)
    labels = labels.squeeze().to(torch.int64).to(effective_device)

    train_mask = g.ndata.get("train_mask")
    val_mask = g.ndata.get("val_mask")
    test_mask = g.ndata.get("test_mask")
    if train_mask is None or val_mask is None or test_mask is None:
        raise RuntimeError("Graph is missing train/val/test masks.")
    train_mask = train_mask.to(effective_device)
    val_mask = val_mask.to(effective_device)
    test_mask = test_mask.to(effective_device)

    train_idx = torch.nonzero(train_mask, as_tuple=True)[0]
    val_idx = torch.nonzero(val_mask, as_tuple=True)[0]
    test_idx = torch.nonzero(test_mask, as_tuple=True)[0]
    if train_idx.numel() == 0 or val_idx.numel() == 0 or test_idx.numel() == 0:
        raise RuntimeError("One of the splits is empty; cannot train/evaluate.")

    model = _build_secgfd_model(
        g,
        repo_root=repo_root,
        device=effective_device,
        hid_dim=hid_dim,
        order_d=order_d,
        high_order=high_order,
    )

    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    # Class imbalance weight on training split.
    pos_w = _class_weight_pos(labels[train_idx])
    ce_weight = torch.tensor([1.0, float(pos_w)], dtype=torch.float32, device=effective_device)

    epochs = SECGFD_DEFAULT_MAX_EPOCHS if max_epochs is None else int(max_epochs)
    es_patience = SECGFD_DEFAULT_PATIENCE if patience is None else int(patience)

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

    return SECGFDModelArtifact(
        repo_root=repo_root,
        state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        threshold=float(th.threshold),
        device=str(device),
        hid_dim=int(hid_dim),
        order_d=int(order_d),
        high_order=int(high_order),
        lemda=float(lemda),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )


def eval_secgfd_model(
    artifact: SECGFDModelArtifact,
    g,
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    import torch

    t0 = time.perf_counter()
    g, effective_device = _resolve_secgfd_graph_device(g, device=artifact.device)

    features = g.ndata.get("feature")
    labels = g.ndata.get("label")
    if features is None or labels is None:
        raise RuntimeError("Graph missing ndata['feature'] or ndata['label']")
    features = features.to(torch.float32).to(effective_device)
    labels = labels.squeeze().to(torch.int64).to(effective_device)

    train_mask = g.ndata.get("train_mask")
    val_mask = g.ndata.get("val_mask")
    test_mask = g.ndata.get("test_mask")
    if train_mask is None or val_mask is None or test_mask is None:
        raise RuntimeError("Graph is missing train/val/test masks.")
    test_idx = torch.nonzero(test_mask.to(effective_device), as_tuple=True)[0]
    if test_idx.numel() == 0:
        raise RuntimeError("Test split is empty; cannot evaluate.")

    model = _build_secgfd_model(
        g,
        repo_root=artifact.repo_root,
        device=effective_device,
        hid_dim=artifact.hid_dim,
        order_d=artifact.order_d,
        high_order=artifact.high_order,
    )
    model.load_state_dict(artifact.state_dict)
    model.eval()
    with torch.no_grad():
        logits, _emb = model(features)
        probs = torch.softmax(logits, dim=1)[:, 1]
        test_scores = probs[test_idx].detach().cpu().numpy()
        test_labels = labels[test_idx].detach().cpu().numpy()

    use_threshold = float(artifact.threshold if threshold is None else threshold)
    roc_auc = roc_auc_binary(test_labels, test_scores)
    ap = average_precision_binary(test_labels, test_scores)
    f1m = f1_macro_at_threshold(test_labels, test_scores, use_threshold)
    return {
        "roc_auc": float(roc_auc) if roc_auc is not None else float("nan"),
        "average_precision": float(ap) if ap is not None else float("nan"),
        "f1_macro": float(f1m),
        "threshold": use_threshold,
        "duration_sec": float(time.perf_counter() - t0),
    }


def train_eval_secgfd(
    g,
    *,
    repo_root: Path,
    training_seed: int,
    device: str,
    max_epochs: int | None,
    patience: int | None,
    hid_dim: int = 32,
    order_d: int = 2,
    high_order: int = 1,
    lemda: float = 0.2,
    lr: float = 0.01,
    weight_decay: float = 0.0,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    artifact = train_secgfd_model(
        g,
        repo_root=repo_root,
        training_seed=training_seed,
        device=device,
        max_epochs=max_epochs,
        patience=patience,
        hid_dim=hid_dim,
        order_d=order_d,
        high_order=high_order,
        lemda=lemda,
        lr=lr,
        weight_decay=weight_decay,
    )
    out = eval_secgfd_model(artifact, g)
    out["duration_sec"] = time.perf_counter() - t0
    return out


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
    secgfd_hid_dim: int | None = None,
    secgfd_order_d: int | None = None,
    secgfd_high_order: int | None = None,
    selected_model_ids: list[str] | None = None,
) -> None:
    out_dir = out_dir.resolve()
    results_csv = out_dir / "results.csv"
    variants_csv = out_dir / "graph_variants.csv"

    ensure_results_csv(results_csv, overwrite=bool(force))
    completed_keys = load_completed_keys(results_csv, retry_errors=bool(retry_errors)) if skip_existing else set()

    variants = read_variants_csv(variants_csv)
    if not variants:
        raise RuntimeError(f"No rows found in {variants_csv}")

    sec_cfg = None
    for m in cfg.get("models", []):
        if str(m.get("model_id", "")) == "secgfd":
            sec_cfg = m
            break
    if not sec_cfg:
        raise RuntimeError("Config has no model entry with model_id='secgfd'")

    stage_model_ids = select_model_ids(
        cfg,
        supported_model_ids={"secgfd"},
        requested_model_ids=selected_model_ids,
    )
    if not stage_model_ids:
        warn_no_matching_models("secgfd", selected_model_ids)
        return

    repo_root = Path(str(sec_cfg.get("repo_path", "")))
    if not repo_root.exists():
        raise FileNotFoundError(f"Configured SEC-GFD repo_path does not exist: {repo_root}")
    secgfd_hparams = resolve_secgfd_hparams(
        sec_cfg,
        hid_dim_override=secgfd_hid_dim,
        order_d_override=secgfd_order_d,
        high_order_override=secgfd_high_order,
    )
    effective_max_epochs, effective_patience = resolve_secgfd_training_controls(
        max_epochs=max_epochs,
        patience=patience,
    )

    training_seeds = get_training_seeds(cfg)
    if max_training_seeds is not None:
        training_seeds = training_seeds[: int(max_training_seeds)]

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
        model_ids=stage_model_ids,
        protocol=protocol,
    )
    preflight = summarize_training_preflight(
        results_csv,
        expected_keys=expected_keys,
        completed_keys=completed_keys,
        skip_existing=bool(skip_existing),
        model_ids=stage_model_ids,
        protocol=protocol,
    )
    print_training_preflight(
        variant_count=len(filtered),
        training_seed_count=len(training_seeds),
        model_ids=stage_model_ids,
        summary=preflight,
    )
    progress = ProgressTracker(stage_label="secgfd", total_runs=preflight.total_runs, already_done=preflight.already_done)

    for v in filtered:
        pending_runs: list[tuple[int, tuple[str, str, str, float, int, int, str, str]]] = []

        for training_seed in training_seeds:
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
            pending_runs.append((int(training_seed), run_key))

        if not pending_runs:
            continue

        graph_t0 = time.perf_counter()
        try:
            g = load_graph_bin(Path(v.graph_path))
        except Exception as e:
            dt = time.perf_counter() - graph_t0
            for training_seed, run_key in pending_runs:
                write_result_row(
                    results_csv,
                    v,
                    model_id="secgfd",
                    training_seed=int(training_seed),
                    protocol=protocol,
                    metrics=None,
                    error=e,
                    duration_sec=float(dt),
                )
                progress.record(
                    model_id="secgfd",
                    scenario_id=v.scenario_id,
                    severity=float(v.severity),
                    graph_seed=int(v.graph_seed),
                    training_seed=int(training_seed),
                    status="error",
                    duration_sec=float(dt),
                )
                completed_keys.add(run_key)
            continue

        for training_seed, run_key in pending_runs:
            run_t0 = time.perf_counter()
            try:
                out = train_eval_secgfd(
                    g,
                    repo_root=repo_root,
                    training_seed=int(training_seed),
                    device=str(device),
                    max_epochs=effective_max_epochs,
                    patience=effective_patience,
                    hid_dim=int(secgfd_hparams["hid_dim"]),
                    order_d=int(secgfd_hparams["order_d"]),
                    high_order=int(secgfd_hparams["high_order"]),
                    lemda=float(secgfd_hparams["lemda"]),
                    lr=float(secgfd_hparams["lr"]),
                    weight_decay=float(secgfd_hparams["weight_decay"]),
                )
                write_result_row(
                    results_csv,
                    v,
                    model_id="secgfd",
                    training_seed=int(training_seed),
                    protocol=protocol,
                    metrics=out,
                )
                progress.record(
                    model_id="secgfd",
                    scenario_id=v.scenario_id,
                    severity=float(v.severity),
                    graph_seed=int(v.graph_seed),
                    training_seed=int(training_seed),
                    status="ok",
                    duration_sec=float(out["duration_sec"]),
                    roc_auc=float(out["roc_auc"]),
                )
            except Exception as e:
                dt = time.perf_counter() - run_t0
                write_result_row(
                    results_csv,
                    v,
                    model_id="secgfd",
                    training_seed=int(training_seed),
                    protocol=protocol,
                    metrics=None,
                    error=e,
                    duration_sec=float(dt),
                )
                progress.record(
                    model_id="secgfd",
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
        out_csv_path=out_dir / "results_summary_secgfd.csv",
        model_ids={"secgfd"},
    )
