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
from .variants import (
    VariantRow,
    class_weights_from_train_labels,
    filter_variants,
    load_graph_bin,
    read_variants_csv,
    set_seeds,
)


@dataclass(frozen=True)
class PMPModelArtifact:
    repo_root: Path
    cfg_pmp: dict[str, Any]
    state_dict: dict[str, Any]
    threshold: float
    device: str


def _row_normalize_features(x, *, eps: float = 0.01):
    import torch

    denom = x.sum(dim=1, keepdim=True) + float(eps)
    return x / denom


def _ensure_label_unk(g, *, label_key: str = "label") -> None:
    """Create PMP's label_unk encoding: 0/1 for train nodes, 2 for all others."""
    import torch

    y = g.ndata.get(label_key)
    if y is None:
        raise RuntimeError(f"Graph missing ndata['{label_key}']")
    y = y.squeeze().to(torch.int64)

    train_mask = g.ndata.get("train_mask")
    if train_mask is None:
        raise RuntimeError("Graph missing ndata['train_mask']")

    label_unk = torch.full((g.num_nodes(),), 2, dtype=torch.int64)
    train_idx = torch.nonzero(train_mask, as_tuple=True)[0]
    label_unk[train_idx] = y[train_idx]
    g.ndata["label_unk"] = label_unk


def _import_pmp_model(repo_root: Path):
    repo_root = repo_root.resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"PMP repo not found: {repo_root}")

    # Ensure we can `import model.LASAGE_S` from the research repo.
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from model.LASAGE_S import LASAGE_S  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Failed to import PMP model from Repos/PMP-master. "
            "If this is a dependency issue, install missing packages or patch the repo as needed."
        ) from e
    return LASAGE_S


def _load_pmp_yaml_config(repo_root: Path, *, dataset_source_name: str, model_name: str = "LA-SAGE-S") -> dict[str, Any]:
    try:
        import yaml
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load PMP configs.") from e

    yml = repo_root / "config" / f"{dataset_source_name}.yml"
    if not yml.exists():
        raise FileNotFoundError(f"PMP config not found: {yml}")

    cfg_all = yaml.safe_load(yml.read_text(encoding="utf-8"))
    if model_name not in cfg_all:
        raise KeyError(f"Model '{model_name}' not found in {yml.name}")
    cfg = dict(cfg_all[model_name])
    cfg["dataset"] = cfg_all.get("dataset", dataset_source_name)
    cfg["model_name"] = model_name
    return cfg


def resolve_pmp_config(
    repo_root: Path,
    *,
    dataset_source_name: str,
    model_cfg: dict[str, Any] | None,
    model_name: str = "LA-SAGE-S",
) -> dict[str, Any]:
    cfg_pmp = _load_pmp_yaml_config(repo_root, dataset_source_name=dataset_source_name, model_name=model_name)
    if model_cfg is None:
        return cfg_pmp

    raw_hparams = model_cfg.get("hparams")
    if raw_hparams is None:
        return cfg_pmp
    if not isinstance(raw_hparams, dict):
        raise RuntimeError("pmp model config hparams must be an object")

    resolved = dict(cfg_pmp)
    resolved.update(raw_hparams)
    return resolved
def _resolve_pmp_num_workers(cfg_pmp: dict[str, Any]) -> int:
    if sys.platform.startswith("win"):
        return 0

    raw = cfg_pmp.get("num_workers", 0)
    try:
        num_workers = int(raw)
    except Exception as e:
        raise RuntimeError("PMP num_workers must be an integer") from e
    return max(0, int(num_workers))


def _make_dataloaders(g, *, train_idx, val_idx, test_idx, cfg_pmp: dict[str, Any]):
    from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler, NeighborSampler

    n_layer = int(cfg_pmp.get("n_layer", 1))
    batch_size = int(cfg_pmp.get("batch_size", 512))
    val_batch_size = int(cfg_pmp.get("val_batch_size", batch_size))
    test_batch_size = int(cfg_pmp.get("test_batch_size", batch_size))
    num_workers = _resolve_pmp_num_workers(cfg_pmp)

    full_neighbors = bool(cfg_pmp.get("full_neighbors", True))
    sampled_neighbors = cfg_pmp.get("sampled_neighbors", [-1] * n_layer)
    if isinstance(sampled_neighbors, list):
        fanouts = [int(sampled_neighbors[i]) for i in range(min(len(sampled_neighbors), n_layer))]
        if len(fanouts) < n_layer:
            fanouts = fanouts + [-1] * (n_layer - len(fanouts))
    else:
        fanouts = [-1] * n_layer

    prefetch_node_feats = ["feature", "label_unk"]
    prefetch_labels = ["label"]

    if full_neighbors:
        sampler = MultiLayerFullNeighborSampler(
            n_layer, prefetch_node_feats=prefetch_node_feats, prefetch_labels=prefetch_labels
        )
    else:
        sampler = NeighborSampler(
            fanouts,
            edge_dir=str(cfg_pmp.get("sampling_type", "in")),
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
        )

    # Windows uses num_workers=0 for DGL stability; Linux/macOS can use the config value.
    train_loader = DataLoader(
        g,
        train_idx,
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        g,
        val_idx,
        sampler,
        batch_size=val_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        g,
        test_idx,
        sampler,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


def _predict_probs(model, relations, loader, *, device: str):
    import numpy as np
    import torch
    import torch.nn.functional as F

    model.eval()
    y_true_parts = []
    y_score_parts = []

    with torch.no_grad():
        for _input_nodes, _output_nodes, blocks in loader:
            blocks = [b.to(device) for b in blocks]
            feats = blocks[0].srcdata["feature"].to(device)
            logits = model(blocks, relations, feats)
            probs = F.softmax(logits, dim=1)[:, 1]
            y = blocks[-1].dstdata["label"].to(device).squeeze().to(torch.int64)

            y_true_parts.append(y.detach().cpu().numpy())
            y_score_parts.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(y_true_parts, axis=0) if y_true_parts else np.array([], dtype=np.int64)
    y_score = np.concatenate(y_score_parts, axis=0) if y_score_parts else np.array([], dtype=np.float64)
    return y_true, y_score


def _resolve_requested_device(device: str) -> str:
    import torch

    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    return str(device)


def _build_pmp_model(g, *, repo_root: Path, cfg_pmp: dict[str, Any], device: str):
    import torch

    LASAGE_S = _import_pmp_model(repo_root)

    feat_dim = int(g.ndata["feature"].shape[1])
    y = g.ndata["label"].squeeze().to(torch.int64)
    num_classes = int(torch.unique(y).numel())
    relations = list(getattr(g, "etypes", []))
    num_relations = max(1, len(relations))

    mlp_act = str(cfg_pmp.get("mlp_activation", "relu")).lower()
    if mlp_act == "elu":
        mlp_activation = torch.nn.ELU(inplace=True)
    else:
        mlp_activation = torch.nn.ReLU(inplace=True)

    proj = bool(cfg_pmp.get("proj", True))
    hid_dim = int(cfg_pmp.get("hid_dim", 48))
    n_layer = int(cfg_pmp.get("n_layer", 1))
    dropout = float(cfg_pmp.get("dropout", 0.0))
    num_trans = int(cfg_pmp.get("num_trans", 1))
    agg = str(cfg_pmp.get("agg", "mean"))
    relation_agg = str(cfg_pmp.get("relation_agg", "cat"))

    model = LASAGE_S(
        in_size=feat_dim,
        hid_size=hid_dim,
        out_size=(num_classes if not proj else hid_dim),
        num_layers=n_layer,
        dropout=dropout,
        proj=proj,
        num_relations=num_relations,
        batch_size=int(cfg_pmp.get("batch_size", 512)),
        num_trans=num_trans,
        mlp_activation=mlp_activation,
        out_proj_size=num_classes,
        agg=agg,
        relation_agg=relation_agg,
    )

    effective_device = _resolve_requested_device(device)
    if effective_device == "cuda":
        try:
            model = model.to("cuda")
        except Exception:
            effective_device = "cpu"
            model = model.to("cpu")
    else:
        model = model.to("cpu")
    return model, relations, effective_device


def train_pmp_model(
    g,
    *,
    repo_root: Path,
    cfg_pmp: dict[str, Any],
    training_seed: int,
    device: str,
    max_epochs: int | None,
    patience: int | None,
) -> PMPModelArtifact:
    import copy

    import torch

    set_seeds(int(training_seed))

    orig_x = g.ndata.get("feature")
    if orig_x is None:
        raise RuntimeError("Graph missing ndata['feature']")
    orig_label_unk = g.ndata.get("label_unk")

    try:
        _ensure_label_unk(g)
        if bool(cfg_pmp.get("norm_feat", True)):
            g.ndata["feature"] = _row_normalize_features(orig_x.to(torch.float32)).to(torch.float32)

        train_mask = g.ndata.get("train_mask")
        val_mask = g.ndata.get("val_mask")
        test_mask = g.ndata.get("test_mask")
        if train_mask is None or val_mask is None or test_mask is None:
            raise RuntimeError("Graph is missing train/val/test masks.")

        train_idx = torch.nonzero(train_mask, as_tuple=True)[0]
        val_idx = torch.nonzero(val_mask, as_tuple=True)[0]
        test_idx = torch.nonzero(test_mask, as_tuple=True)[0]

        train_loader, val_loader, test_loader = _make_dataloaders(
            g, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, cfg_pmp=cfg_pmp
        )

        y = g.ndata["label"].squeeze().to(torch.int64)
        model, relations, effective_device = _build_pmp_model(g, repo_root=repo_root, cfg_pmp=cfg_pmp, device=device)

        lr = float(cfg_pmp.get("lr", 0.01))
        weight_decay = float(cfg_pmp.get("weight_decay", 0.0))
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        y_train = y[train_idx]
        class_w = class_weights_from_train_labels(y_train)
        if class_w is not None:
            class_w = class_w.to(effective_device)

        loss_fn = torch.nn.CrossEntropyLoss(weight=class_w if bool(cfg_pmp.get("weighted_loss", False)) else None)

        epochs = int(cfg_pmp.get("epochs", 100))
        if max_epochs is not None:
            epochs = int(max_epochs)
        es_patience = int(cfg_pmp.get("patience", 10))
        if patience is not None:
            es_patience = int(patience)

        best_monitor = -float("inf")
        best_state = None
        bad_epochs = 0

        for _epoch in range(epochs):
            model.train()
            for _in_nodes, _out_nodes, blocks in train_loader:
                blocks = [b.to(effective_device) for b in blocks]
                feats = blocks[0].srcdata["feature"].to(effective_device)
                labels = blocks[-1].dstdata["label"].to(effective_device).squeeze().to(torch.int64)

                logits = model(blocks, relations, feats)
                loss = loss_fn(logits, labels)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            # Validation monitor: ROC-AUC on val probabilities.
            y_val, s_val = _predict_probs(model, relations, val_loader, device=effective_device)
            val_auc = roc_auc_binary(y_val, s_val)
            monitor = float(val_auc) if math.isfinite(val_auc) else -float("inf")

            if monitor > best_monitor:
                best_monitor = monitor
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
                bad_epochs = 0
            else:
                bad_epochs += 1
                if es_patience > 0 and bad_epochs >= es_patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        y_val, s_val = _predict_probs(model, relations, val_loader, device=effective_device)
        th = best_f1_macro_threshold(y_val, s_val)
        return PMPModelArtifact(
            repo_root=repo_root,
            cfg_pmp=dict(cfg_pmp),
            state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            threshold=float(th.threshold),
            device=str(device),
        )
    finally:
        g.ndata["feature"] = orig_x
        if orig_label_unk is None:
            if "label_unk" in g.ndata:
                del g.ndata["label_unk"]
        else:
            g.ndata["label_unk"] = orig_label_unk


def eval_pmp_model(
    artifact: PMPModelArtifact,
    g,
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    import torch

    t0 = time.perf_counter()

    orig_x = g.ndata.get("feature")
    if orig_x is None:
        raise RuntimeError("Graph missing ndata['feature']")
    orig_label_unk = g.ndata.get("label_unk")

    try:
        _ensure_label_unk(g)
        if bool(artifact.cfg_pmp.get("norm_feat", True)):
            g.ndata["feature"] = _row_normalize_features(orig_x.to(torch.float32)).to(torch.float32)

        train_mask = g.ndata.get("train_mask")
        val_mask = g.ndata.get("val_mask")
        test_mask = g.ndata.get("test_mask")
        if train_mask is None or val_mask is None or test_mask is None:
            raise RuntimeError("Graph is missing train/val/test masks.")

        train_idx = torch.nonzero(train_mask, as_tuple=True)[0]
        val_idx = torch.nonzero(val_mask, as_tuple=True)[0]
        test_idx = torch.nonzero(test_mask, as_tuple=True)[0]
        _train_loader, _val_loader, test_loader = _make_dataloaders(
            g, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, cfg_pmp=artifact.cfg_pmp
        )
        model, relations, effective_device = _build_pmp_model(
            g,
            repo_root=artifact.repo_root,
            cfg_pmp=artifact.cfg_pmp,
            device=artifact.device,
        )
        model.load_state_dict(artifact.state_dict)
        y_test, s_test = _predict_probs(model, relations, test_loader, device=effective_device)
        use_threshold = float(artifact.threshold if threshold is None else threshold)
        roc_auc = roc_auc_binary(y_test, s_test)
        ap = average_precision_binary(y_test, s_test)
        f1m = f1_macro_at_threshold(y_test, s_test, use_threshold)
        return {
            "roc_auc": float(roc_auc) if roc_auc is not None else float("nan"),
            "average_precision": float(ap) if ap is not None else float("nan"),
            "f1_macro": float(f1m),
            "threshold": use_threshold,
            "duration_sec": float(time.perf_counter() - t0),
        }
    finally:
        g.ndata["feature"] = orig_x
        if orig_label_unk is None:
            if "label_unk" in g.ndata:
                del g.ndata["label_unk"]
        else:
            g.ndata["label_unk"] = orig_label_unk


def train_eval_pmp(
    g,
    *,
    repo_root: Path,
    cfg_pmp: dict[str, Any],
    training_seed: int,
    device: str,
    max_epochs: int | None,
    patience: int | None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    artifact = train_pmp_model(
        g,
        repo_root=repo_root,
        cfg_pmp=cfg_pmp,
        training_seed=training_seed,
        device=device,
        max_epochs=max_epochs,
        patience=patience,
    )
    out = eval_pmp_model(artifact, g)
    out["duration_sec"] = time.perf_counter() - t0
    return out


def run_pmp_stage(
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
    out_dir = out_dir.resolve()
    results_csv = out_dir / "results.csv"
    variants_csv = out_dir / "graph_variants.csv"

    ensure_results_csv(results_csv, overwrite=bool(force))
    completed_keys = load_completed_keys(results_csv, retry_errors=bool(retry_errors)) if skip_existing else set()

    variants = read_variants_csv(variants_csv)
    if not variants:
        raise RuntimeError(f"No rows found in {variants_csv}")

    dataset_cfg_by_id = {d["dataset_id"]: d for d in cfg.get("datasets", [])}

    pmp_model_cfg = None
    for m in cfg.get("models", []):
        if str(m.get("model_id", "")) == "pmp":
            pmp_model_cfg = m
            break
    if not pmp_model_cfg:
        raise RuntimeError("Config has no model entry with model_id='pmp'")

    stage_model_ids = select_model_ids(
        cfg,
        supported_model_ids={"pmp"},
        requested_model_ids=selected_model_ids,
    )
    if not stage_model_ids:
        warn_no_matching_models("pmp", selected_model_ids)
        return

    repo_root = Path(str(pmp_model_cfg.get("repo_path", "")))
    if not repo_root.exists():
        raise FileNotFoundError(f"Configured PMP repo_path does not exist: {repo_root}")

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
    progress = ProgressTracker(stage_label="pmp", total_runs=preflight.total_runs, already_done=preflight.already_done)

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
                model_id="pmp",
                protocol=protocol,
            )
            if run_key in completed_keys:
                print(
                    f"[skip] pmp / {v.scenario_id} / sev={float(v.severity):g} / "
                    f"gs={int(v.graph_seed)} / ts={int(training_seed)} already done"
                )
                continue
            pending_runs.append((int(training_seed), run_key))

        if not pending_runs:
            continue

        ds_cfg = dataset_cfg_by_id.get(v.dataset_id, {})
        dataset_source_name = str(ds_cfg.get("source_name", "yelp")).strip().lower()
        cfg_pmp = resolve_pmp_config(
            repo_root,
            dataset_source_name=dataset_source_name,
            model_cfg=pmp_model_cfg,
            model_name="LA-SAGE-S",
        )

        graph_t0 = time.perf_counter()
        try:
            g = load_graph_bin(Path(v.graph_path))
        except Exception as e:
            dt = time.perf_counter() - graph_t0
            for training_seed, run_key in pending_runs:
                write_result_row(
                    results_csv,
                    v,
                    model_id="pmp",
                    training_seed=int(training_seed),
                    protocol=protocol,
                    metrics=None,
                    error=e,
                    duration_sec=float(dt),
                )
                progress.record(
                    model_id="pmp",
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
                out = train_eval_pmp(
                    g,
                    repo_root=repo_root,
                    cfg_pmp=cfg_pmp,
                    training_seed=int(training_seed),
                    device=str(device),
                    max_epochs=max_epochs,
                    patience=patience,
                )
                write_result_row(
                    results_csv,
                    v,
                    model_id="pmp",
                    training_seed=int(training_seed),
                    protocol=protocol,
                    metrics=out,
                )
                progress.record(
                    model_id="pmp",
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
                    model_id="pmp",
                    training_seed=int(training_seed),
                    protocol=protocol,
                    metrics=None,
                    error=e,
                    duration_sec=float(dt),
                )
                progress.record(
                    model_id="pmp",
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
        out_csv_path=out_dir / "results_summary_pmp.csv",
        model_ids={"pmp"},
    )
