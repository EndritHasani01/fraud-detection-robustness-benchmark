from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LoadedDataset:
    dataset_id: str
    source_name: str
    graph: Any  # DGLGraph or DGLHeteroGraph


def _stratified_masks_from_labels(labels, train_size: float, val_size: float, seed: int):
    """Create train/val/test masks with approximate stratification using numpy only."""
    import numpy as np
    import torch

    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)
    labels = labels.reshape(-1)
    n = labels.shape[0]
    idx_all = np.arange(n)

    # Split per class.
    rng = np.random.default_rng(int(seed))
    unique = np.unique(labels)

    # Fractions in "rest" set after train split.
    rest_frac = 1.0 - float(train_size)
    if rest_frac <= 0:
        raise ValueError("train_size must be < 1.0")
    val_frac_in_rest = float(val_size) / rest_frac
    val_frac_in_rest = min(max(val_frac_in_rest, 0.0), 1.0)

    train_idx = []
    val_idx = []
    test_idx = []
    for u in unique.tolist():
        cls_idx = idx_all[labels == u]
        rng.shuffle(cls_idx)
        n_cls = cls_idx.shape[0]
        n_train = int(np.floor(train_size * n_cls))
        rest = cls_idx[n_train:]
        n_val = int(np.floor(val_frac_in_rest * rest.shape[0]))
        train_idx.append(cls_idx[:n_train])
        val_idx.append(rest[:n_val])
        test_idx.append(rest[n_val:])

    train_idx = np.concatenate(train_idx) if train_idx else np.array([], dtype=np.int64)
    val_idx = np.concatenate(val_idx) if val_idx else np.array([], dtype=np.int64)
    test_idx = np.concatenate(test_idx) if test_idx else np.array([], dtype=np.int64)

    # Shuffle within each split so order is not class-blocked.
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[torch.from_numpy(train_idx)] = True
    val_mask[torch.from_numpy(val_idx)] = True
    test_mask[torch.from_numpy(test_idx)] = True

    return train_mask, val_mask, test_mask


def load_dgl_fraud_dataset(dataset_id: str, source_name: str, raw_dir: Path, *, force_reload: bool = False) -> LoadedDataset:
    """Load a dataset from DGL's FraudDataset family.

    source_name should match DGL's naming, e.g. "yelp" or "amazon".
    """
    try:
        from dgl.data.fraud import FraudDataset
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "DGL FraudDataset is required. Install dgl and torch in a compatible Python environment "
            "(Python 3.10/3.11 is the safest choice for ML wheels)."
        ) from e

    ds = FraudDataset(name=source_name, raw_dir=str(raw_dir), force_reload=bool(force_reload))
    g = ds[0]
    return LoadedDataset(dataset_id=dataset_id, source_name=source_name, graph=g)


def ensure_masks(graph, *, train_size: float, val_size: float, split_seed: int, label_key: str = "label") -> None:
    """Ensure graph has train/val/test boolean masks; overwrites existing masks."""
    import numpy as np
    import torch

    labels = graph.ndata.get(label_key)
    if labels is None:
        raise RuntimeError(f"Graph is missing ndata['{label_key}']; cannot split.")

    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(labels)

    train_mask, val_mask, test_mask = _stratified_masks_from_labels(
        labels_np, train_size=float(train_size), val_size=float(val_size), seed=int(split_seed)
    )

    graph.ndata["train_mask"] = train_mask
    graph.ndata["val_mask"] = val_mask
    graph.ndata["test_mask"] = test_mask


def to_canonical_graph(graph, canonical_view: str):
    """Convert to the canonical graph view requested by the benchmark."""
    if canonical_view == "heterograph":
        return graph

    if canonical_view != "homogeneous":
        raise ValueError(f"Unsupported canonical_view: {canonical_view}")

    try:
        import dgl
    except Exception as e:  # pragma: no cover
        raise RuntimeError("DGL is required to convert graphs.") from e

    # Keep ndata needed downstream.
    keep_ndata = ["feature", "label", "train_mask", "val_mask", "test_mask"]
    return dgl.to_homogeneous(graph, ndata=[k for k in keep_ndata if k in graph.ndata])


def ensure_feature_dtype(graph, *, feature_key: str = "feature") -> None:
    """Cast feature tensor to float32 (DGL FraudDataset often uses float64)."""
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required for feature casting.") from e

    feat = graph.ndata.get(feature_key)
    if feat is None:
        raise RuntimeError(f"Graph is missing ndata['{feature_key}']; cannot proceed.")
    if not isinstance(feat, torch.Tensor):
        feat = torch.tensor(feat)
    if feat.dtype != torch.float32:
        graph.ndata[feature_key] = feat.float()


def ensure_label_dtype(graph, *, label_key: str = "label") -> None:
    """Cast label tensor to int64 and squeeze to (N,) if needed."""
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required for label casting.") from e

    lab = graph.ndata.get(label_key)
    if lab is None:
        raise RuntimeError(f"Graph is missing ndata['{label_key}']; cannot proceed.")
    if not isinstance(lab, torch.Tensor):
        lab = torch.tensor(lab)
    lab = lab.squeeze()
    if lab.dtype != torch.int64:
        lab = lab.long()
    graph.ndata[label_key] = lab
