from __future__ import annotations

from typing import Any


def _to_1d_label_tensor(labels):
    # Supports labels shaped (N, 1) or (N,).
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required for graph stats that use tensors.") from e

    if labels is None:
        return None
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    labels = labels.squeeze()
    if labels.ndim != 1:
        raise ValueError(f"Unsupported label shape: {tuple(labels.shape)}")
    return labels


def compute_graph_stats(g, *, label_key: str = "label") -> dict[str, Any]:
    """Compute basic stats for a (homogeneous) DGL graph."""
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required to compute graph stats.") from e

    n_nodes = int(g.num_nodes())
    n_edges = int(g.num_edges())

    in_deg = g.in_degrees().to(torch.int64)
    out_deg = g.out_degrees().to(torch.int64)

    def _mean(x):
        return float(torch.mean(x.float()).item()) if x.numel() else 0.0

    def _median(x):
        if not x.numel():
            return 0.0
        return float(torch.median(x).item())

    labels = g.ndata.get(label_key)
    labels = _to_1d_label_tensor(labels)

    hetero_ratio = None
    pos_rate = None
    if labels is not None and n_edges > 0:
        src, dst = g.edges()
        y_src = labels[src]
        y_dst = labels[dst]
        hetero_ratio = float(torch.mean((y_src != y_dst).float()).item())

        # If labels are binary 0/1, this is fraud/anomaly prevalence.
        # If labels are non-binary, this is still a useful "mean label" scalar.
        pos_rate = float(torch.mean(labels.float()).item())

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "mean_in_degree": _mean(in_deg),
        "median_in_degree": _median(in_deg),
        "mean_out_degree": _mean(out_deg),
        "median_out_degree": _median(out_deg),
        "heterophily_ratio": hetero_ratio,
        "pos_rate": pos_rate,
    }

