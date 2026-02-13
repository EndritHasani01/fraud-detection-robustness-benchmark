from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    family: str
    method: str
    oracle_labels: bool
    severity_param: str
    severity: float
    graph_seed: int
    params: dict[str, Any]

def _stable_u32_seed(text: str) -> int:
    # Stable across runs and Python versions.
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _rng_for_spec(spec: ScenarioSpec):
    import numpy as np

    params_str = json.dumps(spec.params or {}, sort_keys=True, separators=(",", ":"))
    seed_text = f"{spec.scenario_id}|{spec.method}|{spec.graph_seed}|{spec.severity:.8f}|{params_str}"
    return np.random.default_rng(_stable_u32_seed(seed_text))


def _copy_ndata(dst_g, src_g) -> None:
    for k, v in src_g.ndata.items():
        dst_g.ndata[k] = v


def _copy_edata_if_compatible(dst_g, src_g) -> None:
    # Optional: only copy if edge counts match.
    if int(dst_g.num_edges()) != int(src_g.num_edges()):
        return
    for k, v in src_g.edata.items():
        dst_g.edata[k] = v


def _assert_invariants(base_g, new_g, *, feature_key: str, label_key: str) -> None:
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required to validate perturbation invariants.") from e

    if int(base_g.num_nodes()) != int(new_g.num_nodes()):
        raise RuntimeError("Invariant violated: node count changed.")

    for mask_key in ["train_mask", "val_mask", "test_mask"]:
        if mask_key in base_g.ndata:
            if mask_key not in new_g.ndata:
                raise RuntimeError(f"Invariant violated: missing ndata['{mask_key}'] on perturbed graph.")
            if not torch.equal(base_g.ndata[mask_key].cpu(), new_g.ndata[mask_key].cpu()):
                raise RuntimeError(f"Invariant violated: ndata['{mask_key}'] changed.")

    if label_key in base_g.ndata:
        if label_key not in new_g.ndata:
            raise RuntimeError(f"Invariant violated: missing ndata['{label_key}'] on perturbed graph.")
        if not torch.equal(base_g.ndata[label_key].cpu().squeeze(), new_g.ndata[label_key].cpu().squeeze()):
            raise RuntimeError(f"Invariant violated: ndata['{label_key}'] changed.")

    if feature_key in base_g.ndata:
        if feature_key not in new_g.ndata:
            raise RuntimeError(f"Invariant violated: missing ndata['{feature_key}'] on perturbed graph.")
        if tuple(base_g.ndata[feature_key].shape) != tuple(new_g.ndata[feature_key].shape):
            raise RuntimeError(f"Invariant violated: ndata['{feature_key}'] shape changed.")


def _rewire_edge_dst_to_opposite_label(base_g, *, p_rewire: float, rng, label_key: str) -> tuple[Any, dict[str, Any]]:
    if p_rewire <= 0.0:
        return base_g, {"n_rewired_edges": 0}

    try:
        import torch
        import dgl
    except Exception as e:  # pragma: no cover
        raise RuntimeError("DGL and PyTorch are required for edge rewiring perturbations.") from e

    labels = base_g.ndata.get(label_key)
    if labels is None:
        raise RuntimeError(f"Graph is missing ndata['{label_key}']; cannot apply heterophily rewiring.")
    labels = labels.squeeze().to(torch.int64).cpu()

    src, dst = base_g.edges()
    src = src.to(torch.int64).cpu()
    dst = dst.to(torch.int64).cpu()
    m = int(src.shape[0])
    n_rewire = int(p_rewire * m)
    if n_rewire <= 0:
        return base_g, {"n_rewired_edges": 0}

    hetero_before = float(torch.mean((labels[src] != labels[dst]).float()).item()) if m > 0 else None

    # Binary fraud datasets use 0/1; treat label==1 as fraud.
    fraud_nodes = torch.nonzero(labels == 1, as_tuple=True)[0]
    normal_nodes = torch.nonzero(labels == 0, as_tuple=True)[0]
    if fraud_nodes.numel() == 0 or normal_nodes.numel() == 0:
        raise RuntimeError("Need at least one fraud and one normal node to rewire towards opposite labels.")

    edge_idx_np = rng.choice(m, size=n_rewire, replace=False)
    edge_idx = torch.from_numpy(edge_idx_np).to(torch.int64)
    src_sel = src[edge_idx]
    src_lbl = labels[src_sel]

    mask_src_is_fraud = (src_lbl == 1)
    n_src_fraud = int(mask_src_is_fraud.sum().item())
    n_src_normal = n_rewire - n_src_fraud

    dst_rewire = torch.empty(n_rewire, dtype=torch.int64)
    if n_src_fraud > 0:
        choice = rng.integers(0, int(normal_nodes.numel()), size=n_src_fraud, endpoint=False)
        dst_rewire[mask_src_is_fraud] = normal_nodes[torch.from_numpy(choice).to(torch.int64)]
    if n_src_normal > 0:
        choice = rng.integers(0, int(fraud_nodes.numel()), size=n_src_normal, endpoint=False)
        dst_rewire[~mask_src_is_fraud] = fraud_nodes[torch.from_numpy(choice).to(torch.int64)]

    dst2 = dst.clone()
    dst2[edge_idx] = dst_rewire

    hetero_after = float(torch.mean((labels[src] != labels[dst2]).float()).item()) if m > 0 else None

    g2 = dgl.graph((src, dst2), num_nodes=int(base_g.num_nodes()))
    _copy_ndata(g2, base_g)
    _copy_edata_if_compatible(g2, base_g)

    info = {"n_rewired_edges": n_rewire, "heterophily_ratio_before": hetero_before, "heterophily_ratio_after": hetero_after}
    return g2, info


def _feature_camouflage(
    base_g,
    *,
    p_cam_feat: float,
    gamma: float,
    rng,
    feature_key: str,
    label_key: str,
) -> tuple[Any, dict[str, Any]]:
    if p_cam_feat <= 0.0:
        return base_g, {"n_camouflaged_nodes": 0, "gamma": float(gamma)}

    try:
        import torch
        import torch.nn.functional as F
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required for feature camouflage perturbations.") from e

    labels = base_g.ndata.get(label_key)
    feats = base_g.ndata.get(feature_key)
    if labels is None or feats is None:
        raise RuntimeError(f"Graph is missing ndata['{label_key}'] or ndata['{feature_key}']; cannot camouflage.")

    labels = labels.squeeze().to(torch.int64).cpu()
    feats = feats.cpu()

    fraud_nodes = torch.nonzero(labels == 1, as_tuple=True)[0]
    normal_nodes = torch.nonzero(labels == 0, as_tuple=True)[0]
    if fraud_nodes.numel() == 0 or normal_nodes.numel() == 0:
        return base_g, {"n_camouflaged_nodes": 0, "gamma": float(gamma)}

    n_cam = int(p_cam_feat * int(fraud_nodes.numel()))
    if n_cam <= 0:
        return base_g, {"n_camouflaged_nodes": 0, "gamma": float(gamma)}

    fraud_choice = rng.choice(int(fraud_nodes.numel()), size=n_cam, replace=False)
    normal_choice = rng.integers(0, int(normal_nodes.numel()), size=n_cam, endpoint=False)
    fraud_sel = fraud_nodes[torch.from_numpy(fraud_choice).to(torch.int64)]
    normal_sel = normal_nodes[torch.from_numpy(normal_choice).to(torch.int64)]

    feats2 = feats.clone()
    feats2[fraud_sel] = (1.0 - float(gamma)) * feats2[fraud_sel] + float(gamma) * feats2[normal_sel]

    try:
        g2 = base_g.clone()
    except Exception:
        # Fallback: rebuild graph structure to avoid mutating base_g in-place.
        import dgl

        src, dst = base_g.edges()
        g2 = dgl.graph((src, dst), num_nodes=int(base_g.num_nodes()))
        _copy_ndata(g2, base_g)
        _copy_edata_if_compatible(g2, base_g)

    g2.ndata[feature_key] = feats2

    # Proxy logs for report/debugging.
    delta = feats2[fraud_sel] - feats[fraud_sel]
    mean_l2_delta = float(torch.norm(delta, dim=1).mean().item()) if n_cam > 0 else 0.0

    cos_before = F.cosine_similarity(feats[fraud_sel], feats[normal_sel], dim=1)
    cos_after = F.cosine_similarity(feats2[fraud_sel], feats[normal_sel], dim=1)
    info = {
        "n_camouflaged_nodes": n_cam,
        "gamma": float(gamma),
        "mean_l2_delta": mean_l2_delta,
        "mean_cosine_to_sampled_normal_before": float(cos_before.mean().item()),
        "mean_cosine_to_sampled_normal_after": float(cos_after.mean().item()),
    }
    return g2, info


def _add_random_edges(
    base_g,
    *,
    edge_noise_rate: float,
    undirected: bool,
    rng,
) -> tuple[Any, dict[str, Any]]:
    if edge_noise_rate <= 0.0:
        return base_g, {"n_added_edges": 0, "undirected": bool(undirected)}

    try:
        import torch
        import dgl
    except Exception as e:  # pragma: no cover
        raise RuntimeError("DGL and PyTorch are required for random-edge perturbations.") from e

    src, dst = base_g.edges()
    src = src.to(torch.int64).cpu()
    dst = dst.to(torch.int64).cpu()

    m = int(src.shape[0])
    n_add = int(edge_noise_rate * m)
    if n_add <= 0:
        return base_g, {"n_added_edges": 0, "undirected": bool(undirected)}

    n_nodes = int(base_g.num_nodes())
    src_add = torch.from_numpy(rng.integers(0, n_nodes, size=n_add, endpoint=False)).to(torch.int64)
    dst_add = torch.from_numpy(rng.integers(0, n_nodes, size=n_add, endpoint=False)).to(torch.int64)

    if undirected:
        src_all = torch.cat([src, src_add, dst_add], dim=0)
        dst_all = torch.cat([dst, dst_add, src_add], dim=0)
        n_added = 2 * n_add
    else:
        src_all = torch.cat([src, src_add], dim=0)
        dst_all = torch.cat([dst, dst_add], dim=0)
        n_added = n_add

    g2 = dgl.graph((src_all, dst_all), num_nodes=n_nodes)
    _copy_ndata(g2, base_g)

    info = {"n_added_edges": int(n_added), "undirected": bool(undirected)}
    return g2, info


def apply_scenario(base_graph, spec: ScenarioSpec):
    """Apply a stress-test scenario to a graph.

    Returns: (graph, scenario_applied, info_dict)
    - scenario_applied is False for no-op variants (e.g., severity==0)
    - info_dict contains scenario-specific proxy logs (counts, deltas, etc.)
    """
    severity = float(spec.severity)
    if severity < 0:
        raise ValueError("severity must be >= 0")

    feature_key = str(spec.params.get("feature_key", "feature"))
    label_key = str(spec.params.get("label_key", "label"))

    # Severity 0 is a no-op variant.
    if severity == 0.0:
        return base_graph, False, {"reason": "severity==0"}

    rng = _rng_for_spec(spec)

    if spec.method == "rewire_edge_dst_to_opposite_label":
        g2, info = _rewire_edge_dst_to_opposite_label(base_graph, p_rewire=severity, rng=rng, label_key=label_key)
        applied = int(info.get("n_rewired_edges", 0)) > 0
    elif spec.method == "replace_fraud_features_with_normal_features":
        gamma = float(spec.params.get("gamma", 1.0))
        g2, info = _feature_camouflage(
            base_graph,
            p_cam_feat=severity,
            gamma=gamma,
            rng=rng,
            feature_key=feature_key,
            label_key=label_key,
        )
        applied = int(info.get("n_camouflaged_nodes", 0)) > 0
    elif spec.method == "add_random_edges":
        undirected = bool(spec.params.get("undirected", False))
        g2, info = _add_random_edges(base_graph, edge_noise_rate=severity, undirected=undirected, rng=rng)
        applied = int(info.get("n_added_edges", 0)) > 0
    else:
        raise ValueError(f"Unknown scenario method: {spec.method}")

    if applied:
        _assert_invariants(base_graph, g2, feature_key=feature_key, label_key=label_key)

    return g2, applied, info
