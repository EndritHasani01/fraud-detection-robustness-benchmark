from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .relation_utils import (
    EdgeSamplingPolicy,
    copy_compatible_edge_data,
    copy_node_data,
    edge_pair_set,
    edge_sampling_policy_from_params,
    edges_by_relation,
    flatten_relation_edges,
    graph_num_nodes,
    iter_relation_names,
    rebuild_graph_with_edges,
    selected_relation_keys,
    total_edge_count,
    validate_edge_pair,
)


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


def _assert_invariants(base_g, new_g, *, feature_key: str, label_key: str) -> None:
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required to validate perturbation invariants.") from e

    if graph_num_nodes(base_g) != graph_num_nodes(new_g):
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


def _labels_np(base_g, *, label_key: str):
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required for perturbations that read labels.") from e

    labels = base_g.ndata.get(label_key)
    if labels is None:
        raise RuntimeError(f"Graph is missing ndata['{label_key}']; cannot apply this scenario.")
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    return labels.squeeze().to(torch.int64).cpu().numpy()


def _features_tensor(base_g, *, feature_key: str):
    feats = base_g.ndata.get(feature_key)
    if feats is None:
        raise RuntimeError(f"Graph is missing ndata['{feature_key}']; cannot apply this scenario.")
    return feats


def _edge_arrays_cpu(base_g, *, relation_filter: Any = None) -> dict[Any, tuple[Any, Any]]:
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required for graph perturbations.") from e

    out: dict[Any, tuple[Any, Any]] = {}
    for relation_key, (src, dst) in edges_by_relation(base_g, relation_filter=relation_filter).items():
        if not isinstance(src, torch.Tensor):
            src = torch.tensor(src)
        if not isinstance(dst, torch.Tensor):
            dst = torch.tensor(dst)
        out[relation_key] = (src.to(torch.int64).cpu(), dst.to(torch.int64).cpu())
    return out


def _clone_graph_with_same_structure(base_g):
    relation_edges = _edge_arrays_cpu(base_g)
    g2 = rebuild_graph_with_edges(base_g, relation_edges)
    copy_node_data(g2, base_g)
    copy_compatible_edge_data(g2, base_g)
    return g2


def _node_pools(binary_labels_np):
    import numpy as np

    pos_nodes = np.nonzero(binary_labels_np == 1)[0].astype("int64", copy=False)
    neg_nodes = np.nonzero(binary_labels_np == 0)[0].astype("int64", copy=False)
    return pos_nodes, neg_nodes


def _heterophily_ratio(relation_edges: dict[Any, tuple[Any, Any]], labels_np) -> float | None:
    import numpy as np

    if labels_np is None:
        return None
    total = 0
    disagree = 0
    for src, dst in relation_edges.values():
        src_np = src.numpy()
        dst_np = dst.numpy()
        total += int(src_np.shape[0])
        disagree += int(np.count_nonzero(labels_np[src_np] != labels_np[dst_np]))
    if total <= 0:
        return None
    return float(disagree / total)


def _duplicate_mask(values):
    import numpy as np

    if values.shape[0] <= 1:
        return np.zeros(values.shape[0], dtype=bool)
    _, first_idx = np.unique(values, return_index=True)
    keep = np.zeros(values.shape[0], dtype=bool)
    keep[first_idx] = True
    return ~keep


def _attempt_round_limit(pool_size: int, policy: EdgeSamplingPolicy) -> int:
    return max(1, min(max(1, int(pool_size)), 256) * max(1, int(policy.max_attempt_multiplier)))


def _count_names(items: list[Any]) -> dict[str, int]:
    return dict(Counter(iter_relation_names(items)))


def _sampling_rejection_counters(prefix: str) -> dict[str, int]:
    return {
        f"{prefix}_candidate_attempts": 0,
        f"{prefix}_candidate_rejections_no_change": 0,
        f"{prefix}_candidate_rejections_self_loop": 0,
        f"{prefix}_candidate_rejections_existing": 0,
        f"{prefix}_candidate_rejections_duplicate": 0,
        f"{prefix}_candidate_rejections_exhausted": 0,
    }


def _two_means_binary_labels(feats, rng):
    import numpy as np

    feats_np = feats.detach().cpu().float().numpy()
    n = int(feats_np.shape[0])
    if n <= 1:
        return np.zeros(n, dtype="int64")

    seed_idx = rng.choice(n, size=2, replace=False)
    c0 = feats_np[int(seed_idx[0])].copy()
    c1 = feats_np[int(seed_idx[1])].copy()
    assign = np.zeros(n, dtype="int64")

    for _ in range(8):
        d0 = np.sum((feats_np - c0) ** 2, axis=1)
        d1 = np.sum((feats_np - c1) ** 2, axis=1)
        new_assign = (d1 < d0).astype("int64", copy=False)
        if np.array_equal(assign, new_assign):
            break
        assign = new_assign
        if np.count_nonzero(assign == 0) == 0 or np.count_nonzero(assign == 1) == 0:
            break
        c0 = feats_np[assign == 0].mean(axis=0)
        c1 = feats_np[assign == 1].mean(axis=0)

    if np.count_nonzero(assign == 0) == 0 or np.count_nonzero(assign == 1) == 0:
        norms = np.linalg.norm(feats_np, axis=1)
        assign = (norms > float(np.median(norms))).astype("int64", copy=False)
    return assign


def _rewire_relation_edges(
    *,
    src,
    dst,
    selected_edge_idx,
    source_partition_np,
    target_pos_nodes,
    target_neg_nodes,
    rng,
    policy: EdgeSamplingPolicy,
    n_nodes: int,
):
    import numpy as np

    selected_edge_idx = np.asarray(selected_edge_idx, dtype="int64")
    requested = int(selected_edge_idx.shape[0])
    counters = _sampling_rejection_counters("n_rewire")
    if requested <= 0:
        return dst.clone(), {"selected": 0, "actual": 0, **counters}

    src_np = src.numpy()
    dst_np = dst.numpy()
    new_dst_np = dst_np.copy()
    selected_src = src_np[selected_edge_idx]
    selected_old_dst = dst_np[selected_edge_idx]

    existing_hashes = src_np.astype("int64") * n_nodes + dst_np.astype("int64")
    accepted_hashes = np.empty(0, dtype="int64")
    unresolved = np.arange(requested, dtype="int64")
    attempts_left = _attempt_round_limit(max(int(target_pos_nodes.shape[0]), int(target_neg_nodes.shape[0])), policy)

    while unresolved.shape[0] > 0 and attempts_left > 0:
        attempts_left -= 1
        counters["n_rewire_candidate_attempts"] += int(unresolved.shape[0])
        unresolved_src = selected_src[unresolved]
        unresolved_old_dst = selected_old_dst[unresolved]
        unresolved_lbl = source_partition_np[unresolved_src]

        candidate = np.empty(unresolved.shape[0], dtype="int64")
        mask_src_is_pos = unresolved_lbl == 1
        if np.any(mask_src_is_pos):
            choice = rng.integers(0, int(target_neg_nodes.shape[0]), size=int(mask_src_is_pos.sum()), endpoint=False)
            candidate[mask_src_is_pos] = target_neg_nodes[choice]
        if np.any(~mask_src_is_pos):
            choice = rng.integers(0, int(target_pos_nodes.shape[0]), size=int((~mask_src_is_pos).sum()), endpoint=False)
            candidate[~mask_src_is_pos] = target_pos_nodes[choice]

        cand_hash = unresolved_src.astype("int64") * n_nodes + candidate.astype("int64")
        reject_code = np.zeros(unresolved.shape[0], dtype="int8")
        mask = candidate == unresolved_old_dst
        reject_code[mask] = 1
        if not policy.allow_self_loops:
            mask = (candidate == unresolved_src) & (reject_code == 0)
            reject_code[mask] = 2
        if policy.reject_existing:
            mask = np.isin(cand_hash, existing_hashes, assume_unique=False) & (reject_code == 0)
            reject_code[mask] = 3
        if policy.reject_duplicates and accepted_hashes.shape[0] > 0:
            mask = np.isin(cand_hash, accepted_hashes, assume_unique=False) & (reject_code == 0)
            reject_code[mask] = 4

        valid_idx = np.nonzero(reject_code == 0)[0]
        if valid_idx.shape[0] > 0:
            dup_mask = _duplicate_mask(cand_hash[valid_idx])
            if dup_mask.any():
                reject_code[valid_idx[dup_mask]] = 4
                valid_idx = np.nonzero(reject_code == 0)[0]

        counters["n_rewire_candidate_rejections_no_change"] += int(np.count_nonzero(reject_code == 1))
        counters["n_rewire_candidate_rejections_self_loop"] += int(np.count_nonzero(reject_code == 2))
        counters["n_rewire_candidate_rejections_existing"] += int(np.count_nonzero(reject_code == 3))
        counters["n_rewire_candidate_rejections_duplicate"] += int(np.count_nonzero(reject_code == 4))

        if valid_idx.shape[0] > 0:
            new_dst_np[selected_edge_idx[unresolved[valid_idx]]] = candidate[valid_idx]
            accepted_hashes = np.unique(
                np.concatenate([accepted_hashes, cand_hash[valid_idx]], axis=0) if accepted_hashes.size else cand_hash[valid_idx]
            )

        unresolved = unresolved[reject_code != 0]

    counters["n_rewire_candidate_rejections_exhausted"] += int(unresolved.shape[0])

    actual = int(np.count_nonzero(new_dst_np[selected_edge_idx] != selected_old_dst))
    return dst.new_tensor(new_dst_np), {"selected": requested, "actual": actual, **counters}


def _apply_partition_based_rewire(
    base_g,
    *,
    p_rewire: float,
    source_partition_np,
    audit_labels_np,
    rng,
    relation_filter: Any,
    policy: EdgeSamplingPolicy,
):
    all_relation_edges = _edge_arrays_cpu(base_g)
    relation_keys = selected_relation_keys(base_g, relation_filter=relation_filter)
    selected_relation_edges = {key: all_relation_edges[key] for key in relation_keys}
    m_total = total_edge_count(selected_relation_edges)
    info = {
        "n_rewired_edges": 0,
        "n_rewired_edges_selected": 0,
        "n_rewired_edges_actual": 0,
        "relation_filter_applied": iter_relation_names(relation_keys),
        "sampling_policy": policy.as_dict(),
    }
    if p_rewire <= 0.0 or m_total <= 0:
        return base_g, info

    pos_nodes, neg_nodes = _node_pools(source_partition_np)
    if pos_nodes.shape[0] == 0 or neg_nodes.shape[0] == 0:
        return base_g, info

    n_selected = int(p_rewire * m_total)
    if n_selected <= 0:
        return base_g, info

    flat_edges = flatten_relation_edges(selected_relation_edges)
    sampled_positions = rng.choice(len(flat_edges), size=n_selected, replace=False)
    positions_by_relation: dict[Any, list[int]] = {}
    for sampled_pos in sampled_positions.tolist():
        relation_key, edge_idx = flat_edges[int(sampled_pos)]
        positions_by_relation.setdefault(relation_key, []).append(int(edge_idx))

    n_nodes = graph_num_nodes(base_g)
    new_relation_edges = dict(all_relation_edges)
    selected_relation_names: list[Any] = []
    actual_relation_names: list[Any] = []
    total_actual = 0
    counters = _sampling_rejection_counters("n_rewire")

    for relation_key, edge_idx_list in positions_by_relation.items():
        selected_relation_names.extend([relation_key] * len(edge_idx_list))
        src, dst = all_relation_edges[relation_key]
        new_dst, relation_info = _rewire_relation_edges(
            src=src,
            dst=dst,
            selected_edge_idx=edge_idx_list,
            source_partition_np=source_partition_np,
            target_pos_nodes=pos_nodes,
            target_neg_nodes=neg_nodes,
            rng=rng,
            policy=policy,
            n_nodes=n_nodes,
        )
        actual_count = int(relation_info["actual"])
        total_actual += actual_count
        actual_relation_names.extend([relation_key] * actual_count)
        new_relation_edges[relation_key] = (src, new_dst)
        for key in counters:
            counters[key] += int(relation_info.get(key, 0))

    g2 = rebuild_graph_with_edges(base_g, new_relation_edges)
    copy_node_data(g2, base_g)
    copy_compatible_edge_data(g2, base_g)
    info.update(
        {
            "n_rewired_edges": int(total_actual),
            "n_rewired_edges_selected": int(n_selected),
            "n_rewired_edges_actual": int(total_actual),
            "heterophily_ratio_before": _heterophily_ratio(selected_relation_edges, audit_labels_np),
            "heterophily_ratio_after": _heterophily_ratio({key: new_relation_edges[key] for key in relation_keys}, audit_labels_np),
            "n_rewired_edges_selected_by_relation": _count_names(selected_relation_names),
            "n_rewired_edges_actual_by_relation": _count_names(actual_relation_names),
            **counters,
        }
    )
    return g2, info


def _rewire_edge_dst_to_opposite_label(base_g, *, p_rewire: float, rng, label_key: str, relation_filter: Any, policy: EdgeSamplingPolicy):
    labels_np = _labels_np(base_g, label_key=label_key)
    return _apply_partition_based_rewire(
        base_g,
        p_rewire=p_rewire,
        source_partition_np=labels_np,
        audit_labels_np=labels_np,
        rng=rng,
        relation_filter=relation_filter,
        policy=policy,
    )


def _rewire_edge_dst_to_feature_pseudo_opposite_label(
    base_g,
    *,
    p_rewire: float,
    rng,
    feature_key: str,
    label_key: str,
    relation_filter: Any,
    policy: EdgeSamplingPolicy,
):
    feats = _features_tensor(base_g, feature_key=feature_key)
    pseudo_labels_np = _two_means_binary_labels(feats, rng)
    labels_np = _labels_np(base_g, label_key=label_key)
    g2, info = _apply_partition_based_rewire(
        base_g,
        p_rewire=p_rewire,
        source_partition_np=pseudo_labels_np,
        audit_labels_np=labels_np,
        rng=rng,
        relation_filter=relation_filter,
        policy=policy,
    )
    info["selection_proxy"] = "feature_two_means"
    info["pseudo_label_pos_rate"] = float(pseudo_labels_np.mean()) if pseudo_labels_np.size else None
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
        return base_g, {
            "n_camouflaged_nodes": 0,
            "n_camouflaged_nodes_requested": 0,
            "n_camouflaged_nodes_actual": 0,
            "gamma": float(gamma),
        }

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
        return base_g, {
            "n_camouflaged_nodes": 0,
            "n_camouflaged_nodes_requested": 0,
            "n_camouflaged_nodes_actual": 0,
            "gamma": float(gamma),
        }

    n_cam = int(p_cam_feat * int(fraud_nodes.numel()))
    if n_cam <= 0:
        return base_g, {
            "n_camouflaged_nodes": 0,
            "n_camouflaged_nodes_requested": 0,
            "n_camouflaged_nodes_actual": 0,
            "gamma": float(gamma),
        }

    fraud_choice = rng.choice(int(fraud_nodes.numel()), size=n_cam, replace=False)
    normal_choice = rng.integers(0, int(normal_nodes.numel()), size=n_cam, endpoint=False)
    fraud_sel = fraud_nodes[torch.from_numpy(fraud_choice).to(torch.int64)]
    normal_sel = normal_nodes[torch.from_numpy(normal_choice).to(torch.int64)]

    feats2 = feats.clone()
    feats2[fraud_sel] = (1.0 - float(gamma)) * feats2[fraud_sel] + float(gamma) * feats2[normal_sel]

    try:
        g2 = base_g.clone()
    except Exception:
        g2 = _clone_graph_with_same_structure(base_g)

    g2.ndata[feature_key] = feats2

    delta = feats2[fraud_sel] - feats[fraud_sel]
    mean_l2_delta = float(torch.norm(delta, dim=1).mean().item()) if n_cam > 0 else 0.0
    cos_before = F.cosine_similarity(feats[fraud_sel], feats[normal_sel], dim=1)
    cos_after = F.cosine_similarity(feats2[fraud_sel], feats[normal_sel], dim=1)
    info = {
        "n_camouflaged_nodes": int(n_cam),
        "n_camouflaged_nodes_requested": int(n_cam),
        "n_camouflaged_nodes_actual": int(n_cam),
        "gamma": float(gamma),
        "mean_l2_delta": mean_l2_delta,
        "mean_cosine_to_sampled_normal_before": float(cos_before.mean().item()),
        "mean_cosine_to_sampled_normal_after": float(cos_after.mean().item()),
    }
    return g2, info


def _neighbor_mix_for_nodes(relation_edges: dict[Any, tuple[Any, Any]], labels_np, selected_nodes_np) -> dict[str, float | None]:
    import numpy as np

    if selected_nodes_np.shape[0] == 0:
        return {
            "fraud_to_normal_neighbor_ratio": None,
            "fraud_to_fraud_neighbor_ratio": None,
            "mean_selected_out_degree": None,
        }

    selected_mask = np.zeros(labels_np.shape[0], dtype=bool)
    selected_mask[selected_nodes_np] = True

    normal_neighbors = 0
    fraud_neighbors = 0
    edge_count = 0
    for src, dst in relation_edges.values():
        src_np = src.numpy()
        dst_np = dst.numpy()
        mask = selected_mask[src_np]
        selected_dst = dst_np[mask]
        edge_count += int(selected_dst.shape[0])
        if selected_dst.shape[0] > 0:
            normal_neighbors += int((labels_np[selected_dst] == 0).sum())
            fraud_neighbors += int((labels_np[selected_dst] == 1).sum())

    total = normal_neighbors + fraud_neighbors
    if total <= 0:
        return {
            "fraud_to_normal_neighbor_ratio": 0.0,
            "fraud_to_fraud_neighbor_ratio": 0.0,
            "mean_selected_out_degree": 0.0,
        }
    return {
        "fraud_to_normal_neighbor_ratio": float(normal_neighbors / total),
        "fraud_to_fraud_neighbor_ratio": float(fraud_neighbors / total),
        "mean_selected_out_degree": float(edge_count / max(1, int(selected_nodes_np.shape[0]))),
    }


def _relation_camouflage(
    base_g,
    *,
    p_cam_rel: float,
    rng,
    label_key: str,
    relation_filter: Any,
    edges_per_node: int,
    remove_suspicious_ratio: float,
    policy: EdgeSamplingPolicy,
):
    import numpy as np
    import torch

    info = {
        "n_camouflaged_nodes": 0,
        "n_camouflaged_edges_requested": 0,
        "n_camouflaged_edges_added": 0,
        "n_suspicious_edges_removed_requested": 0,
        "n_suspicious_edges_removed": 0,
        "relation_filter_applied": [],
        "sampling_policy": policy.as_dict(),
        **_sampling_rejection_counters("n_camouflage_edge"),
    }
    if p_cam_rel <= 0.0:
        return base_g, info

    labels_np = _labels_np(base_g, label_key=label_key)
    fraud_nodes, normal_nodes = _node_pools(labels_np)
    if fraud_nodes.shape[0] == 0 or normal_nodes.shape[0] == 0:
        return base_g, info

    n_selected = int(p_cam_rel * int(fraud_nodes.shape[0]))
    if n_selected <= 0:
        return base_g, info

    all_relation_edges = _edge_arrays_cpu(base_g)
    relation_keys = selected_relation_keys(base_g, relation_filter=relation_filter)
    selected_nodes = fraud_nodes[rng.choice(fraud_nodes.shape[0], size=n_selected, replace=False)]

    before_stats = _neighbor_mix_for_nodes({key: all_relation_edges[key] for key in relation_keys}, labels_np, selected_nodes)
    additions_by_relation: dict[Any, list[tuple[int, int]]] = {key: [] for key in all_relation_edges}
    existing_by_relation = {key: edge_pair_set(src, dst) for key, (src, dst) in all_relation_edges.items()}
    added_relation_names: list[Any] = []
    requested_edges = int(n_selected * max(1, int(edges_per_node)))

    for node_id in selected_nodes.tolist():
        for _ in range(max(1, int(edges_per_node))):
            relation_key = relation_keys[int(rng.integers(0, len(relation_keys), endpoint=False))]
            pending_edges = set(additions_by_relation[relation_key])
            attempts_left = _attempt_round_limit(int(normal_nodes.shape[0]), policy)
            added = False
            while attempts_left > 0:
                attempts_left -= 1
                info["n_camouflage_edge_candidate_attempts"] += 1
                dst_id = int(normal_nodes[int(rng.integers(0, normal_nodes.shape[0], endpoint=False))])
                reject_reason = validate_edge_pair(
                    src=int(node_id),
                    dst=dst_id,
                    existing_edges=existing_by_relation[relation_key],
                    pending_edges=pending_edges,
                    policy=policy,
                )
                if reject_reason is not None:
                    info[f"n_camouflage_edge_candidate_rejections_{reject_reason}"] += 1
                    continue
                additions_by_relation[relation_key].append((int(node_id), dst_id))
                pending_edges.add((int(node_id), dst_id))
                existing_by_relation[relation_key].add((int(node_id), dst_id))
                added_relation_names.append(relation_key)
                added = True
                break
            if not added:
                info["n_camouflage_edge_candidate_rejections_exhausted"] += 1

    removals_by_relation: dict[Any, set[int]] = {key: set() for key in all_relation_edges}
    added_edges_actual = int(sum(len(v) for v in additions_by_relation.values()))
    requested_removals = int(max(0.0, float(remove_suspicious_ratio)) * added_edges_actual)

    if requested_removals > 0:
        selected_mask = np.zeros(labels_np.shape[0], dtype=bool)
        selected_mask[selected_nodes] = True
        candidates: list[tuple[Any, int]] = []
        for relation_key in relation_keys:
            src, dst = all_relation_edges[relation_key]
            src_np = src.numpy()
            dst_np = dst.numpy()
            suspicious_mask = selected_mask[src_np] & (labels_np[dst_np] == 1)
            for edge_idx in np.nonzero(suspicious_mask)[0].tolist():
                candidates.append((relation_key, int(edge_idx)))
        if candidates:
            chosen = rng.choice(len(candidates), size=min(requested_removals, len(candidates)), replace=False)
            for idx in chosen.tolist():
                relation_key, edge_idx = candidates[int(idx)]
                removals_by_relation[relation_key].add(int(edge_idx))

    new_relation_edges: dict[Any, tuple[Any, Any]] = {}
    for relation_key, (src, dst) in all_relation_edges.items():
        src2 = src
        dst2 = dst
        if removals_by_relation[relation_key]:
            keep_mask = torch.ones(src.shape[0], dtype=torch.bool)
            keep_mask[list(sorted(removals_by_relation[relation_key]))] = False
            src2 = src2[keep_mask]
            dst2 = dst2[keep_mask]
        if additions_by_relation[relation_key]:
            add_src = torch.tensor([src_id for src_id, _ in additions_by_relation[relation_key]], dtype=torch.int64)
            add_dst = torch.tensor([dst_id for _, dst_id in additions_by_relation[relation_key]], dtype=torch.int64)
            src2 = torch.cat([src2, add_src], dim=0)
            dst2 = torch.cat([dst2, add_dst], dim=0)
        new_relation_edges[relation_key] = (src2, dst2)

    g2 = rebuild_graph_with_edges(base_g, new_relation_edges)
    copy_node_data(g2, base_g)
    after_stats = _neighbor_mix_for_nodes({key: new_relation_edges[key] for key in relation_keys}, labels_np, selected_nodes)

    info.update(
        {
            "n_camouflaged_nodes": int(n_selected),
            "n_camouflaged_edges_requested": int(requested_edges),
            "n_camouflaged_edges_added": int(added_edges_actual),
            "n_suspicious_edges_removed_requested": int(requested_removals),
            "n_suspicious_edges_removed": int(sum(len(v) for v in removals_by_relation.values())),
            "fraud_to_normal_neighbor_ratio_before": before_stats["fraud_to_normal_neighbor_ratio"],
            "fraud_to_normal_neighbor_ratio_after": after_stats["fraud_to_normal_neighbor_ratio"],
            "fraud_to_fraud_neighbor_ratio_before": before_stats["fraud_to_fraud_neighbor_ratio"],
            "fraud_to_fraud_neighbor_ratio_after": after_stats["fraud_to_fraud_neighbor_ratio"],
            "mean_selected_out_degree_before": before_stats["mean_selected_out_degree"],
            "mean_selected_out_degree_after": after_stats["mean_selected_out_degree"],
            "relation_filter_applied": iter_relation_names(relation_keys),
            "n_camouflaged_edges_added_by_relation": _count_names(added_relation_names),
        }
    )
    return g2, info


def _add_random_edges(
    base_g,
    *,
    edge_noise_rate: float,
    undirected: bool,
    rng,
    relation_filter: Any,
    policy: EdgeSamplingPolicy,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np
    import torch

    all_relation_edges = _edge_arrays_cpu(base_g)
    relation_keys = selected_relation_keys(base_g, relation_filter=relation_filter)
    selected_relation_edges = {key: all_relation_edges[key] for key in relation_keys}
    m_total = total_edge_count(selected_relation_edges)
    n_requested_pairs = int(edge_noise_rate * m_total)
    info = {
        "n_added_edges": 0,
        "n_added_edges_requested_total": 0,
        "n_added_edges_actual_total": 0,
        "n_added_edge_pairs_requested": 0,
        "n_added_edge_pairs_actual": 0,
        "undirected": bool(undirected),
        "relation_filter_applied": iter_relation_names(relation_keys),
        "sampling_policy": policy.as_dict(),
        **_sampling_rejection_counters("n_noise"),
    }
    if edge_noise_rate <= 0.0 or n_requested_pairs <= 0:
        return base_g, info

    relation_ids = rng.integers(0, len(relation_keys), size=n_requested_pairs, endpoint=False)
    requested_by_relation = Counter(relation_keys[int(idx)] for idx in relation_ids.tolist())
    n_nodes = graph_num_nodes(base_g)
    new_relation_edges = dict(all_relation_edges)
    actual_pair_names: list[Any] = []
    added_pairs_actual = 0

    for relation_key in relation_keys:
        requested = int(requested_by_relation.get(relation_key, 0))
        if requested <= 0:
            continue

        src, dst = all_relation_edges[relation_key]
        src_np = src.numpy()
        dst_np = dst.numpy()
        existing_hashes = src_np.astype("int64") * n_nodes + dst_np.astype("int64")
        accepted_src = np.empty(0, dtype="int64")
        accepted_dst = np.empty(0, dtype="int64")
        accepted_hashes = np.empty(0, dtype="int64")
        accepted_duplicate_keys = np.empty(0, dtype="int64")
        unresolved = np.arange(requested, dtype="int64")
        attempts_left = _attempt_round_limit(n_nodes, policy)

        while unresolved.shape[0] > 0 and attempts_left > 0:
            attempts_left -= 1
            info["n_noise_candidate_attempts"] += int(unresolved.shape[0])
            cand_src = rng.integers(0, n_nodes, size=unresolved.shape[0], endpoint=False, dtype="int64")
            cand_dst = rng.integers(0, n_nodes, size=unresolved.shape[0], endpoint=False, dtype="int64")
            cand_hash = cand_src * n_nodes + cand_dst
            duplicate_key = cand_hash if not undirected else (np.minimum(cand_src, cand_dst) * n_nodes + np.maximum(cand_src, cand_dst))

            reject_code = np.zeros(unresolved.shape[0], dtype="int8")
            if not policy.allow_self_loops:
                mask = cand_src == cand_dst
                reject_code[mask] = 1
            if policy.reject_existing:
                mask = np.isin(cand_hash, existing_hashes, assume_unique=False) & (reject_code == 0)
                reject_code[mask] = 2
                if undirected:
                    mirror_mask = (
                        np.isin(cand_dst * n_nodes + cand_src, existing_hashes, assume_unique=False) & (reject_code == 0)
                    )
                    reject_code[mirror_mask] = 2
            if policy.reject_duplicates and accepted_hashes.shape[0] > 0:
                mask = np.isin(cand_hash, accepted_hashes, assume_unique=False) & (reject_code == 0)
                reject_code[mask] = 3
                if undirected:
                    mirror_mask = (
                        np.isin(cand_dst * n_nodes + cand_src, accepted_hashes, assume_unique=False) & (reject_code == 0)
                    )
                    reject_code[mirror_mask] = 3
                duplicate_mask = np.isin(duplicate_key, accepted_duplicate_keys, assume_unique=False) & (reject_code == 0)
                reject_code[duplicate_mask] = 3

            valid_idx = np.nonzero(reject_code == 0)[0]
            if valid_idx.shape[0] > 0:
                dup_mask = _duplicate_mask(duplicate_key[valid_idx])
                if dup_mask.any():
                    reject_code[valid_idx[dup_mask]] = 3
                    valid_idx = np.nonzero(reject_code == 0)[0]

            info["n_noise_candidate_rejections_self_loop"] += int(np.count_nonzero(reject_code == 1))
            info["n_noise_candidate_rejections_existing"] += int(np.count_nonzero(reject_code == 2))
            info["n_noise_candidate_rejections_duplicate"] += int(np.count_nonzero(reject_code == 3))

            if valid_idx.shape[0] > 0:
                accepted_src = np.concatenate([accepted_src, cand_src[valid_idx]], axis=0)
                accepted_dst = np.concatenate([accepted_dst, cand_dst[valid_idx]], axis=0)
                round_hashes = cand_hash[valid_idx]
                if undirected:
                    round_hashes = np.concatenate([round_hashes, cand_dst[valid_idx] * n_nodes + cand_src[valid_idx]], axis=0)
                accepted_hashes = np.unique(
                    np.concatenate([accepted_hashes, round_hashes], axis=0) if accepted_hashes.size else round_hashes
                )
                round_duplicate_keys = duplicate_key[valid_idx]
                accepted_duplicate_keys = np.unique(
                    np.concatenate([accepted_duplicate_keys, round_duplicate_keys], axis=0)
                    if accepted_duplicate_keys.size
                    else round_duplicate_keys
                )

            unresolved = unresolved[reject_code != 0]

        info["n_noise_candidate_rejections_exhausted"] += int(unresolved.shape[0])

        if accepted_src.shape[0] > 0:
            add_src = torch.from_numpy(accepted_src).to(torch.int64)
            add_dst = torch.from_numpy(accepted_dst).to(torch.int64)
            if undirected:
                mirror_mask = add_src != add_dst
                src2 = torch.cat([src, add_src, add_dst[mirror_mask]], dim=0)
                dst2 = torch.cat([dst, add_dst, add_src[mirror_mask]], dim=0)
            else:
                src2 = torch.cat([src, add_src], dim=0)
                dst2 = torch.cat([dst, add_dst], dim=0)
            new_relation_edges[relation_key] = (src2, dst2)
            added_pairs_actual += int(accepted_src.shape[0])
            actual_pair_names.extend([relation_key] * int(accepted_src.shape[0]))

    g2 = rebuild_graph_with_edges(base_g, new_relation_edges)
    copy_node_data(g2, base_g)

    added_edges_actual_total = 0
    for relation_key in relation_keys:
        before_count = int(all_relation_edges[relation_key][0].shape[0])
        after_count = int(new_relation_edges[relation_key][0].shape[0])
        added_edges_actual_total += max(0, after_count - before_count)

    info.update(
        {
            "n_added_edges": int(added_edges_actual_total),
            "n_added_edges_requested_total": int(n_requested_pairs * (2 if undirected else 1)),
            "n_added_edges_actual_total": int(added_edges_actual_total),
            "n_added_edge_pairs_requested": int(n_requested_pairs),
            "n_added_edge_pairs_actual": int(added_pairs_actual),
            "n_added_edge_pairs_actual_by_relation": _count_names(actual_pair_names),
        }
    )
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
    relation_filter = spec.params.get("relation_filter")
    policy = edge_sampling_policy_from_params(spec.params)

    if severity == 0.0:
        return base_graph, False, {"reason": "severity==0"}

    rng = _rng_for_spec(spec)

    if spec.method == "rewire_edge_dst_to_opposite_label":
        g2, info = _rewire_edge_dst_to_opposite_label(
            base_graph,
            p_rewire=severity,
            rng=rng,
            label_key=label_key,
            relation_filter=relation_filter,
            policy=policy,
        )
        applied = int(info.get("n_rewired_edges_actual", info.get("n_rewired_edges", 0))) > 0
    elif spec.method == "rewire_edge_dst_to_feature_pseudo_opposite_label":
        g2, info = _rewire_edge_dst_to_feature_pseudo_opposite_label(
            base_graph,
            p_rewire=severity,
            rng=rng,
            feature_key=feature_key,
            label_key=label_key,
            relation_filter=relation_filter,
            policy=policy,
        )
        applied = int(info.get("n_rewired_edges_actual", info.get("n_rewired_edges", 0))) > 0
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
        applied = int(info.get("n_camouflaged_nodes_actual", info.get("n_camouflaged_nodes", 0))) > 0
    elif spec.method == "add_relation_camouflage_edges":
        g2, info = _relation_camouflage(
            base_graph,
            p_cam_rel=severity,
            rng=rng,
            label_key=label_key,
            relation_filter=relation_filter,
            edges_per_node=int(spec.params.get("camouflage_edges_per_node", 1) or 1),
            remove_suspicious_ratio=float(spec.params.get("remove_suspicious_ratio", 0.0) or 0.0),
            policy=policy,
        )
        applied = (int(info.get("n_camouflaged_edges_added", 0)) + int(info.get("n_suspicious_edges_removed", 0))) > 0
    elif spec.method == "add_random_edges":
        undirected = bool(spec.params.get("undirected", False))
        g2, info = _add_random_edges(
            base_graph,
            edge_noise_rate=severity,
            undirected=undirected,
            rng=rng,
            relation_filter=relation_filter,
            policy=policy,
        )
        applied = int(info.get("n_added_edges_actual_total", info.get("n_added_edges", 0))) > 0
    else:
        raise ValueError(f"Unknown scenario method: {spec.method}")

    if applied:
        _assert_invariants(base_graph, g2, feature_key=feature_key, label_key=label_key)

    return g2, applied, info
