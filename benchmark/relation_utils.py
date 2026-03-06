from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


RelationKey = tuple[str, str, str] | None


@dataclass(frozen=True)
class EdgeSamplingPolicy:
    """Shared policy for edge rewiring and edge-addition scenarios."""

    allow_self_loops: bool = False
    reject_existing: bool = True
    reject_duplicates: bool = True
    max_attempt_multiplier: int = 20

    def as_dict(self) -> dict[str, Any]:
        return {
            "allow_self_loops": bool(self.allow_self_loops),
            "reject_existing": bool(self.reject_existing),
            "reject_duplicates": bool(self.reject_duplicates),
            "max_attempt_multiplier": int(self.max_attempt_multiplier),
        }


def edge_sampling_policy_from_params(params: dict[str, Any] | None) -> EdgeSamplingPolicy:
    params = params or {}
    max_attempt_multiplier = int(params.get("max_attempt_multiplier", 20) or 20)
    if max_attempt_multiplier <= 0:
        max_attempt_multiplier = 20
    return EdgeSamplingPolicy(
        allow_self_loops=bool(params.get("allow_self_loops", False)),
        reject_existing=bool(params.get("reject_existing", True)),
        reject_duplicates=bool(params.get("reject_duplicates", True)),
        max_attempt_multiplier=max_attempt_multiplier,
    )


def graph_num_nodes(graph: Any) -> int:
    try:
        return int(graph.num_nodes())
    except TypeError:
        ntypes = list(getattr(graph, "ntypes", []))
        if len(ntypes) == 1:
            return int(graph.num_nodes(ntypes[0]))
        raise


def is_relation_aware_graph(graph: Any) -> bool:
    canonical_etypes = list(getattr(graph, "canonical_etypes", []))
    return len(canonical_etypes) > 1


def relation_name(relation_key: RelationKey) -> str:
    if relation_key is None:
        return "homogeneous"
    src_type, etype, dst_type = relation_key
    return f"{src_type}:{etype}:{dst_type}"


def _normalize_relation_filter_values(relation_filter: Any) -> list[Any]:
    if relation_filter is None:
        return []
    if isinstance(relation_filter, (list, tuple, set)):
        return [value for value in relation_filter]
    return [relation_filter]


def selected_relation_keys(graph: Any, relation_filter: Any = None) -> list[RelationKey]:
    if not is_relation_aware_graph(graph):
        return [None]

    canonical_etypes = list(getattr(graph, "canonical_etypes", []))
    raw_values = _normalize_relation_filter_values(relation_filter)
    if not raw_values:
        return canonical_etypes

    selected: list[RelationKey] = []
    for relation_key in canonical_etypes:
        key_text = relation_name(relation_key)
        short_name = relation_key[1]
        if relation_key in raw_values or key_text in raw_values or short_name in raw_values:
            selected.append(relation_key)
    if not selected:
        raise ValueError(f"relation_filter matched no canonical etypes: {raw_values}")
    return selected


def edges_by_relation(graph: Any, *, relation_filter: Any = None) -> dict[RelationKey, tuple[Any, Any]]:
    selected = selected_relation_keys(graph, relation_filter=relation_filter)
    if selected == [None]:
        src, dst = graph.edges()
        return {None: (src, dst)}

    out: dict[RelationKey, tuple[Any, Any]] = {}
    for relation_key in selected:
        src, dst = graph.edges(etype=relation_key)
        out[relation_key] = (src, dst)
    return out


def edge_pair_set(src, dst) -> set[tuple[int, int]]:
    return set(zip(src.tolist(), dst.tolist()))


def validate_edge_pair(
    *,
    src: int,
    dst: int,
    existing_edges: set[tuple[int, int]],
    pending_edges: set[tuple[int, int]],
    policy: EdgeSamplingPolicy,
) -> str | None:
    if (not policy.allow_self_loops) and int(src) == int(dst):
        return "self_loop"
    if policy.reject_existing and (int(src), int(dst)) in existing_edges:
        return "existing"
    if policy.reject_duplicates and (int(src), int(dst)) in pending_edges:
        return "duplicate"
    return None


def rebuild_graph_with_edges(base_graph: Any, relation_edges: dict[RelationKey, tuple[Any, Any]]):
    try:
        import dgl
    except Exception as e:  # pragma: no cover
        raise RuntimeError("DGL is required to rebuild perturbed graphs.") from e

    if not is_relation_aware_graph(base_graph):
        src, dst = relation_edges[None]
        return dgl.graph((src, dst), num_nodes=graph_num_nodes(base_graph))

    num_nodes_dict: dict[str, int] = {}
    for ntype in list(getattr(base_graph, "ntypes", [])):
        num_nodes_dict[str(ntype)] = int(base_graph.num_nodes(ntype))
    data_dict = {relation_key: edge_pair for relation_key, edge_pair in relation_edges.items() if relation_key is not None}
    return dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)


def copy_node_data(dst_graph: Any, src_graph: Any) -> None:
    for key, value in src_graph.ndata.items():
        dst_graph.ndata[key] = value


def copy_compatible_edge_data(dst_graph: Any, src_graph: Any) -> None:
    if is_relation_aware_graph(src_graph):
        return
    if int(dst_graph.num_edges()) != int(src_graph.num_edges()):
        return
    for key, value in src_graph.edata.items():
        dst_graph.edata[key] = value


def relation_counts(relation_edges: dict[RelationKey, tuple[Any, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for relation_key, (src, _dst) in relation_edges.items():
        out[relation_name(relation_key)] = int(src.shape[0])
    return out


def flatten_relation_edges(relation_edges: dict[RelationKey, tuple[Any, Any]]) -> list[tuple[RelationKey, int]]:
    flat: list[tuple[RelationKey, int]] = []
    for relation_key, (src, _dst) in relation_edges.items():
        flat.extend((relation_key, idx) for idx in range(int(src.shape[0])))
    return flat


def total_edge_count(relation_edges: dict[RelationKey, tuple[Any, Any]]) -> int:
    return sum(int(src.shape[0]) for src, _dst in relation_edges.values())


def iter_relation_names(relation_keys: Iterable[RelationKey]) -> list[str]:
    return [relation_name(key) for key in relation_keys]
