from __future__ import annotations

from dataclasses import dataclass
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


def apply_scenario(base_graph, spec: ScenarioSpec):
    """Apply a scenario to a graph.

    TODO-02 will replace this no-op implementation with real perturbations.
    For TODO-01 we keep the caching and orchestration working end-to-end.
    """
    # DGLGraph.clone() exists in modern DGL.
    try:
        g2 = base_graph.clone()
    except Exception:
        # Fallback: return base graph (not ideal, but keeps the runner usable).
        g2 = base_graph
    return g2, False  # (graph, scenario_applied)

