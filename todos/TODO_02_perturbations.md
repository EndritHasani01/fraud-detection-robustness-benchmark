# TODO 02: Implement the Stress-Test Perturbations (Implemented)

Goal: implement controlled graph/feature transformations for heterophily, camouflage, and noise/density, parameterized by severity.

Implemented artifacts:
- Scenario application: `benchmark/scenarios.py`
  - Heterophily stress (oracle): `rewire_edge_dst_to_opposite_label`
  - Feature camouflage (oracle): `replace_fraud_features_with_normal_features` (supports blending via `gamma`)
  - Noise/density: `add_random_edges` (supports `undirected`)
- Graph stats logging (before/after via cached graphs): `benchmark/stats.py`
- Variant metadata includes scenario proxy logs (counts/deltas/ratios) in `meta.json` for each cached variant graph.

Work:
- Implement heterophily stress:
  - Choose one mechanism:
    - Edge rewiring to increase opposite-label neighbors, or
    - Inject new opposite-label edges.
  - Make it deterministic given a RNG seed.
  - Log the heterophily ratio before/after.
- Implement camouflage stress:
  - Feature camouflage (recommended minimal):
    - For a fraction of fraud nodes, replace/blend features with normal node features.
  - Optional: relation camouflage:
    - Add edges from fraud to normal nodes.
  - Log: fraud feature similarity shift and/or fraud-to-normal neighbor ratio shift.
- Implement noise/density stress:
  - Add random edges (global or per-node).
  - Log: edge count increase and degree distribution shift.
- Validate invariants for every perturbation:
  - Masks (`train_mask`, `val_mask`, `test_mask`) remain unchanged.
  - Feature tensor shape stays the same.
  - Node IDs remain stable so cached results align.

Done when:
- The runner can generate perturbed graphs for every scenario/severity and log graph stats.
- Re-running with the same seed reproduces identical perturbed graphs (or identical summary stats if you store graphs).

How to run:

```bash
py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage graphs --force
```

Notes:
- If a perturbation uses true labels from the full graph (including test labels), explicitly tag it as "oracle" in the results and report.
