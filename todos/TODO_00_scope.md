# TODO 00: Finalize Scope and Experimental Design (Implemented)

Goal: lock a minimal, finishable experiment plan that matches the course requirements and stays consistent across all methods.

Implemented artifacts:
- Frozen experiment config: `configs/exp_yelpchi_v1.json`
- Config index: `configs/README.md`

Final decisions (v1):
- Dataset: YelpChi (via DGL FraudDataset `name="yelp"`).
- Split: single fixed split, train/val/test = 0.4/0.2/0.4 (split seed 717).
- Models: MLP baseline, GraphSAGE baseline, PMP, SEC-GFD.
- Canonical graph view: homogeneous DGL graph (relations merged) for cross-model comparability.
- Scenarios:
  - Heterophily rewiring (oracle labels), severities {0.0, 0.1, 0.2, 0.3}.
  - Feature camouflage (oracle labels), severities {0.0, 0.1, 0.2, 0.3}, gamma=1.0 (replace).
  - Random edge noise, severities {0.0, 0.05, 0.1, 0.2}.
- Seeds: 5 training seeds {0,1,2,3,4}.
- Metrics: ROC-AUC, AP, F1-macro; threshold selected on validation only.

Work:
- Choose datasets:
  - Default: YelpChi only.
  - Optional: Amazon only if time/compute allows.
- Choose the evaluated models (required minimum):
  - MLP baseline (features only).
  - 1 baseline GNN (GraphSAGE or GCN).
  - 2 specialized methods from `Repos/` (recommended: PMP + SEC-GFD).
- Choose the graph representation for fairness across models:
  - Recommended: use a homogeneous DGL graph for all models (merge relations) so PMP/SEC-GFD/baselines operate on the same structure.
  - If you keep multi-relation graphs for some models, explicitly document why results are still comparable.
- Define the stress-test matrix:
  - Scenarios: heterophily, camouflage, noise/density.
  - Severity grid per scenario (example): {0.0, 0.1, 0.2, 0.3} or similar.
  - Seeds: minimum 5 (better 10).
  - Splits: minimum 1 fixed split; optional 3-5 splits if feasible.
- Define metrics and thresholding:
  - Metrics: ROC-AUC, AP, F1-macro.
  - Threshold rule: select threshold on validation only; apply once to test.
- Write the final experiment definition into a single config file you will use everywhere (JSON/YAML).

Done when:
- You have a single written experiment plan (dataset(s), models, scenarios/severities, seeds/splits, metrics) that you can paste into the report Methods section.
- All later TODOs can be implemented without changing these choices.

Notes:
- Any stress-test that uses true labels to rewire edges is an "oracle" perturbation. This is acceptable, but must be disclosed in the report.
