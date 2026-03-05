# Decouple Graph Seeds from Training Seeds

## Context

In v1, the single list `seeds.training_seeds` serves double duty: it controls both the randomness of graph perturbations (which variant of a scenario is generated) and the randomness of model training (weight init, dropout, data shuffling). This means that adding one more seed increases both the number of cached graph variants and the number of training runs per variant. Total trainings scale as S^2, which is the primary reason the full experiment matrix takes days instead of hours.

v2 must split this into two independent seed lists so that the experiment can use, say, 1-2 graph seeds (enough to check perturbation variance) and 3-5 training seeds (enough to estimate model variance), bringing total trainings from S^2 to S_graph * S_train.

## Config Schema Change

In the experiment JSON config, the `seeds` object currently has one field:

```json
"seeds": {
  "training_seeds": [0, 1, 2, 3, 4]
}
```

v2 adds a new optional field `graph_seeds`:

```json
"seeds": {
  "graph_seeds": [0, 1],
  "training_seeds": [0, 1, 2]
}
```

When `graph_seeds` is present, it controls which seed values are passed to `apply_scenario` during graph variant generation. When it is absent (backward compat with v1 configs), the runner falls back to using `training_seeds` as it does today.

## Config Validation

In `benchmark/config.py`, the `validate_config` function must be updated. It should accept configs with either `graph_seeds` present or absent. If `graph_seeds` is present, it must be a non-empty list of integers. The function should also emit a warning (print to stderr) if neither `graph_seeds` nor `training_seeds` is present, which would be an invalid config.

## Runner Changes (Graph Stage)

In `benchmark/run.py`, the `_graphs_only` function currently reads graph seeds from `cfg["seeds"]["training_seeds"]` (line 66). This must change to prefer `cfg["seeds"]["graph_seeds"]` when available, falling back to `cfg["seeds"]["training_seeds"]` otherwise. A small helper like `get_graph_seeds(cfg)` and `get_training_seeds(cfg)` in `benchmark/config.py` would keep this logic in one place and let all stages use it consistently.

## Training Stage Changes

Each training stage (`baselines_stage.py`, `pmp_stage.py`, `secgfd_stage.py`) reads training seeds from `cfg["seeds"]["training_seeds"]`. This remains unchanged since training seeds are still drawn from that list. However, the graph variants CSV already records the `graph_seed` used for each variant, so the training stages iterate over variant rows (which already embed the correct graph seed). No changes are needed to training stage seed handling beyond using the shared helper.

## Impact on the Experiment Matrix

With decoupled seeds, the experiment matrix looks like:

- Graph variants: (scenarios * severities * graph_seeds) + 1 clean
- Training runs per variant: training_seeds * models
- Total: variants * training_seeds * models

For the recommended v2 defaults (2 graph seeds, 3 training seeds, 2 severity points, 3 scenarios, 4 models), total trainings drop from ~920 to roughly (3*2*2 + 1) * 3 * 4 = 156, which is about a 6x reduction.

## Backward Compatibility

The v1 config file (`configs/exp_yelpchi_v1.json`) must not be modified. It does not have `graph_seeds`, so the runner falls back to the old behavior. The new v2 configs will include the field. This ensures existing results remain reproducible.
