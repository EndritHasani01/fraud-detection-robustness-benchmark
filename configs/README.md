# Experiment Configs

This folder contains frozen experiment definitions for the robustness benchmark.

These configs are the single source of truth for:
- dataset(s) and split ratios
- models to evaluate
- stress-test scenarios and severity grids
- seeds, splits, and evaluation metrics

Start with:
- `configs/exp_yelpchi_v3.json`

For the next confirmatory run, use:
- `configs/exp_yelpchi_v4_multisplit.json`: three preregistered stratified splits, five training seeds, seed-aware reporting metadata, and bounded degree-relative relation camouflage. It intentionally uses one graph seed so the three-split graph cache fits the measured Kaggle disk budget.

For the expanded factorial Kaggle run, use:
- `configs/exp_yelpchi_v4_factorial.json`: the same three split seeds crossed with 40/20/40 and 60/20/20 allocation regimes, graph seeds 0/1, and five training seeds. Its notebook execution must materialize and evict graph caches one split at a time.

Additional frozen experiment definitions:
- `configs/exp_yelpchi_v3_fast.json`: minimal developer config for quick end-to-end checks across the full v3 scenario surface
- `configs/exp_yelpchi_v3_full.json`: larger v3 config with denser severity curves
- `configs/exp_yelpchi_v2.json`: historical frozen v2 benchmark
- `configs/exp_yelpchi_v2_fast.json`: historical v2 smoke-test config
- `configs/exp_yelpchi_v2_full.json`: historical v2 fuller-curve config
- `configs/exp_yelpchi_v1.json`: historical frozen v1 benchmark kept for backward-compatible reruns

The v3 configs introduce relation camouflage, a non-oracle heterophily path, graph-view controls, and audit-export settings. The current scenario semantics, artifact descriptions, and interpretation boundaries are documented in [docs/V3_BENCHMARK_REFERENCE.md](../docs/V3_BENCHMARK_REFERENCE.md).
