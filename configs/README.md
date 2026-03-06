# Experiment Configs

This folder contains frozen experiment definitions for the robustness benchmark.

These configs are the single source of truth for:
- dataset(s) and split ratios
- models to evaluate
- stress-test scenarios and severity grids
- seeds, splits, and evaluation metrics

Start with:
- `configs/exp_yelpchi_v1.json`

Additional frozen experiment definitions:
- `configs/exp_yelpchi_v2.json`: main v2 benchmark with coarse clean-vs-stressed severity grids
- `configs/exp_yelpchi_v2_fast.json`: minimal developer config for quick end-to-end checks
- `configs/exp_yelpchi_v2_full.json`: larger v2 config with full severity curves
