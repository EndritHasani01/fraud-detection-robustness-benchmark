# V2 Experiment Config Files

## Context

The v1 config (`configs/exp_yelpchi_v1.json`) defines a large experiment matrix that is impractical to run in full. v2 needs new config files that incorporate the decoupled seed schema, reduced severity grids, and tuned model hyperparameters. Three configs should be created, each serving a different purpose.

## Config 1: `configs/exp_yelpchi_v2.json` (Full V2 Experiment)

This is the primary v2 config for producing publishable results. It should include:

- `graph_seeds`: `[0, 1]` (2 seeds for perturbation variance)
- `training_seeds`: `[0, 1, 2]` (3 seeds for model variance)
- Coarse severity grid: each scenario should use only 2 severity points initially: `[0.0, <max>]` where `<max>` is the highest value from v1 (0.3 for rewire/camouflage, 0.2 for noise). This produces clean-vs-stressed drop comparisons quickly. Intermediate points (0.1, 0.2) can be added later by editing the config.
- SEC-GFD hparams tuned for feasibility: `hid_dim=32`, `high_order=1`, `order_d=2`, added under the model entry.
- PMP hparams with `full_neighbors=false` and `sampled_neighbors=[10]` for cost reduction.
- All four models: mlp, sage, pmp, secgfd.
- Same dataset and split as v1.
- `experiment_name`: `"gfd_robustness_benchmark_v2"`.

The expected matrix size with this config is roughly: (3 scenarios * 1 nonzero severity * 2 graph seeds + 1 clean) * 3 training seeds * 4 models = 84 total runs, which should complete in under 2 hours on a T4.

## Config 2: `configs/exp_yelpchi_v2_fast.json` (Fast Dev Config)

A deliberately minimal config for verifying the pipeline end-to-end in minutes. It should include:

- `graph_seeds`: `[0]`
- `training_seeds`: `[0]`
- Severity grid: only `[0.0, <max>]` for each scenario (same as v2 but with 1 seed)
- Only baseline models: mlp, sage (no pmp/secgfd entries)
- `max_epochs` hint in model hparams: 20 for mlp, 20 for sage (or rely on CLI override)
- `experiment_name`: `"gfd_robustness_dev"`

This config exists so that a developer can run `py -m benchmark.run --config configs/exp_yelpchi_v2_fast.json --stage graphs && py -m benchmark.run --config configs/exp_yelpchi_v2_fast.json --stage baselines` and get a complete `results.csv` in under 5 minutes on CPU. It is not intended for publishable results.

## Config 3: `configs/exp_yelpchi_v2_full.json` (Full Curve Config)

This config adds the intermediate severity points back for researchers who have enough compute to produce full degradation curves:

- `graph_seeds`: `[0, 1]`
- `training_seeds`: `[0, 1, 2, 3, 4]`
- Full severity grid matching v1: `[0.0, 0.1, 0.2, 0.3]` for rewire/camouflage, `[0.0, 0.05, 0.1, 0.2]` for noise
- All four models with v2-tuned hparams
- `experiment_name`: `"gfd_robustness_benchmark_v2_full"`

This is the largest v2 config and represents the upper bound of what someone might run if they have access to a GPU for an extended period. With skip-existing, it can be completed incrementally across multiple sessions.

## Validation

After creating these configs, verify that each one passes `validate_config` without errors. Also verify that the v1 config still passes validation (backward compat). If the config validation function needs changes to accept `graph_seeds`, those changes belong in the seed decoupling TODO but should be tested here.
