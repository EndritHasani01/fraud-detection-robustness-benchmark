# Agent Notes (Repo Working Agreement)

This repository is a course final project: a robustness benchmark for graph-based fraud detection.

The current v2 benchmark is expected to:

- load YelpChi from `dgl.data.FraudDataset`
- generate deterministic graph stress tests with decoupled graph and training seeds
- run both evaluation protocols: `train_on_variant` and `train_clean_eval_all`
- evaluate up to four integrated models: `mlp`, `sage`, `pmp`, `secgfd`
- write one unified `results.csv` plus stage summaries and report-ready plots

## Repo Layout

- `Paper_Summary.md`: task 1 write-up
- `FINAL_PROJECT_GUIDE.md`: methodology and report outline
- `configs/`: frozen experiment definitions
- `todos/`: v1 and v2 implementation tasks
- `benchmark/`: integration runner and shared helpers
- `tests/`: regression tests for config, stages, and reporting
- `Repos/`: vendored research repositories

Key shared modules added in v2:

- `benchmark/results.py`: result schema, run-key helpers, resumability helpers
- `benchmark/variants.py`: shared variant parsing/filtering/loading utilities
- `benchmark/preflight.py`: model selection, preflight counting, progress reporting

## Environment Assumptions

- This workspace is Windows + PowerShell.
  - Prefer `py ...` over `python ...`.
  - PowerShell does not support `||`; use `;` or explicit conditionals.
- Python 3.10/3.11 is the safest choice for DGL/PyTorch wheels.
- On Windows, PMP must keep `num_workers=0` for stability.

## Frozen Configs

- `configs/exp_yelpchi_v1.json`: frozen v1 benchmark
- `configs/exp_yelpchi_v2_fast.json`: minimal developer smoke-test config
- `configs/exp_yelpchi_v2.json`: main v2 benchmark
- `configs/exp_yelpchi_v2_full.json`: larger v2 run with fuller severity curves

`configs/*.json` are frozen experiment definitions. If the experiment design changes, add a new config file instead of mutating an old one.

## How To Run (Current State)

Build split masks and cache graphs:

```bash
py -m benchmark.run --config configs/exp_yelpchi_v2.json --stage graphs
```

Run baselines on cached graphs:

```bash
py -m benchmark.run --config configs/exp_yelpchi_v2.json --stage baselines
```

Run PMP only:

```bash
py -m benchmark.run --config configs/exp_yelpchi_v2.json --stage pmp
```

Run SEC-GFD only:

```bash
py -m benchmark.run --config configs/exp_yelpchi_v2.json --stage secgfd
```

Run the clean-train shift protocol directly:

```bash
py -m benchmark.run --config configs/exp_yelpchi_v2.json --stage shift
```

Run the integrated matrix with standard per-variant training:

```bash
py -m benchmark.run --config configs/exp_yelpchi_v2.json --stage matrix
```

Run the integrated matrix with the shift protocol:

```bash
py -m benchmark.run --config configs/exp_yelpchi_v2.json --stage matrix --protocol train_clean_eval_all
```

Generate figures and plot-ready CSVs:

```bash
py -m benchmark.run --config configs/exp_yelpchi_v2.json --stage plots --ci
```

Fast developer sanity check:

```bash
py -m benchmark.run --config configs/exp_yelpchi_v2_fast.json --stage graphs
py -m benchmark.run --config configs/exp_yelpchi_v2_fast.json --stage baselines
```

Useful runtime controls:

- `--models mlp,sage`: run only a subset of configured models
- `--only-clean`: evaluate only the clean graph row
- `--max-variants N`: cap the number of variant rows
- `--max-training-seeds N`: cap the number of training seeds
- `--retry-errors`: retry keys that only have `status=error`
- `--no-skip-existing`: disable resumable skipping
- `--force`: overwrite outputs for the stage and disable skip-existing
- `--secgfd-hid-dim`, `--secgfd-order-d`, `--secgfd-high-order`: SEC-GFD runtime overrides

## Operational Behavior

- Training stages are resumable by default. Completed run keys in `results.csv` are skipped.
- The run key is `(dataset_id, split_id, scenario_id, severity, graph_seed, training_seed, model_id, protocol)`.
- `protocol` distinguishes standard per-variant training from the clean-train shift protocol.
- `train_graph_ref` records which graph the model was trained on.
- Graph generation uses `seeds.graph_seeds` when present; v1 configs fall back to `seeds.training_seeds`.
- Every training stage prints a preflight summary and per-run progress lines with ETA.
- The plots stage emits completeness diagnostics and `missing_or_error_runs.csv`.

## Outputs

Outputs go to `runs/<experiment_name>/`. The exact summary files depend on which stages were run:

- `config.json`
- `graphs/`
- `graph_variants.csv`
- `results.csv`
- `results_summary_baselines.csv`
- `results_summary_pmp.csv`
- `results_summary_secgfd.csv`
- `results_summary_shift.csv`
- `plots/`

Important plot artifacts:

- `plots/summary_curves.csv`
- `plots/performance_drop_max_stress.csv`
- `plots/robustness_scores.csv`
- `plots/*_cross_split.csv`
- `plots/missing_or_error_runs.csv`
- `plots/<protocol>/<dataset>/<split>/*.png`

## Engineering Guidelines

- Treat `configs/*.json` as the single source of truth for experiment design.
- Keep `benchmark/` independent of `Repos/`; integrate research code through thin adapters.
- Avoid touching files in `Repos/` unless it is necessary for correctness or integration.
- Prefer shared helpers in `benchmark/results.py`, `benchmark/variants.py`, and `benchmark/preflight.py` for cross-cutting changes.
- Preserve v1 backward compatibility. Do not silently mutate `configs/exp_yelpchi_v1.json`.
- Reproducibility requirements:
  - fixed split masks per `split_id`
  - deterministic perturbations per `graph_seed`
  - no test leakage; thresholds come from validation only
  - all report artifacts must be traceable back to `results.csv`
- Reporting requirements:
  - disclose oracle perturbation scenarios in downstream outputs
  - keep protocol distinctions visible in summaries and plots
  - prefer one unified `results.csv` schema for all models

## Style / Safety

- Prefer small, reviewable patches.
- Do not add heavy tooling unless the repo already uses it.
- Keep changes to frozen configs explicit by creating a new config file.
- `runs/` is generated output and is ignored by git.
