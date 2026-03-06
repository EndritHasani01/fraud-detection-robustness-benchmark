# Fraud Detection Robustness Benchmark

This repository contains a reproducible evaluation harness for graph-based fraud detection robustness. The current frozen configs target YelpChi from `dgl.data.FraudDataset` and support up to four integrated models:

- `mlp`
- `sage`
- `pmp` (LA-SAGE-S from `Repos/PMP-master`)
- `secgfd` (from `Repos/SEC-GFD-main`)

The benchmark caches graph variants once, runs model evaluations against those cached graphs, writes a unified `results.csv`, and generates plot-ready CSV summaries and figures for the final report.

## What Changed In V2

- Resumable execution is now the default: training stages skip completed run keys already present in `results.csv`.
- Graph seeds are decoupled from training seeds via `seeds.graph_seeds` and `seeds.training_seeds`.
- A second evaluation protocol is available: train once on the clean graph and evaluate on all cached variants (`--stage shift` or `--stage matrix --protocol train_clean_eval_all`).
- PMP and SEC-GFD now support benchmark-side runtime controls through config `hparams` and SEC-GFD CLI overrides.
- Training stages print a preflight summary, per-run progress lines, and rolling ETA.
- `--models` can restrict any training stage to a subset such as `mlp,sage` or `pmp,secgfd`.
- The plots stage now emits completeness diagnostics, protocol-aware outputs, oracle-disclosure columns, optional bootstrap confidence intervals, and cross-split summary CSVs.

`configs/exp_yelpchi_v1.json` remains frozen for backward-compatible reruns. If `graph_seeds` is absent, the graph stage falls back to `training_seeds`.

## Repository Layout

- `benchmark/`: runner, stage implementations, shared helpers, result/schema utilities
- `configs/`: frozen experiment definitions
- `Repos/`: vendored research repositories used through thin adapters
- `todos/`: v1 and v2 implementation tasks
- `tests/`: regression tests for the benchmark harness
- `FINAL_PROJECT_GUIDE.md`: report methodology and deliverables
- `Paper_Summary.md`: task 1 write-up

## Environment

This workspace assumes Windows + PowerShell. Prefer `py ...` over `python ...`.

Python 3.11 is the safest option on Windows for the pinned DGL/PyTorch stack:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
. .\.venv\Scripts\Activate.ps1

py -m pip install "numpy<2" scipy
py -m pip install torch==2.3.0+cpu --index-url https://download.pytorch.org/whl/cpu
py -m pip install https://data.dgl.ai/wheels/dgl-2.2.1-cp311-cp311-win_amd64.whl
py -m pip install torchdata==0.8.0 PyYAML pydantic matplotlib
```

`matplotlib` is only required for `--stage plots`.

## Frozen Configs

| Config | Purpose | Models | Seeds |
|---|---|---|---|
| `configs/exp_yelpchi_v1.json` | Frozen v1 experiment for backward-compatible reruns | `mlp`, `sage`, `pmp`, `secgfd` | No `graph_seeds`; graph stage falls back to `training_seeds` |
| `configs/exp_yelpchi_v2_fast.json` | Fast developer smoke test | `mlp`, `sage` | 1 graph seed, 1 training seed |
| `configs/exp_yelpchi_v2.json` | Main v2 benchmark | `mlp`, `sage`, `pmp`, `secgfd` | 2 graph seeds, 3 training seeds |
| `configs/exp_yelpchi_v2_full.json` | Larger v2 run with fuller severity curves | `mlp`, `sage`, `pmp`, `secgfd` | 2 graph seeds, 5 training seeds |

`configs/README.md` summarizes the same set of frozen experiment definitions.

## Common Workflows

Fast end-to-end smoke test on CPU:

```powershell
py -m benchmark.run --config configs/exp_yelpchi_v2_fast.json --stage graphs
py -m benchmark.run --config configs/exp_yelpchi_v2_fast.json --stage baselines
py -m benchmark.run --config configs/exp_yelpchi_v2_fast.json --stage plots
```

Main v2 benchmark with all integrated models:

```powershell
py -m benchmark.run --config configs/exp_yelpchi_v2.json --stage graphs
py -m benchmark.run --config configs/exp_yelpchi_v2.json --stage matrix
py -m benchmark.run --config configs/exp_yelpchi_v2.json --stage plots --ci
```

Shift-protocol benchmark: train on the clean graph once, evaluate all variants:

```powershell
py -m benchmark.run --config configs/exp_yelpchi_v2.json --stage graphs
py -m benchmark.run --config configs/exp_yelpchi_v2.json --stage matrix --protocol train_clean_eval_all
py -m benchmark.run --config configs/exp_yelpchi_v2.json --stage plots --ci
```

Run only expensive models on an existing cache:

```powershell
py -m benchmark.run --config configs/exp_yelpchi_v2.json --stage matrix --models pmp,secgfd
```

Quick sanity check on the clean graph only:

```powershell
py -m benchmark.run --config configs/exp_yelpchi_v2.json --stage baselines --only-clean --max-training-seeds 1
```

## Stage Reference

| Stage | What it does | Main outputs |
|---|---|---|
| `graphs` | Builds deterministic split masks, caches the base graph and all configured perturbation variants, and writes the variant ledger. | `config.json`, `graphs/`, `graph_variants.csv`, `results.csv` header |
| `baselines` | Trains/evaluates `mlp` and `sage` on cached graphs using the standard per-variant training protocol. | `results.csv`, `results_summary_baselines.csv` |
| `pmp` | Trains/evaluates PMP on cached graphs with benchmark-side config overrides merged on top of the research repo YAML. | `results.csv`, `results_summary_pmp.csv` |
| `secgfd` | Trains/evaluates SEC-GFD on cached graphs with benchmark-side hparams and optional CLI overrides. | `results.csv`, `results_summary_secgfd.csv` |
| `shift` | Trains each selected model once on the clean graph for a split and evaluates that artifact on every selected variant. | `results.csv`, `results_summary_shift.csv` |
| `matrix` | Launches the integrated model matrix. With `--protocol train_clean_eval_all`, it dispatches to the shift protocol instead of per-variant training. | Same outputs as the invoked child stages |
| `plots` | Reads `results.csv`, checks completeness, and writes figures plus plot-ready CSV summaries under `runs/<exp>/plots/`. | `plots/*.csv`, `plots/<protocol>/<dataset>/<split>/*.png` |

## Important CLI Flags

| Flag | Behavior |
|---|---|
| `--skip-existing` | Enabled by default. Skips run keys already present in `results.csv`. |
| `--no-skip-existing` | Disables resumability checks and appends fresh rows. |
| `--retry-errors` | Re-attempts run keys that only have `status=error` rows. |
| `--force` | Overwrites headers/cached outputs for the stage and disables skip-existing. Use sparingly. |
| `--models mlp,sage` | Restricts the selected stage to a model subset. |
| `--protocol train_clean_eval_all` | On `--stage matrix`, runs the clean-train shift protocol instead of per-variant training. |
| `--only-clean` | Evaluates only the clean graph row. |
| `--include-noop` | Includes non-clean scenario rows with severity `0.0`. |
| `--max-variants N` | Limits the number of variant rows loaded from `graph_variants.csv`. |
| `--max-training-seeds N` | Caps the number of training seeds used from the config. |
| `--max-epochs N`, `--patience N` | Global training-control overrides for the training stages. |
| `--secgfd-hid-dim`, `--secgfd-order-d`, `--secgfd-high-order` | SEC-GFD-only CLI overrides for fast runtime experiments. |
| `--ci` | On `--stage plots`, computes bootstrap 95% confidence intervals and uses them for error bars. |

## Outputs

Each run writes to `runs/<experiment_name>/` unless `--out` is supplied.

Core artifacts:

- `config.json`: frozen copy of the config used for that run
- `graphs/`: cached base graph and perturbation variants
- `graph_variants.csv`: graph ledger with `graph_seed`, `scenario_applied`, and `oracle_labels`
- `results.csv`: unified per-run results across all models and protocols

Per-stage summaries:

- `results_summary_baselines.csv`
- `results_summary_pmp.csv`
- `results_summary_secgfd.csv`
- `results_summary_shift.csv`

Plot/report artifacts under `plots/`:

- `summary_curves.csv`
- `performance_drop_max_stress.csv`
- `robustness_scores.csv`
- `summary_curves_cross_split.csv`
- `performance_drop_max_stress_cross_split.csv`
- `robustness_scores_cross_split.csv`
- `missing_or_error_runs.csv`
- `plots/<protocol>/<dataset>/<split>/curve__<scenario>__<metric>.png`
- `plots/<protocol>/<dataset>/across_splits/*.png` when multiple `data_splits` are present

## Results Schema Notes

The unique run key in `results.csv` is:

```text
(dataset_id, split_id, scenario_id, severity, graph_seed, training_seed, model_id, protocol)
```

Important columns added or emphasized in v2:

- `protocol`: `train_on_variant` or `train_clean_eval_all`
- `train_graph_ref`: path to the graph used for training; for shift runs this points at the clean graph
- `average_precision`: added alongside `roc_auc` and `f1_macro`
- `status` and `error`: support resumability, retry, and completeness reporting
- `duration_sec`: populated for both successful and failed runs

## Reporting And Disclosure Notes

- Oracle perturbations are explicitly disclosed. `graph_variants.csv` carries `oracle_labels`, and the plots stage propagates that field into summary CSVs and plot legends/titles.
- The plots stage is protocol-aware. Standard and shift-protocol runs are written to separate output subdirectories and remain distinguishable in plot CSVs.
- With `--ci`, plot CSVs include `ci_lower` and `ci_upper` columns in addition to mean/std.
- Cross-split summary CSVs are always written; they are most useful when the config defines multiple `data_splits`.
- The plots stage prints a completeness summary and writes `missing_or_error_runs.csv` so report gaps are visible before figures are used.

## Reproducibility

- Split masks are deterministic per `split_id`.
- Graph perturbations are deterministic per `graph_seed`.
- Thresholds are chosen on validation only; no test leakage is allowed.
- `results.csv` is the single source of truth for all final tables and figures.

## Troubleshooting

- If DGL import fails with a missing GraphBolt DLL, use `torch==2.3.0+cpu` with the pinned DGL wheel above.
- If NumPy 2.x causes binary-compatibility errors, reinstall with `py -m pip install "numpy<2"`.
- On Windows, PMP forces `num_workers=0` for stability even if a higher value is placed in the config.
