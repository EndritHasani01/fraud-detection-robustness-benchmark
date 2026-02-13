# Agent Notes (Repo Working Agreement)

This repository is a course final project: a robustness benchmark for graph-based fraud detection.

The main goal is to provide a small, reproducible evaluation harness that can:
- load a fraud graph dataset (YelpChi/Amazon),
- generate controlled stress tests (heterophily/camouflage/noise),
- run multiple models (MLP + baseline GNN + 2 specialized methods from `Repos/`),
- write a single `results.csv` and plots for the final report.

## Repo Layout

- `Paper_Summary.md`: Task 1 write-up (motivation + related work).
- `FINAL_PROJECT_GUIDE.md`: the concrete methodology and report outline.
- `configs/`: frozen experiment definitions (single source of truth).
- `todos/`: task breakdown.
- `benchmark/`: the integration runner (benchmark code you extend).
- `Repos/`: official research repositories (vendored).

## Environment Assumptions

- This workspace is Windows + PowerShell.
  - Prefer `py ...` over `python ...` (the `python` alias may not exist).
  - PowerShell does not support `||` as a command separator; use `;` or explicit `if` checks.
- For ML dependencies, Python 3.10/3.11 is the safest choice for DGL/PyTorch wheels.

## How To Run (Current State)

Graphs + caching stage (builds split + caches base and perturbed graphs):

```bash
py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage graphs
```

Baselines stage (MLP + GraphSAGE) on cached graphs:

```bash
py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage baselines
```

PMP stage (specialized method) on cached graphs:

```bash
py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage pmp
```

SEC-GFD stage (specialized method) on cached graphs:

```bash
py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage secgfd
```

Matrix stage (runs baselines + PMP + SEC-GFD):

```bash
py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage matrix
```

Plots stage (generates figures under `runs/<exp>/plots/`):

```bash
py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage plots
```

Notes:
- `--stage plots` requires `matplotlib` in the active environment (`py -m pip install matplotlib`).

Quick sanity check (clean graph only, first training seed only):

```bash
py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage baselines --only-clean --max-training-seeds 1
```

Outputs go to `runs/<experiment_name>/`:
- `config.json` (config snapshot)
- `graphs/` (cached base + variant graphs)
- `graph_variants.csv` (ledger + graph stats)
- `results.csv` (per-run metrics rows; baselines populate this now)
- `results_summary_baselines.csv` (mean/std across training seeds for baselines)
- `results_summary_pmp.csv` (mean/std across training seeds for PMP)
- `results_summary_secgfd.csv` (mean/std across training seeds for SEC-GFD)

`runs/` is ignored by git via `.gitignore`.

## Engineering Guidelines

- Treat `configs/*.json` as frozen experiment definitions:
  - If you change the experiment design, create a new config file (do not silently mutate old ones).
- Keep the benchmark runner (`benchmark/`) independent of `Repos/`:
  - Prefer importing/calling research code via a thin adapter layer.
  - If you must patch a research repo to load external graphs or fix evaluation leakage, keep changes minimal and document them in the report.
- Reproducibility requirements:
  - Fixed split masks per `split_id`.
  - Deterministic perturbations given `graph_seed`.
  - No test leakage: thresholds chosen on validation only.
- Output requirements:
  - One unified `results.csv` schema for all models.
  - All plots/tables in the report should be traceable to rows in `results.csv`.

## Style / Safety

- Prefer small, focused commits/patches (even if you do not commit, keep diffs reviewable).
- Do not add heavy tooling (formatters/lint frameworks) unless the repo already uses them.
- Avoid touching files in `Repos/` unless necessary for integration or correctness.
