# TODO 06: Run the Experiment Matrix and Generate Plots (Implemented)

Goal: execute the final experiment matrix and produce publication-style artifacts (CSV + figures) for the report.

Implemented artifacts:
- Experiment launcher stage: `benchmark/matrix_stage.py`
  - Runner integration: `benchmark/run.py` supports `--stage matrix`
  - Runs baselines + PMP + SEC-GFD on the same cached graph variants and appends rows into `runs/<exp>/results.csv`.
- Plotting stage: `benchmark/plots_stage.py`
  - Runner integration: `benchmark/run.py` supports `--stage plots`
  - Writes plot-ready summaries and figures to `runs/<exp>/plots/`:
    - `summary_curves.csv`
    - `performance_drop_max_stress.csv`
    - `robustness_scores.csv` (optional extra)
    - `missing_or_error_runs.csv` (completeness report)
    - `curve__<scenario_id>__<metric>.png` figures

Work:
- Implement an experiment launcher that loops over:
  - dataset(s)
  - split_id(s)
  - scenario and severity
  - random seeds
  - models (mlp, baseline gnn, pmp, sec-gfd)
- Ensure every run appends exactly one row to `results.csv` with:
  - identifiers (dataset, split_id, seed, scenario, severity, model)
  - metrics (auc, ap, f1_macro)
  - graph stats (edge count, heterophily ratio, degree summary)
  - runtime info (optional)
- Generate plots from `results.csv`:
  - Metric vs severity curves per scenario with error bars (mean +/- std).
  - Clean vs max-stress performance drop per model.
  - Optional: robustness score (area under the metric-vs-severity curve).

Done when:
- `results.csv` is complete for the chosen matrix (no missing combinations).
- Plots are saved (example: `runs/plots/*.png`) and match the numbers in `results.csv`.

Notes:
- If compute is limited, prioritize YelpChi and reduce the matrix before dropping seeds (variance matters).

How to run:

0) Install plotting dependency (once, inside your activated venv):

```bash
py -m pip install matplotlib
```

1) Generate cached graphs (once per experiment config):

```bash
py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage graphs --force
```

2) Run the full model matrix (baselines + PMP + SEC-GFD):

```bash
py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage matrix
```

3) Generate plots and plot-ready summaries:

```bash
py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage plots
```
