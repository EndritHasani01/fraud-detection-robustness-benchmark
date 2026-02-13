# TODO 06: Run the Experiment Matrix and Generate Plots

Goal: execute the final experiment matrix and produce publication-style artifacts (CSV + figures) for the report.

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

