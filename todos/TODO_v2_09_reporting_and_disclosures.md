# Reporting, Disclosures, and Evaluation Improvements

## Context

The v2 plan includes several improvements to how results are reported, how oracle perturbations are disclosed, and optional enhancements like confidence intervals and multiple data splits. These are lower priority than the core performance and resumability work, but they matter for the final paper and for anyone reviewing the benchmark's outputs.

## Oracle Disclosure in Outputs

Some perturbation scenarios (heterophily_rewire_oracle, camouflage_feature_oracle) use ground-truth labels to construct the stress test. This is legitimate for robustness benchmarking but must be disclosed prominently. Currently, `graph_variants.csv` has an `oracle_labels` column, but this information does not propagate to the plots or summary CSVs.

In `benchmark/plots_stage.py`, update the plot titles and legend entries to indicate when a scenario is oracle-based. For example, the title should read `"heterophily_rewire_oracle (oracle)"` or append a dagger symbol. In the summary CSVs (`summary_curves.csv`, `performance_drop_max_stress.csv`, `robustness_scores.csv`), add an `oracle_labels` column that is `true` for oracle scenarios and `false` otherwise. This column can be derived from the scenario config entries.

## Confidence Intervals

The current aggregation computes mean and standard deviation across training seeds. For the final report, bootstrap confidence intervals (95% CI) can be more informative, especially with small sample sizes (3-5 seeds). Add an optional `--ci` flag to the plots stage that, when enabled, computes bootstrap 95% CIs instead of (or in addition to) raw std. Use `numpy.random.choice` with 1000 bootstrap resamples. This is a nice-to-have and should not block the core v2 work.

If implemented, the error bars on plots switch from +/- 1 std to the CI bounds. The summary CSVs should include `ci_lower` and `ci_upper` columns alongside `mean` and `std`.

## Multiple Data Splits

v1 uses a single fixed split (s0). For stronger conclusions, v2 can support multiple splits by adding entries to `data_splits` in the config:

```json
"data_splits": [
  {"split_id": "s0", "train_size": 0.4, "val_size": 0.2, "split_seed": 717},
  {"split_id": "s1", "train_size": 0.4, "val_size": 0.2, "split_seed": 823},
  {"split_id": "s2", "train_size": 0.4, "val_size": 0.2, "split_seed": 456}
]
```

The pipeline already loops over `data_splits` in the graph stage, so adding more entries should work without code changes for graph generation. The training stages also iterate over variant rows that include `split_id`, so they should naturally handle multiple splits. The plots stage, however, currently does not aggregate across splits; it groups by `(dataset_id, split_id)`. To report mean/std across splits, add a cross-split aggregation step that computes statistics over the `(scenario_id, severity, model_id, metric)` group, treating each split's mean as one observation.

This is optional and compute-intensive (triples the total runs), so it should only appear in the `v2_full` config if at all. The code support should be there but not required.

## Completeness Report Enhancements

The existing `_completeness_report` function in `plots_stage.py` writes `missing_or_error_runs.csv`. Enhance it to also print a summary to stdout when the plots stage runs:

```
[plots] Completeness: 78/84 expected runs present (93%), 3 errors, 3 missing
[plots] Missing runs: pmp/noise_edges/sev=0.2/gs=1/ts=2, ...
```

This helps users understand what data gaps exist before looking at the plots.
