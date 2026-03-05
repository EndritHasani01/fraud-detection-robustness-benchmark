# Operational Improvements: Preflight Estimator, Progress Reporting, and Model Filtering

## Context

The v1 pipeline provides no upfront visibility into how large an experiment will be before it starts running. A user can accidentally launch a 2-day run without realizing it. Similarly, once a run is in progress, there are no progress counters or ETAs. These are pure quality-of-life improvements that make the benchmark usable in practice, especially on time-limited environments like Google Colab.

## Preflight Runtime Estimator

Before any stage begins training, print a summary of what is about to happen. Implement this as a function in a new module `benchmark/preflight.py` (or inline in `run.py` if you prefer to keep things flat). The function takes the loaded config and the filtered variant list, and prints:

- Number of graph variants selected (after filtering by `--include-noop`, `--only-clean`, `--max-variants`)
- Number of training seeds (after `--max-training-seeds` cap)
- Number of models that will be evaluated
- Total expected run count: variants * training_seeds * models
- Number of already-completed runs (from the skip-existing check, if enabled) and how many remain
- Optionally, estimated wall time. This can be computed from the median `duration_sec` of completed runs in `results.csv` for each model. If no prior data exists, print "no estimate available" rather than guessing.

This printout should appear at the start of `--stage baselines`, `--stage pmp`, `--stage secgfd`, `--stage matrix`, and `--stage shift`. For `--stage graphs`, print the number of variants that will be generated instead.

The output format should be compact and scannable, for example:

```
[preflight] 13 variants x 3 seeds x 2 models = 78 runs total
[preflight] 45 already done, 33 remaining
[preflight] estimated ~12 min (median 22s/run for baselines)
```

## Progress Reporting During Training

Each training stage currently prints nothing between the start and end of its loop. Add progress counters that print after each completed run. The format should show the position within the overall matrix and the time taken for that run:

```
[baselines] [12/78] sage / noise_edges / sev=0.2 / gs=0 / ts=2  ok  auc=0.831  18.4s
```

Additionally, compute a rolling ETA based on the moving average of the last 5 run durations. Print this every 10 runs or on every run if total runs are fewer than 50:

```
[baselines] [12/78] ... ETA ~20 min
```

Implement this as a lightweight progress tracker class (or just local variables in the stage loop). Do not introduce a dependency on `tqdm` or similar; plain print statements are sufficient and keep the output greppable.

## Per-Model Subset Filtering (--models flag)

Add a `--models` CLI flag to `benchmark/run.py` that accepts a comma-separated list of model IDs (e.g., `--models mlp,sage`). When specified, only those models are run in any training stage. This enables two common workflows:

1. Run the fast baselines first to get quick results: `--models mlp,sage`
2. Run expensive models separately, possibly with different configs: `--models pmp,secgfd --max-variants 5`

The `--models` value should be passed down to each stage function. Inside the stage, the model list is intersected with the models defined in the config. If `--models` is not specified, all models in the config are used (current behavior).

For stages that are model-specific (e.g., `--stage pmp` only runs PMP), the `--models` flag acts as a safety check: if `--models sage` is passed with `--stage pmp`, the stage should print a warning and exit cleanly without running anything.

## Interaction with Matrix Stage

The `--stage matrix` launcher in `benchmark/matrix_stage.py` calls baselines, PMP, and SEC-GFD in sequence. With the `--models` flag, it should only call the stages for models in the requested list. For example, `--models mlp,sage` should skip the PMP and SEC-GFD stages entirely, and `--models pmp` should skip baselines and SEC-GFD.
