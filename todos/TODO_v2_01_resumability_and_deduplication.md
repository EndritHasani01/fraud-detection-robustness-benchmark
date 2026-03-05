# Resumability and Result Deduplication

## Context

The v1 pipeline appends rows to `results.csv` without checking whether a given combination has already been evaluated. This means re-running any stage duplicates rows, and there is no way to safely interrupt a long run and resume it later without manual cleanup. This is the single most important operational fix for v2 because all other improvements (seed decoupling, cheaper configs, etc.) still require the ability to run the experiment matrix incrementally.

## Run Key Definition

Every result row is uniquely identified by a composite key consisting of:

`(dataset_id, split_id, scenario_id, severity, graph_seed, training_seed, model_id, protocol)`

The `protocol` field is new in v2. For all existing v1 rows it should default to `"train_on_variant"` (the model was trained on the same graph it was evaluated on). The train-clean-eval-perturbed protocol introduced separately will use `"train_clean_eval_all"`. Add `protocol` as a new column in `RESULTS_COLUMNS` inside `benchmark/results.py`, placed after `model_id`. Every stage that writes rows must populate it.

## Skip-if-Done Logic

Implement a function in `benchmark/results.py` (suggested name: `load_completed_keys`) that reads `results.csv` once at the start of a stage and returns the set of run keys whose `status` column equals `"ok"`. Each training stage (`baselines_stage.py`, `pmp_stage.py`, `secgfd_stage.py`, and the future `shift_stage.py`) must call this function before entering its variant/seed loop. Inside the inner loop, before loading a graph or constructing a model, the stage checks whether the current run key exists in the completed set. If it does, the stage prints a short skip message (e.g., `[skip] mlp / clean / sev=0.0 / gs=0 / ts=42 already done`) and moves to the next iteration.

## CLI Flags

Add two flags to the argument parser in `benchmark/run.py`:

- `--skip-existing` (default: enabled). When active, the skip-if-done logic described above is used. This should be the default behavior so that resuming a run is safe out of the box.
- `--force` already exists but its semantics should be clarified: when `--force` is passed, skip-existing is disabled regardless, and the CSV header is overwritten (existing behavior). This means `--force` and `--skip-existing` are mutually exclusive in effect; if both are passed, `--force` wins.
- `--retry-errors`. When this flag is active, rows with `status=error` for a given run key are treated as incomplete (i.e., they are not skipped). The stage should attempt the run again and, if successful, append a new `status=ok` row. The old error row remains in the CSV; the plots stage already filters by `status=ok`, so duplicates with mixed status are harmless. If the re-run also fails, a second error row is appended.

## Preventing Duplicate OK Rows

Even with skip-existing, two processes could theoretically race. For this project (single-machine, single-process), the skip-existing check is sufficient. However, the `append_csv_row` function should be wrapped in a file-level lock (e.g., `fcntl` on Linux / `msvcrt` on Windows, or simply a threading lock since stages run sequentially). This is low priority but worth noting.

## Error Handling Improvements

When a training run raises an exception, the current code already writes `status=error` with the exception message. Improve this by also writing `duration_sec` (the wall time elapsed before the error) so that slow failures can be distinguished from fast ones. Ensure the error message is truncated to 500 characters to avoid CSV formatting issues from long tracebacks.

## Verification

After implementing these changes, verify the following behavior manually or with a small integration test:

1. Run `--stage baselines --config configs/exp_yelpchi_v2_fast.json` (the dev config, created in a separate TODO). It should produce N rows.
2. Run the same command again. It should skip all N rows and produce no new rows.
3. Manually edit one row's status from `ok` to `error`. Run with `--retry-errors`. Only that one row should be re-attempted.
4. Run with `--force`. All rows should be re-computed and the CSV should be overwritten.
