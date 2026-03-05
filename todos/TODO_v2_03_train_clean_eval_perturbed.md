# Train-Clean, Eval-Perturbed Protocol

## Context

In v1, every (variant, training_seed, model) combination means training a model from scratch on that variant's graph. This is the most expensive part of the pipeline. A common and defensible alternative in robustness literature is to train the model once on the clean graph and then evaluate it (forward pass only) on each perturbed variant. This answers the question "how robust is a model to test-time distribution shift?" rather than "how well can a model learn under structural stress?", and it is dramatically cheaper because it replaces N trainings with 1 training + N forward passes.

## New Stage: `shift` (or `matrix_shift`)

Implement this as a new stage that can be invoked with `--stage shift`. It should be a new file `benchmark/shift_stage.py`. The stage works as follows:

1. For each (dataset_id, split_id, model_id, training_seed), train the model once on the clean base graph. This is identical to what `baselines_stage.py` / `pmp_stage.py` / `secgfd_stage.py` already do for the clean variant, so the trained model state dict can be reused if the clean run already exists in `results.csv`, or the stage trains from scratch and saves the state dict to a temporary location.

2. For each perturbed variant in `graph_variants.csv`, load the trained model, run a forward pass on the perturbed graph, compute metrics, and write a row to `results.csv`. The `protocol` column (introduced in the resumability TODO) must be set to `"train_clean_eval_all"` so that these rows are distinguishable from the standard protocol rows.

3. The `train_graph_ref` column should also be added to `RESULTS_COLUMNS` to record which graph the model was trained on (always the clean graph path for this protocol).

## Model State Caching

Training a model and then evaluating it on many variants requires the trained model weights to be available across variants. The simplest approach is to hold the model in memory within the stage's loop. The outer loop iterates over (model_id, training_seed), trains once, and then the inner loop iterates over all variant rows for forward-pass evaluation. This avoids serializing state dicts to disk, though disk serialization (e.g., to `runs/<exp>/checkpoints/<model_id>_<training_seed>.pt`) is a reasonable alternative if memory is a concern.

For the baseline models (MLP, GraphSAGE), the existing `train_eval_baseline` function trains and evaluates in one shot. Refactor it (or add a companion function) so that training and evaluation are separable: one function returns a trained model, and another function accepts a model and a graph and returns metrics. The same refactoring applies to `train_eval_pmp` and `train_eval_secgfd`.

## Interaction with Skip-Existing

The shift stage should use the same skip-existing logic from the resumability TODO. The run key for shift rows includes `protocol="train_clean_eval_all"`, which distinguishes them from standard rows even when the variant key is the same.

## CLI Integration

Add `"shift"` to the `--stage` choices in `benchmark/run.py`. The shift stage should accept the same flags as other training stages (`--device`, `--skip-existing`, `--max-variants`, etc.). Additionally, add a `--protocol` flag with choices `["train_on_variant", "train_clean_eval_all"]` that defaults to `"train_on_variant"`. When `--protocol train_clean_eval_all` is passed with `--stage matrix`, the matrix stage should run the shift protocol instead of the per-variant training protocol.

## Plots Stage Awareness

The plots stage must be updated to handle the `protocol` column. By default, plots should be generated per-protocol (separate curves for "trained on variant" vs. "trained on clean"). Alternatively, both can appear on the same plot with different line styles (solid for standard, dashed for shift). The summary CSVs should include the `protocol` field so that the report can reference which protocol produced which numbers.
