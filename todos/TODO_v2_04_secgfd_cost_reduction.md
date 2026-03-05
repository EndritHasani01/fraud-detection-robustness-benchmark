# SEC-GFD Cost Reduction

## Context

SEC-GFD is the most expensive model in the benchmark. Its forward pass performs multiple full-graph DGL `update_all` operations (a low-order GCN pass and a high-order spectral pass), and the YelpChi homogeneous graph has roughly 8 million edges. A single training run can take many minutes even on a GPU, and the full experiment matrix calls for hundreds of these runs. Reducing the per-run cost of SEC-GFD is essential for making the benchmark feasible on student hardware.

## Expose Hyperparameter Knobs via Config

The function `train_eval_secgfd` in `benchmark/secgfd_stage.py` already accepts keyword arguments for `hid_dim`, `order_d`, `high_order`, `lemda`, `lr`, and `weight_decay`, but the calling code in `run_secgfd_stage` always uses the defaults (hid_dim=64, order_d=2, high_order=2). These should be configurable through the experiment JSON config.

Add an optional `hparams` dict to the SEC-GFD model entry in the config:

```json
{
  "model_id": "secgfd",
  "repo_path": "Repos/SEC-GFD-main",
  "hparams": {
    "hid_dim": 32,
    "order_d": 2,
    "high_order": 1,
    "lemda": 0.2,
    "lr": 0.01,
    "weight_decay": 0.0
  }
}
```

In `run_secgfd_stage`, read these from the config's model entry and pass them through to `train_eval_secgfd`. If `hparams` is absent, use the current defaults. The `--max-epochs` and `--patience` CLI overrides should continue to take precedence over config values.

Recommended v2 defaults for feasibility: `hid_dim=32`, `high_order=1`, `max_epochs=50`, `patience=10`. These reduce the model size and the number of spectral message-passing rounds while still producing meaningful results for a robustness comparison.

## Cache Polynomial Coefficient Computation

The SEC-GFD model class (in `Repos/SEC-GFD-main/model/SECGFD.py` or the High_GNN submodule) computes polynomial coefficients using SymPy/SciPy during `__init__`. This computation depends only on the polynomial degree `d` and is deterministic, but it runs every time a new model instance is created. Since the benchmark creates a fresh model for every (variant, training_seed) combination, this work is repeated hundreds of times with identical inputs.

Implement a module-level cache in `benchmark/secgfd_stage.py` that stores the computed coefficients keyed by `(order_d,)`. Before constructing the SECGFD model, check the cache. If the coefficients are available, monkey-patch or pre-set them on the model instance after construction. If the SEC-GFD code computes them inside `__init__` as a side effect, the practical approach is to let the first construction run normally, extract the computed values from the model's attributes, and store them. On subsequent constructions, overwrite the model's attributes with the cached values before training begins.

If monkey-patching is too fragile, an alternative is to pre-compute the coefficients once in a standalone function (copying the relevant SymPy/SciPy logic from the repo code) and pass them into the model. This requires understanding the specific attribute names used by SECGFD, so read through `Repos/SEC-GFD-main/model/High_GNN.py` to identify them.

## CLI Shorthand

For quick experimentation, add CLI flags `--secgfd-hid-dim`, `--secgfd-high-order`, and `--secgfd-order-d` to `benchmark/run.py`. These override the config values and are passed through to `run_secgfd_stage`. This allows testing different cost/quality tradeoffs without editing the config file.
