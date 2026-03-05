# PMP Sampling Configuration

## Context

PMP (Partitioning Message Passing) uses DGL's neighbor sampling infrastructure for mini-batch training. The benchmark's PMP stage (`benchmark/pmp_stage.py`) loads its training hyperparameters from the research repo's YAML config (`Repos/PMP-master/config/yelp.yml`), which sets `full_neighbors: true` by default. Full-neighbor sampling on YelpChi's 8M-edge graph is expensive. v2 should allow the benchmark to override PMP's sampling behavior without modifying files inside the research repo.

## Benchmark-Side Overrides

Add an optional `hparams` dict to the PMP model entry in the experiment JSON config, similar to what is done for SEC-GFD:

```json
{
  "model_id": "pmp",
  "repo_path": "Repos/PMP-master",
  "hparams": {
    "full_neighbors": false,
    "sampled_neighbors": [10, 5],
    "batch_size": 512,
    "num_workers": 0,
    "epochs": 100,
    "patience": 10
  }
}
```

In `run_pmp_stage`, after loading the YAML config via `_load_pmp_yaml_config`, merge the benchmark-side overrides on top. This means the YAML provides the base PMP configuration and the experiment JSON provides selective overrides. The merge should be a shallow dict update: `cfg_pmp.update(pmp_model_cfg.get("hparams", {}))`.

## Key Parameters to Expose

The following PMP parameters have the largest impact on cost and should be explicitly overridable:

- `full_neighbors` (bool): When false, uses `NeighborSampler` with fanouts instead of `MultiLayerFullNeighborSampler`. Setting this to false with reasonable fanouts (e.g., [10] or [10, 5]) significantly reduces per-batch memory and compute.
- `sampled_neighbors` (list of ints): Fanout per layer when `full_neighbors` is false. The length should match `n_layer`.
- `batch_size` (int): Larger batches can be faster on GPU but use more memory.
- `num_workers` (int): On Linux/Colab, increasing this (e.g., to 2 or 4) improves data loading throughput. On Windows, it must remain 0 for stability due to DGL/PyTorch multiprocessing issues. The `_make_dataloaders` function in `pmp_stage.py` currently hardcodes `num_workers=0`. Change it to read from `cfg_pmp.get("num_workers", 0)`.
- `epochs` and `patience`: Already partially exposed via `--max-epochs` and `--patience` CLI flags, but should also be overridable from the config.

## Platform-Aware Defaults

Add a helper that detects the platform (`sys.platform`) and sets `num_workers` to 0 on Windows regardless of what the config says. On Linux, default to the config value or 0 if unspecified. This prevents cryptic DGL crashes on Windows when someone copies a Colab config locally.

## No Changes to Research Repo

All changes must be confined to `benchmark/pmp_stage.py` and the experiment config files. The files under `Repos/PMP-master/` must not be modified, because the benchmark's value proposition includes running research code as-is.
