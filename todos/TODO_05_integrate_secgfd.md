# TODO 05: Integrate SEC-GFD (Specialized Method 2) (Implemented)

Goal: run SEC-GFD from `Repos/SEC-GFD-main/` inside the benchmark on the same cached graph variants, with fair evaluation.

Implemented artifacts:
- SEC-GFD evaluation stage: `benchmark/secgfd_stage.py`
  - Runner integration: `benchmark/run.py` supports `--stage secgfd`
  - Writes per-run rows to: `runs/<experiment>/results.csv` with `model_id=secgfd`
  - Writes mean/std summary (across training seeds) to: `runs/<experiment>/results_summary_secgfd.csv`
- Minimal SEC-GFD repo patch for DGL GraphConv on graphs with isolated nodes:
  - `Repos/SEC-GFD-main/model/SECGFD.py` sets `allow_zero_in_degree=True` in the internal GCN's `GraphConv` layers.

Integration strategy used:
- The benchmark loads each cached `graph.bin` directly (from `graph_variants.csv`) and reuses the masks already stored in the graph (`train_mask`, `val_mask`, `test_mask`).
- Training/selection uses **validation only**:
  - best epoch is selected by validation ROC-AUC
  - threshold for macro-F1 is selected on validation, then applied once on test
- No test leakage: the training loop never consults the test split for model/threshold selection.
- Implementation note: the SEC-GFD repo's `cal_nceloss(...)` indexes cosine similarities using positions-within-train rather than global node IDs. In this benchmark we compute the same loss but index by the actual training node IDs (fixed indexing). Disclose this in the report as a patch/bugfix.

Work:
- Add external-graph loading support:
  - Patch `Repos/SEC-GFD-main/main.py` (or `Repos/SEC-GFD-main/dataset.py`) to accept a `--graph_path` that loads a DGL graph saved by the benchmark.
  - Ensure masks are reused from the loaded graph.
- Fix practical issues for reproducibility:
  - Make device selection configurable (do not hardcode `cuda:1`).
  - Make split seeds configurable if SEC-GFD generates splits internally.
- Fix evaluation leakage (important):
  - Do not use the test set during training to select thresholds or choose the "best" epoch.
  - Select thresholds on validation and compute test metrics once per run (or disclose if the original behavior is kept).
- Make SEC-GFD output machine-readable results:
  - Write JSON/CSV output for metrics per run.

Done when:
- SEC-GFD runs on clean + perturbed graphs produced by the benchmark and writes results into `results.csv`.
- The training loop does not consult test labels/mask for model selection or threshold tuning.

Notes:
- SEC-GFD currently assumes a homogeneous graph in its DGL pipeline; keep the benchmark representation consistent with this choice (or explicitly adapt SEC-GFD to heterographs).

How to run (after `--stage graphs` has produced cached graphs):

```bash
py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage secgfd
```

Useful flags:
- `--only-clean` runs SEC-GFD only on the clean graph (fast sanity check).
- `--max-training-seeds N` reduces runtime by using only the first N training seeds.
- `--max-epochs E --patience P` overrides training loop settings for quicker experiments.
