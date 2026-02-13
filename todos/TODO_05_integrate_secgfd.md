# TODO 05: Integrate SEC-GFD (Specialized Method 2)

Goal: run SEC-GFD from `Repos/SEC-GFD-main/` inside the benchmark on the same cached graph variants, with fair evaluation.

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

