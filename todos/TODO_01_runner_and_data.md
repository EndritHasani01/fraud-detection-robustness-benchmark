# TODO 01: Build the Benchmark Runner and Data Pipeline

Goal: create one reproducible runner that can generate graph variants, train/evaluate models, and write results to disk.

Work:
- Create a benchmark package or scripts directory (example: `benchmark/`) with:
  - A single CLI entrypoint (example: `py -m benchmark.run --config configs/exp.json`).
  - Central config loading (dataset, splits, seeds, scenarios, severities, output directory).
  - A consistent results schema (one row per: dataset, split_id, seed, scenario, severity, model, metrics, runtime, graph stats).
- Implement dataset loading using DGL's fraud datasets:
  - YelpChi and Amazon via `dgl.data.FraudDataset` (or the specific dataset classes used in the repos).
  - Create/keep `train_mask`, `val_mask`, `test_mask` in the graph.
  - Ensure the same split masks are reused for all severities within a split.
- Implement caching:
  - Cache the base graph for each split.
  - Cache each perturbed graph variant (scenario + severity + seed) so all models run on identical inputs.
  - Save graphs with `dgl.data.utils.save_graphs` and load with `load_graphs`.
- Implement stats logging helpers:
  - node count, edge count
  - mean/median degree
  - heterophily ratio (if labels available)

Done when:
- A single command can:
  - generate base split(s)
  - generate and cache graph variants
  - produce an empty-but-correct `results.csv` with the right columns (even before models are integrated)

Notes:
- Keep the runner independent of the research repos. The runner should call/invoke those repos (or import them) rather than being tightly coupled.

