# Fraud Detection Robustness Benchmark — Project Status

> Last updated: 2026-03-04
> Last active development: 2026-02-13

---

## 1. What Is This Project?

This is a **course final project** for Theoretical Graphs that builds a reproducible benchmark for evaluating how robust graph-based fraud detection models are under controlled stress tests.

The core question it answers: *When we deliberately degrade a fraud graph (add noise, increase heterophily, camouflage fraudsters), which detection models break first and by how much?*

It uses the **YelpChi** fraud dataset (45,954 nodes, ~8M edges, 14.5% fraud rate) from DGL and evaluates four models — two baselines and two specialized research methods — across three types of graph perturbations at multiple severity levels.

---

## 2. What It Does

The benchmark is organized as a **6-stage pipeline**, each invoked via a CLI command:

```
py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage <STAGE>
```

| Stage | What It Does |
|-------|-------------|
| `graphs` | Loads YelpChi, generates 62 graph variants (clean + perturbed), caches them to disk |
| `baselines` | Trains and evaluates MLP and GraphSAGE on all variants |
| `pmp` | Trains and evaluates PMP (LA-SAGE-S, ICLR 2024) on all variants |
| `secgfd` | Trains and evaluates SEC-GFD (AAAI 2024) on all variants |
| `matrix` | Runs all three stages above in sequence (convenience launcher) |
| `plots` | Reads results CSV and generates publication-ready figures |

### Models

| Model | Type | What It Tests |
|-------|------|--------------|
| **MLP** | Baseline | Features only, no graph structure — lower bound |
| **GraphSAGE** | Baseline | Standard message-passing — typical GNN performance |
| **PMP** (LA-SAGE-S) | Specialized | Partitions messages by neighbor label status — designed for heterophily |
| **SEC-GFD** | Specialized | Spectral filtering + environmental constraints — designed for heterophily |

### Stress-Test Scenarios

| Scenario | What It Does | Severity Levels |
|----------|-------------|----------------|
| **Heterophily rewire** | Rewires edges so neighbors have opposite labels | 0%, 10%, 20%, 30% |
| **Feature camouflage** | Replaces fraud node features with normal-looking ones | 0%, 10%, 20%, 30% |
| **Edge noise** | Adds random directed edges to increase density | 0%, 5%, 10%, 20% |

Each perturbation is applied with 5 different random seeds, producing 62 total graph variants (1 clean + 61 perturbed). All models are evaluated with 5 training seeds per variant, yielding up to **1,240 individual training runs** for the full matrix.

### Evaluation Protocol

- Metrics: **ROC-AUC**, **Average Precision**, **F1-macro**
- Threshold selection: best F1 on validation set, applied once to test set (no test leakage)
- Reporting: mean and standard deviation across training seeds

---

## 3. What Has Been Completed

### Code (all implemented and committed)

- **TODO-00 through TODO-06** are done:
  - Project scaffolding and scope definition
  - CLI runner with config loading and validation
  - DGL dataset loading and train/val/test splitting (40/20/40)
  - Three perturbation scenarios with deterministic seeding
  - Graph caching and statistics computation
  - Baseline model definitions (MLP, GraphSAGE) and training loop
  - PMP integration (imports from vendored `Repos/PMP-master/`, mini-batch training via DGL DataLoader)
  - SEC-GFD integration (imports from vendored `Repos/SEC-GFD-main/`, full-graph training)
  - Matrix launcher that runs all models sequentially
  - Plots stage that generates metric-vs-severity curves with error bars
  - Custom metrics (ROC-AUC, AP, F1-macro) without sklearn dependency
  - Results CSV schema and seed-aggregation summaries

### Data & Partial Runs

- **62 graph variants** generated and cached (~11 GB in `runs/gfd_robustness_benchmark_v1/graphs/`)
- `graph_variants.csv` ledger with statistics for all variants
- **3 rows in `results.csv`** from test runs on the clean graph only:
  - SEC-GFD (seed 0): AUC = 0.839, AP = 0.526, F1 = 0.710 — took **480 seconds**
  - PMP (seed 0): AUC = 0.762, AP = 0.369, F1 = 0.652 — took **96 seconds**
  - SEC-GFD (seed 0, 2nd run): AUC = 0.442, AP = 0.124, F1 = 0.474 — took **8 seconds** (likely a debug/interrupted run)

---

## 4. Where Development Was Left Off

Development stopped on **2026-02-13** after completing the code for all 6 pipeline stages (TODO-06). The situation at that point:

1. **All code is written and committed** — the full pipeline from data loading through plot generation is functional.
2. **Only sanity-check runs were completed** — a few single-seed runs on the clean graph to verify PMP and SEC-GFD work.
3. **The full experiment matrix was never executed** — the `--stage matrix` command exists but was not run across all 62 variants and 5 training seeds.
4. **No baselines results exist** — MLP and GraphSAGE were never run (0 rows in results.csv for these models).
5. **No plots were generated** — the plots stage needs a populated results.csv to work with.

The immediate next action when resuming is to run the full matrix, then generate plots, then write the final report.

---

## 5. Training Speed Bottleneck (Known Issue)

**This was flagged as a concern before the break, and it is still relevant.** Here is what the data shows:

### Timing Per Single Run (1 model, 1 variant, 1 seed)

| Model | Time per Run | Training Style | Bottleneck |
|-------|-------------|---------------|-----------|
| MLP | ~1–2 sec (est.) | Full-graph, features only | Not a concern |
| GraphSAGE | ~5–10 sec (est.) | Full-graph, 2-layer | Not a concern |
| **PMP** | **~96 seconds** | Mini-batch with `MultiLayerFullNeighborSampler` | **Slow — see below** |
| **SEC-GFD** | **~480 seconds** | Full-graph, spectral decomposition | **Very slow** |

### Why PMP Is Slow

- Uses `MultiLayerFullNeighborSampler` — every training batch samples **all neighbors** for each node, which on a graph with mean degree 175 is extremely expensive.
- The YelpChi graph has ~8M edges and high average degree. Full neighbor sampling means each mini-batch reads a large subgraph.
- Runs on **CPU only** (no CUDA available in the current environment), so there is no GPU acceleration.
- The PMP config (`Repos/PMP-master/config/yelp.yml`) sets `full_neighbors: true`, which is the most expensive sampling mode.

### Why SEC-GFD Is Slow

- Full-graph forward pass on every epoch (no mini-batching) — the entire 45K-node, 8M-edge graph goes through the model each iteration.
- Includes spectral decomposition (`order_d=2`, `high_order=2`) which adds computation.
- The 480-second run suggests ~4.8 sec/epoch for 100 epochs, but this is on CPU with a large dense graph.

### Full Matrix Time Estimate

For the complete experiment (62 variants x 5 seeds x 4 models):

| Model | Runs | Est. Time per Run | Est. Total |
|-------|------|-------------------|-----------|
| MLP | 310 | ~2 sec | ~10 min |
| GraphSAGE | 310 | ~8 sec | ~40 min |
| PMP | 310 | ~96 sec | **~8.3 hours** |
| SEC-GFD | 310 | ~480 sec | **~41 hours** |
| **Total** | **1,240** | — | **~50 hours** |

This is the core bottleneck: **running the full matrix on CPU is expected to take roughly 2 full days of continuous computation.**

### Possible Mitigations

1. **Reduce training seeds** — use `--max-training-seeds 3` instead of 5 (saves 40% time)
2. **Reduce epochs** — use `--max-epochs 50 --patience 5` (models may converge faster)
3. **Use GPU** — if a CUDA-capable machine is available, pass `--device cuda` (would dramatically speed up both PMP and SEC-GFD)
4. **Run models in parallel** — run baselines, PMP, and SEC-GFD as separate processes on different machines
5. **Use neighbor sampling for PMP** — change `full_neighbors` to `false` in the PMP YAML config and set reasonable fanouts (e.g., `[25, 10]`)
6. **Skip redundant severity=0.0 variants** — these are no-ops that point to the base graph (already handled by `--include-noop` flag)

---

## 6. Current Issues Summary

| Issue | Severity | Notes |
|-------|----------|-------|
| Full matrix not yet run | **High** | No results to analyze or plot |
| Training is very slow on CPU | **High** | ~50 hours estimated for full matrix |
| No baseline results | **Medium** | MLP/GraphSAGE never executed |
| SEC-GFD has inconsistent results | **Low** | Two runs on same graph gave AUC 0.84 vs 0.44 — likely one was interrupted or used different params |
| CARE-GNN and GAGA not integrated | **Low** | Code exists in Repos/ but no benchmark integration (marked as optional in TODO-08) |
| ~11 GB of cached graphs on disk | **Low** | Large but expected for this dataset |

---

## 7. What Still Needs to Be Done

### Must Do (for project completion)

1. **Run the full experiment matrix**
   ```
   py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage matrix --force
   ```
   Consider using `--max-training-seeds 3` and `--max-epochs 50 --patience 5` to reduce total runtime.

2. **Generate plots**
   ```
   py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage plots
   ```
   This reads `results.csv` and produces metric-vs-severity curves, performance drop tables, and robustness scores.

3. **Write the final report** (TODO-07)
   - Methodology, results tables, figures, discussion
   - Must disclose oracle perturbations
   - Every figure traceable to `results.csv`

### Nice to Have (TODO-08, optional)

- Integrate CARE-GNN and/or GAGA models
- Test on additional datasets (Amazon fraud dataset)
- Add more perturbation types or severity levels

---

## 8. Project Structure Quick Reference

```
fraud-detection-robustness-benchmark/
├── benchmark/            # Core Python package (runner, models, metrics, stages)
│   ├── run.py            # CLI entry point
│   ├── data.py           # Dataset loading
│   ├── scenarios.py      # Graph perturbations
│   ├── baselines.py      # MLP + GraphSAGE definitions
│   ├── baselines_stage.py
│   ├── pmp_stage.py      # PMP integration
│   ├── secgfd_stage.py   # SEC-GFD integration
│   ├── matrix_stage.py   # Full matrix launcher
│   ├── plots_stage.py    # Visualization
│   ├── metrics.py        # ROC-AUC, AP, F1 (no sklearn)
│   ├── results.py        # CSV schema
│   └── summarize.py      # Mean/std aggregation
├── configs/
│   └── exp_yelpchi_v1.json   # Frozen experiment definition (single source of truth)
├── Repos/                # Vendored research code
│   ├── PMP-master/       # ICLR 2024
│   └── SEC-GFD-main/     # AAAI 2024
├── todos/                # Task breakdown (TODO-00 through TODO-08)
├── runs/                 # Experiment outputs (~11 GB)
│   └── gfd_robustness_benchmark_v1/
│       ├── graphs/       # Cached graph binaries
│       ├── results.csv   # Results table (3 rows so far)
│       └── graph_variants.csv  # Variant ledger (62 rows)
├── docs/                 # This document
├── README.md             # Usage instructions and command reference
├── AGENTS.md             # Working agreements and quick start
├── FINAL_PROJECT_GUIDE.md
└── Paper_Summary.md      # Literature review
```

---

## 9. Key Commands

```bash
# Generate graph variants (already done, ~11 GB)
py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage graphs

# Run full matrix (THE MAIN REMAINING TASK)
py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage matrix --force

# Run with reduced scope for faster iteration
py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage matrix --force \
    --max-training-seeds 3 --max-epochs 50 --patience 5

# Quick sanity check on clean graph only
py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage matrix --only-clean

# Generate plots (after matrix completes)
py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage plots
```
