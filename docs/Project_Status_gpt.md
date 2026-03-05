# Fraud Detection Robustness Benchmark — Project Status

Last updated: **2026-03-04**

## 1) What the project is (purpose + high-level overview)

This repository is a **course final project**: a small, reproducible **robustness benchmark** for **graph-based fraud/anomaly detection** (node classification).

Instead of reporting performance on only one “clean” dataset, the benchmark creates **controlled stress tests** of the graph (e.g., increased heterophily, feature camouflage, random edge noise) and measures how different models’ performance **degrades as stress severity increases**.

## 2) What it does (core functionality + components)

At a high level, the benchmark is a pipeline with **cached graph variants** and **repeatable training/evaluation**:

### Core pipeline stages (CLI)

The entry point is `benchmark/run.py` (invoked via `py -m benchmark.run`), with stages:

- `graphs`: loads the dataset, creates a fixed split, generates and caches stress-test variants, writes `graph_variants.csv`
- `baselines`: trains/evaluates **MLP** + **GraphSAGE** on cached graphs, appends rows to `results.csv`
- `pmp`: trains/evaluates **PMP (LA-SAGE-S)** from `Repos/PMP-master/`, appends rows to `results.csv`
- `secgfd`: trains/evaluates **SEC-GFD** from `Repos/SEC-GFD-main/`, appends rows to `results.csv`
- `matrix`: convenience launcher that runs `baselines` → `pmp` → `secgfd`
- `plots`: generates plot-ready summaries and figures from `results.csv`

### Key modules (where the logic lives)

- Experiment config (frozen): `configs/exp_yelpchi_v1.json`
- Dataset loading + split masks: `benchmark/data.py`
- Stress tests (perturbations): `benchmark/scenarios.py`
- Graph caching and paths: `benchmark/cache.py`, `benchmark/paths.py`
- Graph statistics ledger: `benchmark/stats.py` → `graph_variants.csv`
- Models:
  - Baselines: `benchmark/baselines.py`, trained in `benchmark/baselines_stage.py`
  - PMP adapter: `benchmark/pmp_stage.py` (imports from `Repos/PMP-master/`)
  - SEC-GFD adapter: `benchmark/secgfd_stage.py` (imports from `Repos/SEC-GFD-main/`)
- Unified results schema: `benchmark/results.py` → `results.csv`
- Plotting and summaries: `benchmark/plots_stage.py`, `benchmark/summarize.py`

## 3) What has already been implemented or completed

Implementation-wise, the repo is in a “pipeline is built” state:

- ✅ Frozen experiment config for YelpChi: `configs/exp_yelpchi_v1.json`
- ✅ Dataset load + deterministic split masks (train/val/test): `benchmark/data.py`
- ✅ Stress-test generation (heterophily rewire, feature camouflage, random edge noise): `benchmark/scenarios.py`
- ✅ Caching of base graph + variants to disk and a variant ledger CSV: `benchmark/run.py` (stage `graphs`)
- ✅ Baselines (MLP + GraphSAGE) training/evaluation and `results.csv` writing: `benchmark/baselines_stage.py`
- ✅ Integration adapters for two “repo models”:
  - PMP (ICLR 2024): `benchmark/pmp_stage.py` → `Repos/PMP-master/`
  - SEC-GFD (AAAI 2024): `benchmark/secgfd_stage.py` → `Repos/SEC-GFD-main/`
- ✅ A “matrix” stage to launch all integrated models: `benchmark/matrix_stage.py`
- ✅ Plotting stage to produce curves/tables from `results.csv`: `benchmark/plots_stage.py`

Project-planning TODOs are marked “Implemented” through `todos/TODO_06_experiments_and_plots.md`. The remaining work is primarily **running the full experiment matrix** and writing the **final report** (`todos/TODO_07_final_report.md`).

## 4) Where development was last left off (observable state in this repo)

There is an existing run directory at `runs/gfd_robustness_benchmark_v1/`:

- `graph_variants.csv` exists (last modified **2026-02-13**), indicating `--stage graphs` was run successfully.
  - Total cached rows: **61**
  - Rows that would be used by default evaluation (clean + applied variants; excludes “no-op” variants): **46**
- `results.csv` exists (last modified **2026-02-14**) but contains only **3 successful runs** (`status=ok`), meaning the full matrix was **not** executed.
  - Observed: PMP ran once on the clean graph; SEC-GFD ran twice on the clean graph (same key repeated).
- Plot outputs do **not** exist yet (no `runs/gfd_robustness_benchmark_v1/plots/`), so `--stage plots` was not run on a complete results file.

There is also an untracked snapshot `old-stats/results.csv` containing many baseline rows, which looks like an earlier/alternative results capture (useful reference, but not part of the current `runs/<exp>/` pipeline output).

## 5) Current issues or bottlenecks

### A) Training time (still a major bottleneck)

The “repo models” are computationally heavy on YelpChi’s **~8.05M edges** (homogeneous view). This is still relevant based on existing artifacts:

- `runs/gfd_robustness_benchmark_v1/results.csv` contains a SEC-GFD clean run with `duration_sec ≈ 479.9s` (~8 minutes) on CPU.
- The PMP clean run in the same file has `duration_sec ≈ 95.6s` (~1.6 minutes).
- Baseline reference from `old-stats/results.csv` (clean graph examples):
  - MLP: typically **seconds to tens of seconds**
  - GraphSAGE: typically **tens of seconds**

Why this happens (high-level):

- **SEC-GFD is full-graph** and performs multiple DGL `update_all` passes per forward (see `Repos/SEC-GFD-main/model/SECGFD.py`), so each epoch is expensive on an 8M-edge graph.
- **PMP uses DGL mini-batching**, but the default PMP config often uses **full-neighbor sampling** (`full_neighbors: true` in PMP YAML), which can still be very large per batch.
- On **Windows**, the recommended setup in `README.md` installs a **CPU-only DGL wheel**, so even if `--device cuda` is passed, stages may fall back to CPU.

Runtime implications for the *default config* (order-of-magnitude):

- Default evaluation uses ~**46 graph variants** × **5 training seeds** = **230 trainings per model**
- With 4 models (`mlp`, `sage`, `pmp`, `secgfd`), that’s ~**920 total training runs**
- Using the observed per-run times above, a full CPU run can easily reach **~40+ hours** end-to-end (dominated by SEC-GFD).

### B) Results file is append-only (duplicates are easy)

Stages append to `runs/<exp>/results.csv` without de-duplication. If a stage is re-run without `--force` (or without cleaning the output directory), it can create **duplicate rows** for the same (dataset/split/scenario/severity/seeds/model) key. The current `results.csv` already shows this for SEC-GFD on the clean graph.

### C) Environment/dependency friction (repo models)

The integrated “repo models” bring extra dependencies and assumptions (e.g., SEC-GFD imports `sympy` and `scipy`). The benchmark code tries to be robust, but Windows ML stacks can still be fragile. The repo’s `README.md` recommends Python 3.10/3.11 and pinned CPU wheels for stability.

## 6) What still needs to be done / next steps

### Immediate “get back into it” checklist

1. Decide your compute strategy:
   - If you have access to a GPU (WSL2/Linux/Colab), consider running **PMP + SEC-GFD** there to reduce runtime.
   - If staying on CPU/Windows, plan to **reduce the matrix** (fewer variants/seeds/epochs).
2. Re-run a small sanity slice to confirm everything still works:
   - `py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage baselines --only-clean --max-training-seeds 1`
   - `py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage pmp --only-clean --max-training-seeds 1 --max-epochs 20 --patience 5`
   - `py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage secgfd --only-clean --max-training-seeds 1 --max-epochs 20 --patience 5`
3. Run a reduced experiment matrix first, then scale up:
   - Use `--max-variants N`, `--max-training-seeds N`, `--max-epochs N`, and `--patience N` to control runtime.
4. Once `results.csv` is complete for your chosen matrix:
   - Run `py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage plots`
5. Write the final report using `Paper_Summary.md` + `FINAL_PROJECT_GUIDE.md`:
   - See `todos/TODO_07_final_report.md` for the report checklist.

### Optional extensions (only if time remains)

`Repos/CARE-GNN-master/` and `Repos/GAGA-master/` are present but not integrated into the unified runner. If you want additional insights, see `todos/TODO_08_optional_extensions.md`.

## 7) Notes for future-you (practical tips)

- Prefer starting a new run directory (or use `--force`) when re-running stages to avoid duplicate `results.csv` rows.
- If SEC-GFD remains too slow:
  - reduce `--max-epochs` aggressively and rely on early stopping (`--patience`)
  - reduce the evaluated matrix size (variants/seeds) to get plot-ready curves quickly
  - consider lowering SEC-GFD model complexity (this would require changing `benchmark/secgfd_stage.py`, which is currently hard-coded to the repo defaults)

