# Fraud Detection Robustness Benchmark

This repo implements a small robustness benchmark for **graph-based fraud / anomaly detection**.

Core idea:
- Start from a real fraud graph dataset (YelpChi).
- Generate controlled "stress-test" graph variants (heterophily, camouflage, noise/density).
- Cache every graph variant to disk and log graph statistics to a single CSV.
- Later steps (TODO-03+) will train/evaluate models and write results to `results.csv`.

The benchmark runner is in `benchmark/` and the current frozen experiment definition is `configs/exp_yelpchi_v1.json`.

## Step-by-Step: Create Venv, Run Graph Generation, Explain Outputs

These steps reproduce the command:

```bash
py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage graphs --force
```

### 1. Create a Python 3.11 Virtual Environment

What this does:
- Creates an isolated environment under `.venv/` so the benchmark dependencies do not affect your global Python.

Command:

```bash
py -3.11 -m venv .venv
```

### 2. Activate the Virtual Environment

What this does:
- Puts `.venv`'s Python and `pip` first on your PATH so `py`/`python` refer to the venv interpreter.

PowerShell:

```powershell
# If your execution policy blocks scripts, bypass it for this shell only:
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force

. .\.venv\Scripts\Activate.ps1
```

CMD (alternative):

```bat
.venv\Scripts\activate.bat
```

### 3. Install Dependencies (DGL + PyTorch)

What this does:
- Installs the ML stack required by the `graphs` stage:
  - `torch` (tensor backend)
  - `dgl` (graph library + FraudDataset loader)
  - `numpy`/`scipy` (dataset loading + numerics)
  - `torchdata`, `PyYAML`, `pydantic` (required by DGL internals on Windows)

Important version notes (why these pins exist):
- `numpy<2` avoids binary-compatibility issues with compiled wheels (common for graph/ML libs).
- DGL 2.2.1 ships GraphBolt DLLs up to PyTorch 2.3.x, so using a much newer PyTorch can break DGL import.
- The CPU DGL wheel avoids CUDA runtime dependencies on machines without CUDA installed.

Commands (run inside the activated venv):

```bash
py -m pip install "numpy<2" scipy

# CPU-only PyTorch (keeps setup simple on Windows without CUDA):
py -m pip install torch==2.3.0+cpu --index-url https://download.pytorch.org/whl/cpu

# CPU DGL wheel for Python 3.11 on Windows:
py -m pip install https://data.dgl.ai/wheels/dgl-2.2.1-cp311-cp311-win_amd64.whl

# Required by DGL internals:
py -m pip install torchdata==0.8.0 PyYAML pydantic
```

### 4. Run the Graph Caching Stage

What this command does (high-level):
1. Loads **YelpChi** via DGL's `FraudDataset`.
2. Creates a **single fixed split** (train/val/test = 0.4/0.2/0.4) using the `split_seed` from the config.
3. Converts the source heterograph into a **homogeneous** graph (relations merged) so all models/perturbations share one canonical view.
4. Generates and caches stress-test graph variants for every:
   - scenario (heterophily/camouflage/noise),
   - severity level,
   - `graph_seed` (currently taken from the config's seed list).
5. Writes a ledger CSV (`graph_variants.csv`) containing per-variant graph statistics.

What this command does (ordered, detailed, matching the code path):
1. Reads and validates the JSON experiment config (`--config`).
2. Creates the output directory (default: `runs/<experiment_name>/`, or `--out` if provided).
3. Copies the config to `runs/<experiment_name>/config.json` for reproducibility.
4. Creates (or overwrites) CSV files:
   - `runs/<experiment_name>/results.csv` (schema only until training is implemented)
   - `runs/<experiment_name>/graph_variants.csv` (one row per cached variant + graph stats)
5. Loads the raw dataset via DGL into `runs/<experiment_name>/data/dgl/`.
   - If you see `Done loading data from cached files.`, it means DGL found the dataset already downloaded/cached.
6. Normalizes graph data fields:
   - ensures node features and labels have consistent dtypes
   - overwrites/creates `train_mask`, `val_mask`, `test_mask` deterministically from the config split seed
7. Converts to the canonical graph view (in this config: `dgl.to_homogeneous`) and saves the base graph:
   - `runs/<experiment_name>/graphs/base/<dataset>/<split_id>/graph.bin`
8. Appends one `graph_variants.csv` row for the base graph (with `scenario_id=clean`).
9. For every `(scenario, severity, graph_seed)` triple:
   - applies the perturbation to a copy of the base graph
   - writes `meta.json` describing the perturbation and proxy logs
   - saves the variant graph to `runs/<experiment_name>/graphs/variants/.../graph.bin`
   - for severity `0.0`, stores `graph_ref.json` pointing to the base graph instead of duplicating the binary
   - appends a row to `graph_variants.csv` containing stats for that variant

Notes about console output:
- DGL may print caching messages during dataset loading (this is normal).
- You may see a `torchdata` deprecation warning on Windows; it is safe to ignore for the `graphs` stage as long as imports succeed.

Run it:

```bash
py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage graphs --force
```

What `--force` means:
- Overwrites `runs/<experiment_name>/results.csv` and `runs/<experiment_name>/graph_variants.csv` headers.
- Overwrites cached graph files for the same paths.

### 5. Understand the Outputs

After the command finishes, outputs are under:

`runs/gfd_robustness_benchmark_v1/`

Key files/folders:
- `runs/gfd_robustness_benchmark_v1/config.json`
  - Snapshot of the config used for the run (reproducibility).
- `runs/gfd_robustness_benchmark_v1/data/dgl/`
  - DGL-downloaded dataset artifacts (zip + extracted cache).
- `runs/gfd_robustness_benchmark_v1/graphs/base/yelpchi/s0/graph.bin`
  - The base (clean) graph for split `s0`.
- `runs/gfd_robustness_benchmark_v1/graphs/variants/yelpchi/s0/<scenario>/<severity>/seed_<k>/`
  - Cached graph variants. Each directory contains:
    - `meta.json` (scenario parameters + proxy logs)
    - `graph.bin` for applied perturbations, or
    - `graph_ref.json` for no-op variants (severity 0.0) that point back to the base graph to avoid duplication.
- `runs/gfd_robustness_benchmark_v1/graph_variants.csv`
  - One row per cached variant with graph stats:
    - node/edge counts, degree summaries, heterophily ratio, label prevalence, etc.
- `runs/gfd_robustness_benchmark_v1/results.csv`
  - Results schema (metrics columns exist but stay empty until TODO-03+ integrates training/evaluation).

## Troubleshooting

If you see DGL import errors:
- "graphbolt_pytorch_*.dll not found":
  - Your `torch` version is too new for the installed `dgl` wheel. Use `torch==2.3.0+cpu` (as above) with `dgl==2.2.1`.
- "A module compiled using NumPy 1.x cannot be run in NumPy 2.x":
  - Downgrade NumPy: `py -m pip install "numpy<2"`.

## Next Steps

- `todos/TODO_03_baselines.md`: implement MLP + baseline GNN training and populate `results.csv`.
- `todos/TODO_04_integrate_pmp.md` and `todos/TODO_05_integrate_secgfd.md`: integrate the two specialized methods.
