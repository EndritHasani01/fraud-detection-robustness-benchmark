# Fraud Detection Robustness Benchmark

This repo implements a small robustness benchmark for **graph-based fraud / anomaly detection**.

Core idea:
- Start from a real fraud graph dataset (YelpChi).
- Generate controlled "stress-test" graph variants (heterophily, camouflage, noise/density).
- Cache every graph variant to disk and log graph statistics to a single CSV.
- Train/evaluate baseline models (MLP + GraphSAGE) and specialized methods (PMP, SEC-GFD) and write results to `results.csv`.
- Later steps (TODO-06+) run the full grid and produce plots/report artifacts.

The benchmark runner is in `benchmark/` and the current frozen experiment definition is `configs/exp_yelpchi_v1.json`.

## Run Commands (Complete List)

All commands below assume you are in the repository root:
`d:\Revolucion\1. Fakulltet\4.1 Semestri\6. Theoretical Graphs\GitHub\fraud-detection-robustness-benchmark`

### Environment Commands

| Command | What it does |
|---|---|
| `py -3.11 -m venv .venv` | Creates a Python 3.11 virtual environment in `.venv/`. |
| `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force` | PowerShell-only: allows running the activation script in the current shell session. |
| `. .\.venv\Scripts\Activate.ps1` | Activates the venv for PowerShell (uses the venv's Python and pip). |
| `.venv\Scripts\activate.bat` | Activates the venv for CMD (alternative to PowerShell). |
| `deactivate` | Deactivates the venv (returns to your global shell environment). |

### Dependency Install Commands (Run Inside The Activated Venv)

| Command | What it does |
|---|---|
| `py -m pip install "numpy<2" scipy` | Installs base numeric dependencies and pins NumPy below 2.0 for wheel compatibility. |
| `py -m pip install torch==2.3.0+cpu --index-url https://download.pytorch.org/whl/cpu` | Installs CPU-only PyTorch (keeps setup simple on Windows without CUDA). |
| `py -m pip install https://data.dgl.ai/wheels/dgl-2.2.1-cp311-cp311-win_amd64.whl` | Installs DGL 2.2.1 CPU wheel for Python 3.11 on Windows. |
| `py -m pip install torchdata==0.8.0 PyYAML pydantic` | Installs packages required by DGL internals on Windows. |
| `py -c "import torch, dgl; print(torch.__version__); print(dgl.__version__)"` | Quick import check to confirm the ML stack loads correctly. |

### Benchmark Runner Commands

| Command | What it does |
|---|---|
| `py -m benchmark.run --help` | Shows all available stages and CLI flags. |
| `py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage graphs` | Builds the fixed split and caches the base graph + all stress-test variants defined in the config. |
| `py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage graphs --force` | Same as above, but overwrites cached graph outputs and rewrites CSV headers. |
| `py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage graphs --force-reload` | Forces DGL to reload/redownload the dataset artifacts. |
| `py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage graphs --out runs\\my_run` | Writes outputs to a custom directory instead of `runs/<experiment_name>/`. |
| `py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage baselines` | Trains/evaluates `mlp` + `sage` on cached graphs and appends rows to `runs/<experiment>/results.csv`. Also writes `results_summary_baselines.csv` (mean/std across training seeds). Requires that `--stage graphs` already ran in the same output directory. |
| `py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage baselines --only-clean` | Baselines on the clean graph only (fast sanity check). |
| `py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage baselines --include-noop` | Also evaluates no-op scenario rows (severity 0.0). Normally you do not need this because the `clean` row already covers the unperturbed graph. |
| `py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage baselines --force` | Overwrites `runs/<experiment>/results.csv` before writing new baseline results (useful for a clean rerun). |
| `py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage baselines --max-variants 10` | Evaluates at most N variant rows from `graph_variants.csv` (useful for quick smoke tests). |
| `py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage baselines --max-training-seeds 1` | Uses only the first N training seeds from the config (useful to reduce runtime). |
| `py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage baselines --max-epochs 50 --patience 5` | Overrides baseline training loop settings (max epochs and early stopping patience). |
| `py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage baselines --device cpu` | Runs baselines on CPU (recommended for this repo's pinned CPU DGL wheel). |
| `py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage baselines --device cuda` | Attempts CUDA training; if the environment/graph backend does not support it, the code falls back to CPU. |
| `py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage pmp` | Trains/evaluates PMP (LA-SAGE-S) on cached graphs and appends rows to `runs/<experiment>/results.csv` (with `model_id=pmp`). Also writes `results_summary_pmp.csv` (mean/std across training seeds). Requires that `--stage graphs` already ran in the same output directory. |
| `py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage pmp --only-clean` | PMP on the clean graph only (fast sanity check). |
| `py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage pmp --max-training-seeds 1 --max-epochs 20 --patience 5` | PMP quick-run settings to reduce runtime while verifying the pipeline. |
| `py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage secgfd` | Trains/evaluates SEC-GFD on cached graphs and appends rows to `runs/<experiment>/results.csv` (with `model_id=secgfd`). Also writes `results_summary_secgfd.csv` (mean/std across training seeds). Requires that `--stage graphs` already ran in the same output directory. |
| `py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage secgfd --only-clean` | SEC-GFD on the clean graph only (fast sanity check). |
| `py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage secgfd --max-training-seeds 1 --max-epochs 20 --patience 5` | SEC-GFD quick-run settings to reduce runtime while verifying the pipeline. |

Notes:
- If you use `--out` for `--stage graphs`, you must also pass the same `--out` for `--stage baselines` so it can find `graph_variants.csv` and the cached graphs.
- Same applies to `--stage pmp` (use the same `--out` so it can find the cached graphs and variant ledger).
- Same applies to `--stage secgfd`.
- Running `--stage graphs --force` rewrites `results.csv` and will delete any previously-written baseline results. If you do that, re-run `--stage baselines`.
- If you run `--stage pmp --force`, it rewrites `results.csv` and will delete previously-written rows (including baselines). Re-run baselines afterward if needed.
- If you run `--stage secgfd --force`, it rewrites `results.csv` and will delete previously-written rows (including baselines and PMP). Re-run other stages afterward if needed.

### Output Inspection Commands (Optional)

| Command | What it does |
|---|---|
| `Get-ChildItem runs\\gfd_robustness_benchmark_v1 -Force` | Lists the top-level outputs for the frozen experiment. |
| `Get-Content runs\\gfd_robustness_benchmark_v1\\graph_variants.csv -TotalCount 5` | Prints the CSV header and first few variant rows (graph stats ledger). |
| `Get-Content runs\\gfd_robustness_benchmark_v1\\results.csv -TotalCount 5` | Prints the CSV header and first few results rows (populated after `--stage baselines` / `--stage pmp` / `--stage secgfd`). |
| `Get-Content runs\\gfd_robustness_benchmark_v1\\results_summary_baselines.csv -TotalCount 5` | Prints the baseline mean/std summary across training seeds. |
| `Get-Content runs\\gfd_robustness_benchmark_v1\\results_summary_pmp.csv -TotalCount 5` | Prints the PMP mean/std summary across training seeds. |
| `Get-Content runs\\gfd_robustness_benchmark_v1\\results_summary_secgfd.csv -TotalCount 5` | Prints the SEC-GFD mean/std summary across training seeds. |

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
   - `runs/<experiment_name>/results.csv` (header; populated by `--stage baselines` and later stages)
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
  - Per-run metrics rows (baselines + PMP + SEC-GFD append to this file).
- `runs/gfd_robustness_benchmark_v1/results_summary_baselines.csv`
  - Mean/std across training seeds for the baselines.
- `runs/gfd_robustness_benchmark_v1/results_summary_pmp.csv`
  - Mean/std across training seeds for PMP.
- `runs/gfd_robustness_benchmark_v1/results_summary_secgfd.csv`
  - Mean/std across training seeds for SEC-GFD.

## Troubleshooting

If you see DGL import errors:
- "graphbolt_pytorch_*.dll not found":
  - Your `torch` version is too new for the installed `dgl` wheel. Use `torch==2.3.0+cpu` (as above) with `dgl==2.2.1`.
- "A module compiled using NumPy 1.x cannot be run in NumPy 2.x":
  - Downgrade NumPy: `py -m pip install "numpy<2"`.

## Next Steps

- `todos/TODO_06_experiments_and_plots.md`: run the full grid and produce plots for the report.

## Baselines (MLP + GraphSAGE)

Once you have cached graphs, you can train/evaluate the baselines and write rows into `runs/<experiment>/results.csv`:

```bash
py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage baselines
```

Sanity-check on only the clean graph:

```bash
py -m benchmark.run --config configs/exp_yelpchi_v1.json --stage baselines --only-clean
```

After the run, a mean/std summary across training seeds is written to:
- `runs/gfd_robustness_benchmark_v1/results_summary_baselines.csv`
