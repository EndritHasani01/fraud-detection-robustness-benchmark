# Running The Benchmark On Google Colab T4

This guide explains how to run the current benchmark on a Google Colab GPU runtime with an NVIDIA T4, typically about 15 GiB VRAM.

The repo is a static robustness benchmark for graph-based fraud detection on YelpChi. It loads YelpChi through `dgl.data.FraudDataset`, builds deterministic graph variants, evaluates `mlp`, `sage`, `pmp`, and `secgfd`, writes one `results.csv`, and generates report-ready CSVs and plots.

Use the current v3 configs unless you intentionally need a historical v2 rerun:

- `configs/exp_yelpchi_v3_fast.json`: fast smoke test
- `configs/exp_yelpchi_v3.json`: main benchmark
- `configs/exp_yelpchi_v3_full.json`: larger severity/seed run

Colab uses Linux shell commands, so use `python ...`, not Windows `py ...`.

## 1. Start The Right Colab Runtime

In Colab:

1. Open `Runtime > Change runtime type`.
2. Select `T4 GPU`.
3. Save.

Then verify the GPU:

```python
!nvidia-smi
```

You should see a T4. If `torch.cuda.is_available()` is false later, the runtime is not using a GPU.

## 2. Clone Or Upload The Repo

If the repo is on GitHub:

```python
%cd /content
!git clone https://github.com/<YOUR_USER_OR_ORG>/fraud-detection-robustness-benchmark.git
%cd /content/fraud-detection-robustness-benchmark
```

If you already uploaded or copied the project into Colab, just `cd` into it:

```python
%cd /content/fraud-detection-robustness-benchmark
```

Optional, but recommended for long runs: mount Google Drive so results survive runtime disconnects.

```python
from google.colab import drive
drive.mount("/content/drive")
```

Choose a persistent output directory:

```python
import os
OUT = "/content/drive/MyDrive/fraud-benchmark-runs/gfd_robustness_benchmark_v3"
os.environ["OUT"] = OUT
```

For quick tests, local output is faster but disappears when the VM is reset:

```python
import os
OUT = "runs/gfd_robustness_benchmark_v3"
os.environ["OUT"] = OUT
```

## 3. Install A Colab-Compatible GPU Stack

The main compatibility issue is DGL. The Windows README pins a Windows DGL wheel, but Colab needs Linux CUDA wheels. The safest Colab stack for this repo is:

- Python 3.11
- PyTorch CUDA 11.8 wheel
- DGL `1.1.3+cu118`
- `numpy<2`

DGL's wheel index includes Linux `dgl-1.1.3+cu118` wheels for CPython 3.11, while newer DGL CUDA wheel availability is uneven across Linux/Python combinations. PyTorch's official previous-version page provides CUDA 11.8 install commands for PyTorch 2.1.x. Relevant references:

- DGL install docs: https://www.dgl.ai/dgl_docs/install/index.html
- DGL CUDA 11.8 wheel index: https://data.dgl.ai/wheels/cu118/repo.html
- PyTorch previous versions: https://pytorch.org/get-started/previous-versions/

Run this in a fresh Colab runtime:

```python
import sys
print(sys.version)
assert sys.version_info[:2] == (3, 11), "This guide assumes Colab Python 3.11."
```

Install packages:

```python
!pip uninstall -y dgl torch torchdata torchvision torchaudio
!pip install -q "numpy<2" scipy sympy PyYAML pydantic matplotlib tqdm scikit-learn pandas
!pip install -q torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
!pip install -q "dgl==1.1.3+cu118" -f https://data.dgl.ai/wheels/cu118/repo.html
```

Restart the runtime after installing the stack:

```python
import os
os.kill(os.getpid(), 9)
```

After the restart, `cd` back into the repo and remount Drive if needed:

```python
%cd /content/fraud-detection-robustness-benchmark
from google.colab import drive
drive.mount("/content/drive")
import os
OUT = "/content/drive/MyDrive/fraud-benchmark-runs/gfd_robustness_benchmark_v3"
os.environ["OUT"] = OUT
```

Verify imports and CUDA:

```python
import torch, dgl, numpy as np
from dgl.data.fraud import FraudDataset

print("torch:", torch.__version__)
print("dgl:", dgl.__version__)
print("numpy:", np.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

g = dgl.rand_graph(10, 20).to("cuda")
print("DGL CUDA graph nodes:", g.num_nodes())
```

## 4. Run A Fast End-To-End Smoke Test

Run this before the main benchmark. It checks dataset download, graph generation, training, result writing, and plotting.

```python
!python -m benchmark.run \
  --config configs/exp_yelpchi_v3_fast.json \
  --stage graphs \
  --out "$OUT" \
  --force
```

```python
!python -m benchmark.run \
  --config configs/exp_yelpchi_v3_fast.json \
  --stage baselines \
  --out "$OUT" \
  --device cuda \
  --max-epochs 5 \
  --patience 2 \
  --force
```

```python
!python -m benchmark.run \
  --config configs/exp_yelpchi_v3_fast.json \
  --stage plots \
  --out "$OUT"
```

Check the output files:

```python
!find "$OUT" -maxdepth 3 -type f | sort | head -80
```

## 5. Main V3 Benchmark On T4

The full v3 matrix is resumable. Completed run keys in `results.csv` are skipped by default, so run models in chunks and re-run the same commands after a Colab disconnect.

First build graphs:

```python
!python -m benchmark.run \
  --config configs/exp_yelpchi_v3.json \
  --stage graphs \
  --out "$OUT" \
  --force
```

Run baseline models:

```python
!python -m benchmark.run \
  --config configs/exp_yelpchi_v3.json \
  --stage matrix \
  --models mlp,sage \
  --out "$OUT" \
  --device cuda
```

Run PMP:

```python
!python -m benchmark.run \
  --config configs/exp_yelpchi_v3.json \
  --stage matrix \
  --models pmp \
  --out "$OUT" \
  --device cuda
```

Run SEC-GFD:

```python
!python -m benchmark.run \
  --config configs/exp_yelpchi_v3.json \
  --stage matrix \
  --models secgfd \
  --out "$OUT" \
  --device cuda
```

Then run the clean-train shift protocol. This writes rows with `protocol=train_clean_eval_all`.

```python
!python -m benchmark.run \
  --config configs/exp_yelpchi_v3.json \
  --stage matrix \
  --protocol train_clean_eval_all \
  --models mlp,sage \
  --out "$OUT" \
  --device cuda
```

```python
!python -m benchmark.run \
  --config configs/exp_yelpchi_v3.json \
  --stage matrix \
  --protocol train_clean_eval_all \
  --models pmp \
  --out "$OUT" \
  --device cuda
```

```python
!python -m benchmark.run \
  --config configs/exp_yelpchi_v3.json \
  --stage matrix \
  --protocol train_clean_eval_all \
  --models secgfd \
  --out "$OUT" \
  --device cuda
```

Generate report artifacts:

```python
!python -m benchmark.run \
  --config configs/exp_yelpchi_v3.json \
  --stage plots \
  --out "$OUT" \
  --ci
```

## 6. T4 Runtime Controls

Use these controls when Colab time or VRAM is tight.

Run only a few variants:

```python
!python -m benchmark.run \
  --config configs/exp_yelpchi_v3.json \
  --stage matrix \
  --models mlp,sage \
  --out "$OUT" \
  --device cuda \
  --max-variants 5 \
  --max-training-seeds 1
```

Shorten training:

```python
!python -m benchmark.run \
  --config configs/exp_yelpchi_v3.json \
  --stage matrix \
  --models pmp \
  --out "$OUT" \
  --device cuda \
  --max-epochs 30 \
  --patience 5
```

Reduce SEC-GFD cost:

```python
!python -m benchmark.run \
  --config configs/exp_yelpchi_v3.json \
  --stage matrix \
  --models secgfd \
  --out "$OUT" \
  --device cuda \
  --max-epochs 30 \
  --patience 5 \
  --secgfd-hid-dim 16 \
  --secgfd-order-d 1 \
  --secgfd-high-order 1
```

Important: if you use runtime-reduction flags for final results, disclose them. For report-facing results, prefer the frozen config defaults unless Colab constraints make that impossible.

## 7. Monitor Completeness

After any stage, inspect `results.csv`:

```python
import pandas as pd
from pathlib import Path

out = Path(OUT)
results = pd.read_csv(out / "results.csv")
display(results["status"].value_counts(dropna=False))
display(results.groupby(["protocol", "model_id", "status"]).size().reset_index(name="n"))
```

After the plots stage:

```python
missing = pd.read_csv(Path(OUT) / "plots" / "missing_or_error_runs.csv")
display(missing.head(20))
print("missing/error rows:", len(missing))
```

If rows are missing because Colab disconnected, re-run the same matrix command. The runner skips completed keys by default.

If rows have `status=error`, retry only error keys:

```python
!python -m benchmark.run \
  --config configs/exp_yelpchi_v3.json \
  --stage matrix \
  --models secgfd \
  --out "$OUT" \
  --device cuda \
  --retry-errors
```

## 8. Download Or Save Artifacts

If `OUT` is on Google Drive, the artifacts are already persistent. The main files are:

- `config.json`
- `graph_variants.csv`
- `variant_audit.csv`
- `results.csv`
- `results_summary_baselines.csv`
- `results_summary_pmp.csv`
- `results_summary_secgfd.csv`
- `results_summary_shift.csv`
- `plots/summary_curves.csv`
- `plots/performance_drop_max_stress.csv`
- `plots/robustness_scores.csv`
- `plots/audit_curves.csv`
- `plots/performance_audit_join.csv`
- `plots/missing_or_error_runs.csv`
- `plots/<protocol>/<dataset>/<split>/*.png`

To download a zip from Colab:

```python
!zip -r /content/gfd_robustness_benchmark_v3_outputs.zip "$OUT"
from google.colab import files
files.download("/content/gfd_robustness_benchmark_v3_outputs.zip")
```

## 9. Troubleshooting

`ImportError: Cannot load GraphBolt C++ library`

You likely installed DGL 2.x with a mismatched PyTorch version. Reinstall the stack in section 3, restart the runtime, and verify `dgl.__version__ == "1.1.3"`.

`torch.cuda.is_available()` is false

Change the runtime type to T4 GPU and restart. Installing CUDA packages alone does not enable a GPU runtime.

`No module named dgl`

The runtime restarted after install, or the install cell did not finish. Re-run the install cells, restart, then verify imports.

`DGL graph cannot move to cuda`

You installed CPU DGL. Reinstall `dgl==1.1.3+cu118` from `https://data.dgl.ai/wheels/cu118/repo.html`.

Out-of-memory during SEC-GFD

Retry with smaller SEC-GFD controls:

```python
--secgfd-hid-dim 16 --secgfd-order-d 1 --secgfd-high-order 1 --max-epochs 30 --patience 5
```

Colab disconnects

Use a Drive-backed `--out` path and re-run the same command. The benchmark is resumable through `results.csv`.

Python version is not 3.11

Do not blindly mix DGL wheels. This guide assumes Python 3.11 because the pinned DGL CUDA wheel is available for CPython 3.11. If Colab changes its default Python, check the DGL wheel index for a matching `cp3xx` Linux CUDA wheel before installing.

## 10. Legacy V2 Run

If you specifically need the v2 benchmark from the older working agreement, use the same Colab environment and swap the config:

```python
!python -m benchmark.run --config configs/exp_yelpchi_v2_fast.json --stage graphs --out "$OUT" --force
!python -m benchmark.run --config configs/exp_yelpchi_v2_fast.json --stage baselines --out "$OUT" --device cuda --force
```

For the main v2 matrix:

```python
!python -m benchmark.run --config configs/exp_yelpchi_v2.json --stage graphs --out "$OUT" --force
!python -m benchmark.run --config configs/exp_yelpchi_v2.json --stage matrix --out "$OUT" --device cuda
!python -m benchmark.run --config configs/exp_yelpchi_v2.json --stage matrix --protocol train_clean_eval_all --out "$OUT" --device cuda
!python -m benchmark.run --config configs/exp_yelpchi_v2.json --stage plots --out "$OUT" --ci
```

Use a different `OUT` directory for v2 and v3 so their `results.csv` files do not mix.
