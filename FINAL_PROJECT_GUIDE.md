# Final Project Guide: Fraud Detection Robustness Benchmark

This document explains a concrete, finishable implementation plan for your final project:

> Build a small robustness benchmark for graph-based fraud detection and evaluate how model performance degrades as the graph becomes more disordered (heterophily, camouflage, noise/density), with repeated seeds (and splits if feasible).

It is designed to build directly on your Task 1 write-up in `Paper_Summary.md`.

---

## 1. Project Goal (What You Are Solving)

**Task framing (node classification).**
Given a graph (transactions/reviews/users as nodes, relationships as edges), predict which nodes are fraudulent/anomalous.

**Robustness framing (benchmark).**
Instead of only reporting accuracy on one fixed dataset, you will:

1. Create **controlled stress-test transformations** of the graph (increasing heterophily, camouflage, and noise/density).
2. Run several models on the same datasets under increasing stress.
3. Report how performance changes with stress severity, including variance across random seeds (and optionally splits).

---

## 2. What You Will Deliver (Submission Checklist)

Your final submission should include these 3 deliverables:

1. **Benchmark code** (data loading + perturbations + experiment runner).
2. **Results artifacts**
   - `results.csv` (all runs, one row per model x dataset x scenario x severity x seed[/split])
   - 3-6 plots (performance vs severity curves, with error bars)
   - a short table summarizing clean vs stressed performance (mean +/- std)
3. **Final report** (8-12 pages is typical, adjust to course expectations)
   - methodology, stress-test definitions, evaluation protocol, results, discussion, limitations.

---

## 3. Repo Assets You Already Have

In this workspace:

- `Paper_Summary.md`: your Task 1 review; use it as motivation and related work.
- Official method implementations under `Repos/`:
  - `Repos/PMP-master/` (PMP, ICLR 2024)
  - `Repos/SEC-GFD-main/` (SEC-GFD, AAAI 2024)
  - `Repos/CARE-GNN-master/` (CARE-GNN, CIKM 2020)
  - `Repos/GAGA-master/` (GAGA, WWW 2023)

**Important:** these repos are not currently integrated under one unified runner. The benchmark you build is the integration layer.

---

## 4. Environment Setup (Make It Run)

This project touches multiple research repos. The fastest path is to run in a GPU-friendly Linux environment.

Recommended options:

1. **Google Colab (recommended for PMP)**
   - PMP in this workspace is CUDA-centric, so a Colab GPU runtime avoids most setup pain.
2. **WSL2 (Ubuntu) on Windows**
   - Often easier than native Windows for DGL/PyTorch + CUDA combos.
3. **Native Windows**
   - Possible, but deep learning package compatibility can be more fragile.

Python version guidance:
- Use **Python 3.10 or 3.11** for best compatibility.
- Avoid relying on the system default Python if it is very new (many ML wheels lag behind).

Dependency guidance:
- Treat each method repo as its own "unit".
  - PMP: start from `Repos/PMP-master/requirements.txt`
  - CARE-GNN: start from `Repos/CARE-GNN-master/requirements.txt`
  - GAGA: start from `Repos/GAGA-master/pytorch_gaga/gaga_env.yaml` (but it is pinned to older versions)
  - SEC-GFD: install `torch`, `dgl`, `scikit-learn`, `numpy`

For the benchmark runner you write, pick one consistent stack (recommended):
- `torch` + `dgl` + `numpy` + `scikit-learn` + `pandas` + `matplotlib`

---

## 5. Recommended Minimal Scope (Finishable and Strong)

To keep the project finishable and still meaningful, I recommend this minimal experimental set:

**Datasets**
- Primary: **YelpChi** (multi-relation, commonly used in all four papers)
- Optional second dataset: **Amazon** (include only if time allows)

**Models**
- Baseline 1: **MLP** (features only)
- Baseline 2: **Vanilla GraphSAGE or GCN** (standard message passing)
- Specialized 1 (from `Repos/`): **PMP** (`Repos/PMP-master/`)
- Specialized 2 (from `Repos/`): **SEC-GFD** (`Repos/SEC-GFD-main/`)

This matches your requirement: `MLP + baseline GNN + 2 specialized methods`.

**Why these two specialized methods?**
- PMP and SEC-GFD target heterophily/label-imbalance issues in different ways, so degradation trends under heterophily/noise are interesting.
- Both can be run on YelpChi with a DGL-based data pipeline.

**Optional extension (if you want a camouflage-focused result)**
- Add **CARE-GNN** as an extra model and compare its degradation under camouflage vs the others.

---

## 6. Evaluation Protocol (Avoiding "Pitfalls")

Your review cites *Pitfalls of GNN Evaluation*. To apply that lesson, make your benchmark reproducible and variance-aware:

**Splits**
- Keep one **fixed train/val/test split** across all stress levels for fairness, or generate multiple splits and report variance.
- Minimum: 1 split.
- Better: 3-5 different splits (if compute allows).

**Seeds**
- Minimum: 5 seeds.
- Better: 10 seeds.

**Metrics (imbalance-aware)**
- ROC-AUC (threshold-free)
- AP (Average Precision; more informative under imbalance)
- F1-macro (class-balanced, but threshold-sensitive)

**Thresholding rule (must be consistent)**
- For F1-based metrics, pick a threshold on the **validation set** (e.g., best F1 on val) and apply it to test.
- Do not choose the threshold directly on the test set (this inflates results).

**Report**
- Always report `mean +/- std` across seeds (and splits if used).

---

## 7. Benchmark Design: What "Stress Tests" Mean

You will implement 3 families of stress tests. Each test is parameterized by a **severity** value in `[0, 1]` (or a small set like `{0.0, 0.1, 0.2, 0.3}`).

### 7.1 Heterophily Stress (Increase disagreeing-label edges)

**Goal:** Increase the fraction of edges that connect different labels, making neighbor aggregation less reliable.

**Core statistic to track**
- Edge heterophily ratio:
  - `hetero_ratio = (# edges (u,v) where y[u] != y[v]) / |E|`
  - Track per relation and overall.

**Implementation options**
1. **Edge rewiring (degree-preserving-ish)**
   - Pick a fraction `p` of edges.
   - For each selected edge `(u,v)`, replace it with `(u,v')` where `y[v'] != y[u]`.
2. **Heterophily edge injection**
   - Add `k` new edges from nodes to opposite-label nodes.
   - This increases density too; log density separately.

**Severity parameter**
- `p_rewire` or `p_add` (e.g., 0.0/0.1/0.2/0.3).

**Leakage note**
- This stress test uses true labels to construct heterophily edges. This is acceptable for an *oracle benchmark* (controlled stress test), but you must state it explicitly in the report.

**Suggested pseudocode (DGL-style, per relation)**

```python
# y: shape [N], values {0,1}
# For a DGL etype: src, dst = g.edges(etype=etype)

def hetero_ratio(src, dst, y):
    return (y[src] != y[dst]).float().mean().item()

def rewire_edges_to_increase_heterophily(src, dst, y, p_rewire, rng):
    m = len(src)
    n_rewire = int(p_rewire * m)
    edge_idx = rng.choice(m, size=n_rewire, replace=False)

    fraud_nodes = (y == 1).nonzero(as_tuple=True)[0]
    normal_nodes = (y == 0).nonzero(as_tuple=True)[0]

    dst2 = dst.clone()
    for e in edge_idx:
        u = int(src[e])
        pool = normal_nodes if int(y[u]) == 1 else fraud_nodes
        v_prime = int(pool[rng.integers(0, len(pool))])
        dst2[e] = v_prime

    return src, dst2
```

### 7.2 Camouflage Stress (Fraud nodes resemble normal ones)

Model both camouflage types discussed in CARE-GNN:

**A) Feature camouflage**
- For a fraction `p_cam_feat` of fraud nodes:
  - Replace fraud feature vector with a normal node feature vector, or
  - Blend it: `x_fraud = (1-gamma)*x_fraud + gamma*x_normal`.

**B) Relation camouflage**
- For a fraction `p_cam_edge` of fraud nodes:
  - Add edges from fraud nodes to randomly sampled normal nodes (per relation).
  - Optionally remove some fraud-to-fraud edges (if present).

**Severity parameters**
- `p_cam_feat` and/or `p_cam_edge` (keep one knob if you want to simplify).

**What to report**
- Performance degradation vs severity.
- Also log "camouflage success" proxies:
  - Change in fraud-to-normal neighbor ratio.
  - Change in fraud feature similarity to normals (e.g., cosine similarity).

**Suggested pseudocode (feature camouflage)**

```python
# X: shape [N, D] float32 features
# y: shape [N] labels {0,1}

def feature_camouflage(X, y, p_cam_feat, gamma, rng):
    fraud = (y == 1).nonzero(as_tuple=True)[0]
    normal = (y == 0).nonzero(as_tuple=True)[0]
    n = int(p_cam_feat * len(fraud))
    chosen = fraud[rng.choice(len(fraud), size=n, replace=False)]

    X2 = X.clone()
    for i in chosen:
        j = int(normal[rng.integers(0, len(normal))])
        X2[i] = (1.0 - gamma) * X2[i] + gamma * X2[j]
    return X2
```

### 7.3 Noise / Dense Neighborhood Stress

**Goal:** Add irrelevant edges so neighborhoods get "flooded" (as discussed in PMP's dense-graph analysis).

**Implementation**
- Add `m` random edges (uniformly random pairs), optionally per relation.
- Or add `k` random neighbors per node (degree inflation).

**Severity parameter**
- `edge_noise_rate = added_edges / |E|` (e.g., 0.0/0.05/0.10/0.20).

**What to log**
- Edge count before/after.
- Mean/median degree before/after.
- Optional: 2-hop neighborhood size distribution before/after (shows neighborhood explosion).

**Suggested pseudocode (random edge noise)**

```python
# Add random edges; if you want "undirected noise", also add the reverse edges.

def add_random_edges(g, etype, edge_noise_rate, rng):
    src, dst = g.edges(etype=etype)
    m_add = int(edge_noise_rate * len(src))
    n_nodes = g.num_nodes()

    src_add = rng.integers(0, n_nodes, size=m_add)
    dst_add = rng.integers(0, n_nodes, size=m_add)

    g2 = g.clone()
    g2 = dgl.add_edges(g2, src_add, dst_add, etype=etype)
    return g2
```

---

## 8. A Practical Architecture (How to Implement This Cleanly)

To keep results comparable across models, use this pattern:

1. **Load a base dataset once.**
2. **Generate perturbed graph variants** for each `(scenario, severity, seed)` (or per split).
3. **Evaluate all models on the exact same graph variant**.
4. Save a single `results.csv`.

### 8.1 Data Representation (Recommended)

Use **DGL graphs** as the canonical in-memory representation because:
- PMP uses DGL's fraud datasets internally.
- SEC-GFD uses DGL.
- GAGA preprocessing is DGL-based.

CARE-GNN uses SciPy matrices + pickled adjacency lists; you can treat it as an optional extension.

### 8.2 Suggested Output Structure (Artifacts)

Create a `runs/` folder (or similar) that contains:

- `runs/graphs/<dataset>/<scenario>/<severity>/<split_id>/<seed>.bin`
  - DGL graphs saved via `dgl.data.utils.save_graphs`
- `runs/results.csv`
- `runs/plots/*.png`
- `runs/config.json` (experiment definition for reproducibility)

Even if you do not fully implement this exact structure, keep the idea:
**graphs and results must be cached and traceable to a config.**

---

## 9. How to Run the Two Specialized Methods (From `Repos/`)

These are the baseline reproduction commands from the checked-in repos. Your benchmark will either:

- call these methods as subprocesses (simpler but less integrated), or
- import their code and run them inside your runner (more integrated, cleaner results logging).

### 9.1 PMP (ICLR 2024) - `Repos/PMP-master/`

Repo entrypoint: `Repos/PMP-master/main.py`

Notes before you run it:
- The code is GPU-oriented (`torch.cuda.set_device` and `.cuda()` calls). It is easiest to run in a GPU environment (e.g., Colab).
- The CLI uses `--train_size` / `--val_size` (the README mentions `--train_ratio`, but the code uses `--train_size`).

Example command (Yelp):

```bash
cd Repos/PMP-master
pip install -r requirements.txt
python main.py --dataset yelp --train_size 0.4 --val_size 0.2 --gpu_id 0 --multirun 5 --model LA-SAGE-S
```

Where to inject stress tests if you integrate it:
- Dataset is loaded and masks are set inside `Repos/PMP-master/DataHelper/datasetHelper.py` in `DatasetHelper.load()`.
- Apply your graph perturbations immediately after `data = dataset[0]` and before loaders are created.

### 9.2 SEC-GFD (AAAI 2024) - `Repos/SEC-GFD-main/`

Repo entrypoint: `Repos/SEC-GFD-main/main.py`

Example command (YelpChi via DGL):

```bash
cd Repos/SEC-GFD-main
pip install dgl torch scikit-learn numpy
python main.py --dataset yelp --train_ratio 0.4 --epoch 100 --ntrials 5
```

Important notes (practical fixes you may need):
- `Repos/SEC-GFD-main/main.py` hardcodes `device = cuda:1` when CUDA is available. If you only have one GPU, this will crash. Change it to `cuda:0` or make it configurable.
- The split is currently fixed by `random_state=2` in `train_test_split`. For multiple splits, change `random_state` to depend on the run seed.
- The script currently uses the **test set** inside training to pick thresholds / track "best" performance. For fair evaluation, select thresholds on validation, monitor validation for early stopping, and compute test metrics once at the end (or clearly disclose this leakage if you keep the original behavior).

Where to inject stress tests if you integrate it:
- Dataset graph is created in `Repos/SEC-GFD-main/dataset.py` (class `Dataset`), then used in `main.py`.
- Apply perturbations right after `graph = Dataset(...).graph` and before calling `train(...)`.

---

## 10. Baselines (MLP + Vanilla GNN)

You need baselines to interpret robustness results.

### 10.1 MLP Baseline (Features Only)

**Purpose:** measures how much "graph structure" helps beyond raw attributes, and is a required sanity check per Pitfalls-style evaluation.

Implementation:
- A 2-3 layer MLP on node features.
- Use the same train/val/test masks as the GNNs.

### 10.2 Vanilla GNN Baseline (GraphSAGE or GCN)

**Purpose:** a standard message passing model that should be sensitive to heterophily and noisy neighbors.

Implementation:
- 2-layer GraphSAGE (mean aggregation) or GCN.
- Same splits, same metrics.

---

## 11. Experiment Matrix (What to Actually Run)

If you try to run everything, the Cartesian product explodes. Use a minimal, publishable matrix:

### Minimal matrix (recommended)
- Dataset: YelpChi
- Models: MLP, GraphSAGE, PMP, SEC-GFD
- Scenarios and severities:
  - Heterophily: `p = {0.0, 0.1, 0.2, 0.3}`
  - Camouflage (feature): `p = {0.0, 0.1, 0.2, 0.3}`
  - Noise edges: `r = {0.0, 0.05, 0.10, 0.20}`
- Seeds: 5
- Splits: 1 (optional: 3 splits if time allows)

### Add Amazon only if time allows
- Repeat only the clean + highest severity points (e.g., severity 0.0 and 0.3/0.2) to keep compute bounded.

---

## 12. Results: How to Present Robustness Clearly

Your report should not only show "best numbers"; it should show **robustness curves**.

Recommended plots (pick 3-6):

1. Performance vs severity curves for each scenario (AUC and AP are most informative).
2. Same curves for F1-macro (with consistent thresholding).
3. Bar plot: clean vs max-stress drop (`delta = metric_clean - metric_stress`).
4. Optional: robustness score = area under the curve (AUC over severity), per method.

Recommended tables:
- Clean metrics (mean +/- std)
- Max-stress metrics (mean +/- std)
- Drop (mean +/- std)

Always annotate:
- dataset, split ratio, number of seeds, number of splits, and whether perturbations used oracle labels.

---

## 13. Final Report Outline (Suggested)

Use your existing `Paper_Summary.md` structure as background and then add:

1. Introduction and motivation (messy graphs, camouflage, heterophily, noisy neighborhoods)
2. Related work summary (briefly reference CARE-GNN / PMP / SEC-GFD / GAGA + Pitfalls)
3. Benchmark definition
   - datasets, task, splits, metrics
   - stress tests with clear algorithms and severity parameters
4. Models evaluated (2 baselines + 2 specialized)
5. Results
   - curves, tables, variance
6. Discussion
   - which scenario hurts which model most and why
   - practical implications for fraud detection pipelines
7. Limitations
   - oracle-label stress tests, static graphs, dataset version issues
8. Conclusion and next steps (optional extensions)

---

## 14. Common Pitfalls (Practical, Not Theoretical)

These issues are easy to miss and will cost time late in the project:

1. **Inconsistent splits across stress levels**
   - Fix masks per split and reuse them for all severities.
2. **Threshold leakage**
   - Do not choose test thresholds on test data.
3. **Dataset version mismatch**
   - If you compare across repos, confirm they load the same dataset variant and relation definitions.
4. **GPU-only code paths**
   - PMP in this workspace is CUDA-centric; plan to run it on GPU (Colab is fine).
5. **CARE-GNN dataset path mismatch (if you add it)**
   - `Repos/CARE-GNN-master/utils.py` expects `data/YelpChi.mat` and `data/Amazon.mat`.
   - In this workspace the `.mat` files are under `Repos/CARE-GNN-master/data/YelpChi/YelpChi.mat` and `Repos/CARE-GNN-master/data/Amazon/Amazon.mat`.
   - You must either move/copy the `.mat` files or update the file paths in `Repos/CARE-GNN-master/utils.py` and `Repos/CARE-GNN-master/data_process.py` before running.

---

## 15. Quick "Day-by-Day" Execution Plan

If you want a realistic schedule:

1. Day 1-2: Implement dataset loading + fixed split logic + baseline MLP.
2. Day 3-4: Implement baseline GraphSAGE/GCN + metrics + CSV logging.
3. Day 5-7: Implement the 3 perturbation families + graph stats logging.
4. Day 8-10: Integrate PMP and SEC-GFD into the runner (or run them as subprocesses) and confirm clean-graph results are reasonable.
5. Day 11-14: Run the experiment matrix, generate plots, write the final report.

---

## 16. Optional Extensions (If Everything Works Early)

If you finish the core benchmark early, the best extensions (high value per effort) are:

- Add CARE-GNN as an extra method and show it degrades less under camouflage.
- Add a "train-clean, test-perturbed" mode to approximate test-time attacks.
- Add a robustness metric like worst-case performance over severities.
- Add GAGA (requires preprocessing) and compare attention/grouping vs message passing.
