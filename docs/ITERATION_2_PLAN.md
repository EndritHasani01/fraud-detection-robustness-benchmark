# Iteration 2 (v2) Plan - Improvements and Next Engineering Steps

Last updated: **2026-03-04**

This document describes a **second iteration** of the benchmark (a "v2" pass): what should be improved, why it matters, and what concrete changes to implement so the project is faster to run, easier to resume, and easier to extend.

> Context: In v1, the end-to-end pipeline exists (graphs -> models -> results.csv -> plots). The main blocker is that **running the full experiment matrix is very slow**, especially for the vendored research models (PMP, SEC-GFD), even on a GPU.

---

## 1) Goals for v2

1. **Make the full experiment feasible** on typical student hardware/time budgets.
2. **Avoid wasted compute** (re-running already completed combinations, duplicating rows).
3. **Improve reproducibility and traceability** (every plot/table maps to results rows; runs are resumable).
4. **Make the system extensible** (add CARE-GNN/GAGA/Amazon later with minimal friction).

---

## 2) What is limiting v1 right now (root causes)

### 2.1 The experiment matrix explodes (too many trainings)

In the current design, a "run" is roughly:

- choose a cached graph variant (scenario + severity + perturbation seed)
- choose a training seed
- train a model from scratch

For `configs/exp_yelpchi_v1.json` the default evaluation space is large:

- ~46 evaluated graph variants (clean + applied variants)
- 5 training seeds
- 4 models
- ~920 separate training runs

Even if one training run is "only minutes", 920 of them is multiple days.

### 2.2 Seeds are coupled, creating near-quadratic scaling

In v1, the list `seeds.training_seeds` is used for:

1) **Training randomness** (model init / shuffling / etc), and also
2) **Graph perturbation randomness** (`graph_seed` when generating variants)

That means increasing the number of seeds increases:

- how many *variants* exist, and also
- how many *trainings per variant* exist

Total trainings scales like **S (graph seeds) * S (training seeds)**, which becomes painful fast.

### 2.3 SEC-GFD is inherently expensive per epoch (full-graph, many message passes)

SEC-GFD's forward pass (in `Repos/SEC-GFD-main/model/SECGFD.py`) performs multiple DGL `update_all` passes over the full graph (plus a GCN), and the benchmark trains it for multiple epochs. YelpChi in homogeneous view has ~8 million edges, so this is heavy even on a T4.

### 2.4 Results are append-only and not resumable by key

Stages append to `results.csv` without enforcing uniqueness, so:

- re-running can create duplicate rows
- resuming a partially completed long run requires manual editing/cleanup

This makes iteration slow and error-prone.

---

## 3) v2 improvement themes (what to change)

### A) Reduce compute cost without losing the "robustness story"

**A1. Decouple `graph_seeds` from `training_seeds` (must-have)**

Add a separate seed list in the config:

- `seeds.graph_seeds`: controls randomness of perturbations
- `seeds.training_seeds`: controls randomness of model training

Recommended default for v2:

- `graph_seeds`: 1 (or 2) seeds per scenario/severity (enough to show perturbation variance)
- `training_seeds`: 3-5 seeds per run (variance for models)

This changes scaling from ~S^2 to ~S_graph * S_train.

Implementation notes:

- Keep v1 config frozen.
- Create a new config file (e.g. `configs/exp_yelpchi_v2.json`) and update the runner to prefer `graph_seeds` when present.

**A2. Add an optional "train-on-clean, eval-on-perturbed" mode (high impact)**

Add a mode where for each `(model, training_seed)` you:

1) train once on the clean graph
2) evaluate (forward only) on all perturbed variants

This answers a slightly different robustness question ("test-time shift"), but it is a standard and defensible setting and is dramatically cheaper.

Recommended approach:

- Implement it as a separate stage or flag (example names):
  - `--protocol train_clean_eval_all`
  - or a new stage: `--stage matrix_shift`
- Write rows to the same `results.csv` but add a field like `train_graph_ref` to indicate training graph identity (clean vs stressed).

**A3. Coarse-to-fine severity grids (practical)**

Instead of 4 severity points per scenario, run:

- 2 points first: `{0.0, max}`
- then add intermediate points only for scenarios where curves are interesting

This produces publishable "drop" plots quickly and later expands to curves if time allows.

**A4. Allow per-model subsets**

In practice, you can run:

- full matrix for baselines (fast)
- reduced matrix for PMP/SEC-GFD (expensive), e.g. only clean + max-stress

Make this easy in v2 by supporting:

- `--models mlp,sage` and `--models pmp,secgfd`
- or per-model variant caps in config

---

### B) Make runs resumable and prevent duplicated work

**B1. Add a stable "run key" and skip-if-done (must-have)**

Define a unique key for each expected row:

`(dataset_id, split_id, scenario_id, severity, graph_seed, training_seed, model_id, protocol)`

Before training, check `results.csv` for an existing `status=ok` row with that same key.
If it exists, skip training and move on.

This enables:

- safe interruption/resume (especially on Colab)
- re-running matrix without duplicating rows

**B2. Add a `--skip-existing` flag**

- Default could be `--skip-existing` on (safer), with `--force` to rebuild.

**B3. Improve error handling and reporting**

When a run errors:

- write `status=error`, include a short message
- allow `--retry-errors` mode to only rerun failed keys

---

### C) Make the expensive repo models cheaper per run

**C1. Expose SEC-GFD cost knobs**

In `benchmark/secgfd_stage.py`, SEC-GFD is currently created with fixed defaults:

- `hid_dim`, `order_d`, `high_order`
- `epochs`, `patience`

v2 should expose these via:

- config (preferred), and/or
- CLI flags (quick iteration)

Practical defaults for feasibility:

- fewer epochs (e.g. 20-50 with early stopping)
- smaller `hid_dim` (e.g. 32)
- smaller `high_order` (0-1)

**C2. Cache SEC-GFD theta computation**

SEC-GFD computes polynomial coefficients using SymPy/Scipy during model construction.
This should be computed once per `(d)` and reused across runs to reduce per-run overhead.

**C3. PMP sampling configuration**

PMP often runs with `full_neighbors: true`, which can be heavy on YelpChi.
v2 should allow safe overrides (without editing the research repo):

- set `full_neighbors: false`
- set `sampled_neighbors` fanouts per layer (e.g. `[10]` or `[10, 5]`)
- increase `num_workers` on Linux/Colab (keep `0` on Windows for stability)

---

### D) Make the benchmark easier to operate

**D1. Add a pre-flight runtime estimator**

Before launching, print:

- number of variants selected
- number of training seeds
- number of models
- expected total run count
- (optional) estimated wall time based on previous `duration_sec` medians

This prevents accidentally starting a 2-day run.

**D2. Better progress reporting**

- progress counters: `variant i/N`, `seed j/K`, `model m/M`
- periodic ETA based on moving average duration

**D3. Add a "fast dev" config**

Provide a small, intentionally cheap config like:

- 1 graph seed
- 1 training seed
- severities `{0, max}`
- only baselines

Example: `configs/exp_yelpchi_dev.json`

This makes it easy to verify the pipeline in minutes.

---

### E) Data, evaluation, and reporting improvements

**E1. Multiple splits (optional if compute allows)**

To strengthen conclusions, add `data_splits` entries (s0, s1, s2) and report mean/std across splits as well as seeds.

**E2. Add confidence intervals / bootstrap (optional)**

For the final report, confidence intervals can be more informative than raw std.

**E3. Keep disclosures explicit**

Some perturbations are "oracle" (use ground-truth labels to rewire/camouflage).
v2 should surface this clearly in:

- `graph_variants.csv`
- plots legends/titles
- report text

---

## 4) Proposed v2 implementation plan (phased)

### Phase 1: Quick wins (1-2 days)

1. Add `--skip-existing` and stable run-key checking for all training stages.
2. Add runtime estimator printout (counts + expected rows).
3. Add new config `configs/exp_yelpchi_v2_fast.json` with fewer severities and seeds.

### Phase 2: Biggest speedup (2-4 days)

4. Decouple `graph_seeds` from `training_seeds` (new config field + runner support).
5. Implement "train-clean, eval-perturbed" protocol as an optional stage/flag.

### Phase 3: Model-level tuning (as needed)

6. Expose SEC-GFD complexity knobs and reduce defaults in the v2 config.
7. Allow PMP sampler overrides via benchmark-side config.

### Phase 4: Extensions (only if time remains)

8. Integrate CARE-GNN or GAGA behind an adapter layer, keeping `Repos/` changes minimal.
9. Add Amazon dataset config + rerun a reduced matrix.

---

## 5) Definition of done for v2

v2 is "done" when:

- You can start a long run, interrupt it, and resume **without duplicating rows**.
- A reasonable config finishes in **hours, not days**, on a T4.
- The benchmark can generate plots (`--stage plots`) from a complete `results.csv`.
- The report can cite every figure/table back to specific rows in `results.csv`.

