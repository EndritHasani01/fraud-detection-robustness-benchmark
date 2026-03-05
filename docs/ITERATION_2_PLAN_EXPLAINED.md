# Iteration 2 Plan — Improvement Proposals (Plain-English Guide)

This file explains, in non-technical terms, what each improvement proposal in [`docs/ITERATION_2_PLAN.md`](ITERATION_2_PLAN.md) means and why it matters.

## A quick glossary (so the rest reads easily)

- **Dataset / graph**: the fraud “network” we study (accounts/businesses are dots; relationships are lines).
- **Scenario**: *what kind* of stress we apply (e.g., add noise, make neighbors less similar, camouflage fraudsters).
- **Severity**: *how strong* the stress is (low → high).
- **Variant**: one “edited copy” of the original dataset after applying a scenario at a chosen severity.
- **Model**: the detection method we are testing (MLP, GraphSAGE, PMP, SEC‑GFD).
- **Run**: one training + evaluation of a model on a specific variant.
- **Seed**: a fixed “randomness setting” so results can be reproduced (like shuffling a deck in a repeatable way).

---

## A) Reduce runtime without losing the robustness story

**A1. Decouple `graph_seeds` from `training_seeds`**

- **What it means**: Use one set of seeds for *creating stressed datasets* (graph variants) and a separate set of seeds for *training the models*. Today, one seed list drives both, which multiplies work.
- **Why it matters**: You can increase confidence in results (more training repeats) without also creating a huge number of extra dataset variants (or vice‑versa). This prevents the number of runs from exploding.

**A2. Add “train once on clean, evaluate on all stressed variants” mode**

- **What it means**: Instead of re-training the model from scratch for every stressed dataset, you train the model **one time** on the original (“clean”) dataset, then you **only test** it on each stressed variant.
- **Why it matters**: This is a standard robustness question (“How does a model trained on normal data handle changes at test time?”) and it’s dramatically cheaper because training is the slow part.

**A3. Use a coarse‑to‑fine severity grid**

- **What it means**: Start with just two severity points per scenario: **no stress** and **maximum stress**. Add middle severities later only if you need a detailed curve.
- **Why it matters**: You get an early, publishable “performance drop” result quickly, without committing to a long full sweep.

**A4. Allow per‑model subsets (fast models run more, slow models run less)**

- **What it means**: Run the full set of variants for cheap baselines, but run only a reduced set (e.g., clean + max stress) for expensive research models.
- **Why it matters**: You still compare models fairly on the most important cases, while keeping total runtime practical.

---

## B) Make runs resumable and prevent duplicated work

**B1. Add a stable “run key” and skip‑if‑done**

- **What it means**: Give every expected results row a unique ID (based on dataset, scenario, severity, seeds, model, etc.). Before doing work, check whether a successful result for that exact ID already exists.
- **Why it matters**: If the process stops (laptop sleep, crash, time limit), you can restart and it continues where it left off—without duplicating results or wasting time.

**B2. Add a `--skip-existing` option**

- **What it means**: A simple switch to “don’t redo things that already have results,” plus a separate “force rerun” option when you *do* want to overwrite.
- **Why it matters**: It makes reruns safe by default and reduces accidental wasted compute.

**B3. Improve error reporting and add “retry only failed”**

- **What it means**: When something fails, record that failure clearly in the results (what failed and why), and allow a mode that re-runs only those failed cases.
- **Why it matters**: Long experiments become manageable: you don’t restart from zero just because a few runs failed.

---

## C) Make the two expensive research models cheaper per run

**C1. Expose SEC‑GFD “cost knobs”**

- **What it means**: Make SEC‑GFD settings adjustable (like model size and training length) through config/CLI instead of hard‑coded defaults.
- **Why it matters**: You can choose “fast, good‑enough” settings for the benchmark and reserve heavier settings only when needed.

**C2. Cache SEC‑GFD’s one‑time math setup**

- **What it means**: SEC‑GFD does a small but nontrivial calculation during setup; compute it once and reuse it across runs when the relevant setting is the same.
- **Why it matters**: This removes repeated overhead that adds up across many runs.

**C3. Add safer PMP sampling options**

- **What it means**: Instead of always looking at *all* neighbors in the graph (slow), allow PMP to look at a limited sample of neighbors per step (much faster), controlled from the benchmark config.
- **Why it matters**: Sampling is a common speed trick in graph learning and can cut runtime significantly while keeping results meaningful.

---

## D) Make the benchmark easier to run day-to-day

**D1. Add a pre‑flight runtime estimator**

- **What it means**: Before starting, print how many variants, seeds, and models are selected, and how many total runs that implies (optionally with a rough time estimate).
- **Why it matters**: It prevents accidentally launching an experiment that takes days.

**D2. Better progress reporting (and ETA)**

- **What it means**: Show “where we are” in the experiment (e.g., variant 12/46, seed 2/5) and an estimated time remaining.
- **Why it matters**: You can tell whether things are stuck and plan your time (especially on shared machines or limited sessions).

**D3. Add a “fast dev” config**

- **What it means**: Provide a small, intentionally cheap experiment definition (few seeds, only extremes of severity, only baselines) that finishes in minutes.
- **Why it matters**: It’s the quickest way to confirm the pipeline works end‑to‑end before launching big runs.

---

## E) Data, evaluation, and reporting improvements

**E1. Use multiple train/validation/test splits (if compute allows)**

- **What it means**: Repeat the experiment with a few different ways of splitting the dataset into training and testing portions.
- **Why it matters**: It reduces the chance that results are an “accident” of one particular split and makes conclusions stronger.

**E2. Add confidence intervals (e.g., via bootstrap)**

- **What it means**: Instead of only reporting “average ± standard deviation,” estimate a confidence band that better communicates uncertainty.
- **Why it matters**: It makes the final report more statistically defensible and easier to interpret.

**E3. Make “oracle” perturbations explicit**

- **What it means**: Some stress tests may use ground‑truth labels to construct the perturbation (they “know the answers” while generating the stress). This needs to be clearly labeled everywhere.
- **Why it matters**: It prevents misleading interpretations and keeps the report transparent about what was (and wasn’t) realistic.

