# Stakeholder Guide: Fraud Detection Robustness Benchmark

This guide explains the current v3 benchmark in plain English for supervisors, reviewers, collaborators, and other non-technical readers.

For the full technical contract of the benchmark, including scenario semantics, output schemas, and interpretation boundaries, use [docs/V3_BENCHMARK_REFERENCE.md](V3_BENCHMARK_REFERENCE.md).

## Executive Summary

This project is a controlled testing lab for graph-based fraud detection models.

It answers one practical question:

> If the fraud graph becomes noisier, more deceptive, or structurally less reliable, which models hold up best?

The benchmark does this by:

1. Starting from one real fraud dataset.
2. Generating deterministic stressed versions of that graph.
3. Running several fraud detection models under the same conditions.
4. Measuring how performance changes.
5. Exporting both model results and perturbation audits that explain what actually changed.

## What The Current V3 System Adds

Compared with the earlier benchmark versions, v3 strengthens the system in four important ways.

- It separates feature camouflage from relation camouflage instead of treating camouflage as one vague family.
- It adds a non-oracle heterophily path so the benchmark is not limited to privileged-label perturbations.
- It exports `variant_audit.csv`, which records requested and realized perturbation strength for every graph variant.
- It keeps both evaluation protocols visible all the way through the reporting layer.

## What The Benchmark Does

The current benchmark uses YelpChi and evaluates up to four models:

- `mlp`
- `sage`
- `pmp`
- `secgfd`

It applies five named stress scenarios in the main v3 config:

- `heterophily_rewire_oracle`
- `heterophily_rewire_nonoracle`
- `camouflage_feature_oracle`
- `camouflage_relation_oracle`
- `noise_edges_uniform`

These scenarios are deterministic with respect to graph seeds, and the model-training randomness is controlled separately through training seeds.

## What The Stress Scenarios Mean

### Heterophily stress

This makes neighborhoods less label-consistent by rewiring edges. The benchmark includes both an oracle version that uses true labels and a non-oracle version that uses feature-derived pseudo partitions.

This is useful for asking:
How sensitive is a model to misleading neighborhood structure?

### Feature camouflage

This edits the features of selected fraud nodes so they look more like sampled normal nodes.

This is useful for asking:
How much does a model depend on fraud nodes remaining obviously separable in feature space?

### Relation camouflage

This changes who selected fraud nodes connect to by adding benign-looking structural neighbors and optionally removing part of the suspicious neighborhood.

This is useful for asking:
Can a model still detect fraud when the local relation pattern is made to look more normal?

### Noise stress

This adds random edges to increase graph density and clutter.

This is useful for asking:
How robust is a model when the graph becomes more crowded and less informative?

## Two Evaluation Modes

The benchmark supports two different evaluation questions.

### `train_on_variant`

The model is retrained on each stressed graph before it is evaluated.

This answers:
How well can the model adapt when both training and evaluation data are already stressed?

### `train_clean_eval_all`

The model is trained once on the clean graph and then evaluated on every stressed variant.

This answers:
How much performance is lost under test-time shift when the model was trained only under normal conditions?

These two protocols are both important, and they should not be averaged together.

## The Main Artifacts

Outputs are written under `runs/<experiment_name>/`.

The most important files are:

- `graph_variants.csv`
  - the ledger of clean and stressed graph files
- `variant_audit.csv`
  - the companion audit table that records requested and realized perturbation behavior
- `results.csv`
  - the unified run-level evaluation table
- `plots/summary_curves.csv`
  - model performance versus severity
- `plots/audit_curves.csv`
  - realized perturbation behavior versus severity
- `plots/performance_audit_join.csv`
  - the report-stage join between model results and audit evidence

In plain terms:

- `results.csv` tells you how a model performed
- `variant_audit.csv` tells you what the scenario actually did
- the plots stage combines those into report-ready summaries

## Why The Audit File Matters

Raw severity numbers are not enough by themselves.

For example, a severity of `0.3` can mean very different things across different scenario families. The audit file makes the benchmark easier to trust because it shows what was actually realized:

- how many nodes or edges changed
- how much heterophily increased
- how much cosine similarity shifted under feature camouflage
- how much neighborhood composition shifted under relation camouflage
- how many unique edges were actually added under noise

## What The Benchmark Is Not

This benchmark is not:

- a live fraud detection product
- a realistic attacker simulator
- a temporal fraud evolution model
- an adaptive attacker-defender environment
- proof that a model is production-ready

It is a static, reproducible stress benchmark for comparative evaluation.

## How To Interpret Results Responsibly

Stakeholders should keep four rules in mind.

### 1. Read scenarios as controlled stress tests

The benchmark is designed to reveal failure modes under stress, not to replay real-world fraud behavior end to end.

### 2. Keep oracle status visible

Some scenarios use privileged information to build the perturbation. That is acceptable for benchmarking, but it must be disclosed clearly and not described as a realistic attack process.

### 3. Keep protocol meaning visible

The same metric curve means something different under `train_on_variant` and `train_clean_eval_all`. Conclusions should always state which protocol they refer to.

### 4. Compare severity within a family first

Do not assume that `0.3` means the same stress strength in heterophily, camouflage, and noise. Use within-family trends and realized audit metrics instead.

## Recommended Review Workflow

If you are reviewing a benchmark run, the simplest sequence is:

1. Confirm which frozen config was used: `v3_fast`, `v3`, or `v3_full`.
2. Confirm which protocol is being discussed.
3. Review the model-performance plots.
4. Review the corresponding audit curves or `variant_audit.csv` rows.
5. Check whether oracle scenarios are disclosed.
6. Ask for one plain-language conclusion:
   Which model degrades most under which stress, and what evidence shows that the stress was actually realized?

## Quick Commands

### Fast validation

```powershell
py -m benchmark.run --config configs/exp_yelpchi_v3_fast.json --stage graphs
py -m benchmark.run --config configs/exp_yelpchi_v3_fast.json --stage baselines
py -m benchmark.run --config configs/exp_yelpchi_v3_fast.json --stage plots
```

### Main v3 benchmark

```powershell
py -m benchmark.run --config configs/exp_yelpchi_v3.json --stage graphs
py -m benchmark.run --config configs/exp_yelpchi_v3.json --stage matrix
py -m benchmark.run --config configs/exp_yelpchi_v3.json --stage matrix --protocol train_clean_eval_all
py -m benchmark.run --config configs/exp_yelpchi_v3.json --stage plots --ci
```

## Final Takeaway

The current v3 system is best understood as a reproducible evidence generator.

It does not tell you:
"This is the one model that will always work in production."

It tells you:

- which models are more or less robust under controlled graph stress
- whether that conclusion changes across protocols
- what the perturbation actually did
- where the benchmark deliberately stops short of simulation realism
