# Project Feasibility And Benchmark Review

Review date: 2026-04-24  
Repository: `fraud-detection-robustness-benchmark`

## Executive Verdict

This project is feasible and has real research meaning if it is framed precisely:

> A controlled, static robustness benchmark for graph-based fraud detection on YelpChi, measuring how model performance changes under deterministic graph and feature stress tests.

That framing is defensible for a faculty research project. It is not a full adversarial fraud simulator, not a temporal fraud evolution model, and not evidence that a detector is production-ready. The current repository mostly understands this boundary in `docs/V3_BENCHMARK_REFERENCE.md` and `docs/STAKEHOLDER_GUIDE.md`.

The benchmark is meaningful because the papers reviewed in `Paper_Summary.md` do not share one common robustness benchmark. CARE-GNN, PMP, SEC-GFD, and GAGA each emphasize different failure modes: camouflage, heterophily, imbalance, spectral behavior, noisy neighborhoods, and low homophily. This project turns those themes into a shared evaluation harness with common splits, metrics, stress scenarios, and output artifacts.

The implementation is stronger than a typical student prototype. It has frozen configs, deterministic graph caches, separate graph and training seeds, unified result schema, validation-only threshold selection, perturbation audits, resumable runs, tests, and plot-ready outputs. There are still important limitations and a few concrete implementation issues that should be fixed or disclosed before the final report.

## Current Scope

The current repo has moved beyond the older v2 notes. The active docs and configs describe v3:

- Main config: `configs/exp_yelpchi_v3.json`
- Fast config: `configs/exp_yelpchi_v3_fast.json`
- Full config: `configs/exp_yelpchi_v3_full.json`
- Technical contract: `docs/V3_BENCHMARK_REFERENCE.md`

The main v3 config evaluates:

- Dataset: YelpChi from `dgl.data.FraudDataset`
- Canonical graph view: homogeneous graph after source heterograph generation
- Models: `mlp`, `sage`, `pmp`, `secgfd`
- Scenarios:
  - `heterophily_rewire_oracle`
  - `heterophily_rewire_nonoracle`
  - `camouflage_feature_oracle`
  - `camouflage_relation_oracle`
  - `noise_edges_uniform`
- Seeds:
  - `graph_seeds = [0, 1]`
  - `training_seeds = [0, 1, 2]`
- Splits:
  - one split, `s0`, using train/val/test ratio 0.4/0.2/0.4

This is a reasonable final-project scope. It is not too small, because four models and five stress scenarios already create a nontrivial matrix. It is also not too large, because it avoids integrating every vendored research repository.

## What The Benchmark Actually Measures

The benchmark answers two related but different questions:

1. `train_on_variant`
   - Train and evaluate on the same stressed graph.
   - Measures adaptation under stressed training data.

2. `train_clean_eval_all`
   - Train once on the clean graph and evaluate on stressed graphs.
   - Measures test-time graph or feature shift robustness.

These protocols must not be averaged together. A model can be fragile under shift but still adapt when retrained on the stressed graph.

The current checked `runs/gfd_robustness_benchmark_v3/results.csv` contains only `train_clean_eval_all` rows:

- `mlp`: 63 ok rows
- `sage`: 63 ok rows
- `pmp`: 63 ok rows
- `secgfd`: 63 ok rows

So the current run supports shift-robustness discussion only. It does not yet support conclusions about `train_on_variant` for the main v3 run unless that protocol is run separately.

## Research Meaning

The project has a valid research question:

> Which graph-fraud detectors degrade most under controlled changes to neighborhood label consistency, feature camouflage, relation camouflage, and graph density?

That question matters because graph fraud detection methods often claim robustness to messy neighborhoods, but the reviewed papers evaluate under different datasets, splits, metrics, and stress assumptions. A shared benchmark makes those claims more comparable.

The strongest research contribution is not inventing a new model. It is benchmark design:

- common data source and split masks
- controlled perturbation families
- model-agnostic metrics
- explicit oracle disclosure
- separate training and graph randomness
- audit evidence for requested versus realized perturbation strength
- protocol-aware reporting

That is a legitimate contribution for a faculty project, especially if the final report is honest about limitations.

## Feasibility Assessment

### Feasible Parts

The following parts are feasible and already mostly implemented:

- Loading YelpChi via DGL.
- Creating deterministic train/validation/test masks.
- Generating graph variants for multiple stress families.
- Caching graph variants.
- Running feature-only MLP and GraphSAGE baselines.
- Running PMP and SEC-GFD through thin benchmark-side adapters.
- Writing one unified `results.csv`.
- Producing summary CSVs and plots.
- Exporting `variant_audit.csv` for perturbation evidence.
- Keeping oracle status visible.

### Main Feasibility Risks

The practical risks are environment and runtime, not benchmark design:

- DGL is not installed in the current Python environment used for this review.
- The current shell uses Python 3.13.7, but the README recommends Python 3.10 or 3.11 for DGL/PyTorch compatibility.
- Existing v3 run CSVs reference graph paths from a different checkout location and the local graph binaries are not present under the current `runs/` directory.
- PMP and SEC-GFD are heavier than the baselines and may need careful epoch/patience settings on CPU.
- Only one data split is configured in the main v3 config, so final statistical claims should be modest.

### Validation Performed

Command run:

```powershell
py -m pytest
```

Result:

```text
71 passed in 7.40s
```

Environment check:

```text
Python 3.13.7
torch 2.8.0+cpu
ModuleNotFoundError: No module named 'dgl'
```

Interpretation:

- The repository's unit and regression tests pass.
- The current environment cannot regenerate graphs or run model stages until DGL is installed in a compatible Python environment.
- The tests use fakes/mocks for much of the DGL-dependent logic, so passing tests does not prove that the full benchmark can currently run end to end on this machine.

## Benchmark Design Review

### Strong Design Choices

The benchmark has several well-designed pieces.

First, it decouples graph randomness from training randomness. `graph_seeds` control perturbation realizations, while `training_seeds` control model initialization and training noise. This is important because graph perturbation variance and optimization variance are different uncertainty sources.

Second, the run key is complete enough:

```text
(dataset_id, split_id, scenario_id, severity, graph_seed, training_seed, model_id, protocol)
```

This prevents mixing rows from different protocols, seeds, models, and stress variants.

Third, the benchmark uses validation-only threshold selection for F1. The baseline, PMP, and SEC-GFD stages all select the F1 threshold from validation scores and then evaluate on test scores. This avoids the most common test leakage error for threshold-sensitive fraud metrics.

Fourth, the MLP baseline is included. This is important. If a graph stress changes graph structure but not features, the MLP should not change under `train_clean_eval_all`. That behavior is visible in the current results and acts as a useful sanity check.

Fifth, v3 added `variant_audit.csv`. This is a major improvement because requested severity is not always realized severity. For example, edge rejection policies can reduce actual rewires or added edges. The audit file makes the benchmark more trustworthy.

Sixth, oracle scenarios are labeled. This is essential because oracle perturbations use true labels to construct stress. They are acceptable as controlled probes but should not be described as realistic attacks.

### Conceptual Limitations

The benchmark is not illogical, but it has boundaries that must be stated clearly.

#### Static Stress, Not Real Fraud Dynamics

The benchmark edits a static graph. It does not simulate:

- temporal fraud campaigns
- delayed labels
- fraudster adaptation
- defender retraining loops
- account creation or deletion
- transaction timing
- manual review
- business rules

This is acceptable for a course project. The final report should call it a static robustness benchmark, not a fraud-risk simulator.

#### Oracle Perturbations Are Not Operational Attacks

The oracle heterophily and camouflage scenarios use true labels to select fraud or normal nodes. That is useful for controlled stress testing, but unrealistic for a live attacker.

Safe wording:

> Oracle scenarios measure sensitivity to label-informed worst-case structural or feature changes.

Unsafe wording:

> These scenarios simulate how fraudsters attack the detector in production.

#### Severity Is Family-Specific

A severity of `0.3` in feature camouflage does not mean the same amount of stress as `0.3` in edge noise or relation camouflage. The configs and docs already recognize this. The report should compare severity mostly within a scenario family and use audit metrics for cross-family interpretation.

#### Homogeneous Evaluation Weakens Relation-Specific Claims

The v3 configs generate some perturbations on the source heterograph, then convert to a homogeneous canonical graph for evaluation. This improves perturbation semantics, but it means the evaluated models are not really being tested as full multi-relation models.

This matters especially for PMP and fraud GNNs whose paper motivation involves relation-aware behavior. In the current benchmark, the final evaluation graph is homogeneous, so relation-specific advantages are partly collapsed.

This is not fatal, but the report should say:

> Relation-aware perturbation generation is used, but downstream evaluation is on a shared homogeneous graph view for comparability.

#### YelpChi Is Dense, So Some Structural Perturbations May Be Weak

The current v3 graph has about 45,954 nodes and 8,051,348 edges. Average in/out degree is about 175. In such a dense graph, adding two camouflage edges per selected fraud node may barely change neighborhood composition.

The current audit curves show relation camouflage shifts the fraud-to-normal neighbor ratio only slightly. Therefore, if relation camouflage produces almost no performance drop, that may mean the perturbation is too weak, not that every model is robust to relation camouflage.

#### Feature Camouflage With `gamma = 1.0` Is Strong And Artificial

The main v3 config fully replaces selected fraud-node features with sampled normal-node features. This is a clean stress test, but it is more artificial than gradual blending.

A future config with `gamma = 0.25, 0.5, 0.75, 1.0` or separate gamma/severity controls would support a more nuanced analysis.

#### One Split Limits Statistical Confidence

The benchmark currently uses one fixed split. That is acceptable for a finishable project, but final conclusions should not overstate ranking stability. The final report can say that variance is measured over training and graph seeds, not over multiple data splits.

## Implementation Review

### Data And Splits

`benchmark/data.py` creates stratified masks from labels using NumPy and overwrites any existing DGL masks. This is good for reproducibility because all models use the same masks.

Potential issue:

- Stratification floors per-class counts. This is fine for YelpChi but should be checked for very small or highly imbalanced future datasets.

### Graph Generation

`benchmark/run.py` builds a source graph, converts it to the canonical graph, records the clean graph, then generates scenario variants.

Good points:

- Graph generation uses temporary CSV files and replaces final manifests only after completion.
- No-op variants can reference the base graph instead of duplicating graph binaries.
- `variant_audit.csv` is generated alongside `graph_variants.csv`.

Potential issue:

- The clean row uses `graph_seed = split_seed`, while perturbed rows use configured graph seeds. This is workable, but semantically confusing because the clean graph has no perturbation seed. A future schema could use a dedicated clean seed value such as `0` plus a separate `split_seed` column, or document the current convention clearly.

### Scenario Implementation

The five implemented scenarios are conceptually aligned with the papers:

- oracle heterophily rewiring: label-disagreement stress
- non-oracle heterophily rewiring: feature-cluster proxy stress
- feature camouflage: fraud features become normal-like
- relation camouflage: fraud nodes connect to normal nodes and optionally lose suspicious edges
- uniform noise edges: density and neighborhood clutter

Good points:

- Labels and masks are protected invariants.
- Node count is preserved.
- Rewiring records selected versus actual rewired edges.
- Edge sampling rejects self-loops, existing edges, and duplicates when configured.
- Non-oracle heterophily avoids true labels in target construction.

Limitations:

- The non-oracle path uses a simple two-means feature partition. This is a useful baseline perturbation, but not a learned attacker or a strong fraud-behavior model.
- Relation camouflage currently adds a small fixed number of edges per selected fraud node. In dense graphs, this may be too weak.
- Edge perturbations do not model temporal plausibility or relation-specific business constraints.

### Metrics

The metrics are appropriate for imbalanced binary fraud detection:

- ROC-AUC
- Average Precision
- F1-macro

Validation-based thresholding is implemented in `benchmark/metrics.py` and used by all integrated model stages. This is a strong point.

One reporting caution:

- F1-macro is threshold-sensitive, so always disclose that the threshold was selected on validation by best F1.

### Baselines

The MLP and GraphSAGE baselines are useful and correctly scoped.

Interpretation:

- MLP should be affected by feature camouflage.
- MLP should not be affected by graph-only perturbations under `train_clean_eval_all`, because it uses node features only.
- GraphSAGE should be more sensitive to heterophily/noise than MLP if message passing is harmed.

The current result pattern follows this sanity check.

### PMP Integration

The PMP adapter creates `label_unk` using labels only for train nodes and marks others as unknown. That is important because PMP uses label status during message passing.

Good points:

- On Windows, PMP uses `num_workers = 0`.
- Benchmark config overrides are merged into the PMP YAML.
- Feature normalization is restored after training/evaluation.
- Validation thresholding is used.

Limitations:

- The current homogeneous graph view means PMP is not tested in a fully relation-rich setting.
- Hparams are benchmark-side feasibility settings, not necessarily paper-faithful settings.

### SEC-GFD Integration

The SEC-GFD adapter avoids test-threshold leakage and includes benchmark-side hparams.

Good points:

- Device selection is safer than the original hardcoded CUDA behavior.
- The contrastive-like loss indexing is fixed relative to the original code comment.
- Hyperparameters are validated.
- Validation thresholding is used.

Implementation concern:

- In `benchmark/secgfd_stage.py`, validation monitoring happens before `opt.step()`, and `best_state` is copied before the optimizer update. This means the saved checkpoint can correspond to the pre-update model for that epoch, not the model after the training step that was just computed. This is not data leakage, but it can make early stopping suboptimal. A cleaner order is: train step, `opt.step()`, validation forward, checkpoint post-update state.

### Shift Stage

The shift protocol is conceptually important and implemented in `benchmark/shift_stage.py`.

Good points:

- Each model is trained once on the clean graph per dataset/split/training seed.
- The artifact is evaluated across all selected variants.
- `train_graph_ref` points to the clean graph.

Concrete bug:

- `benchmark/shift_stage.py` calls `truncate_error_message(...)` in error paths but does not import it from `benchmark.results`. This will cause a `NameError` if clean graph loading or clean training fails. The current tests do not hit that error path.

Recommended fix:

```python
from .results import (
    PROTOCOL_TRAIN_CLEAN_EVAL_ALL,
    ensure_results_csv,
    load_completed_keys,
    make_run_key,
    truncate_error_message,
    write_result_row,
)
```

### Plotting And Reporting

The plotting stage is one of the better parts of the repository.

Good points:

- It keeps protocol visible.
- It joins performance rows with audit rows.
- It writes missing/error diagnostics.
- It exports summary curves, max-stress drops, robustness scores, and audit curves.
- It supports bootstrap confidence intervals.

Important caution:

- Bootstrap intervals over only a few runs should be described as descriptive uncertainty, not strong statistical inference.

## Current Result Interpretation

The current v3 run appears to be a completed shift-protocol run only.

Clean ROC-AUC means from `plots/summary_curves.csv`:

- MLP: about 0.780
- GraphSAGE: about 0.733
- PMP: about 0.848
- SEC-GFD: about 0.815

Under max oracle heterophily stress:

- MLP drop: about 0.000
- GraphSAGE drop: about 0.006
- PMP drop: about 0.082
- SEC-GFD drop: about 0.284

Under max non-oracle heterophily stress:

- MLP drop: about 0.000
- GraphSAGE drop: about 0.005
- PMP drop: about 0.030
- SEC-GFD drop: about 0.006

Under max feature camouflage:

- MLP drops from about 0.780 to 0.700
- GraphSAGE drops from about 0.733 to 0.669
- PMP drops from about 0.848 to 0.765
- SEC-GFD drops from about 0.815 to 0.725

Under max relation camouflage:

- Drops are near zero for all models.

Under max uniform noise:

- MLP is unchanged.
- GraphSAGE changes very little.
- PMP drops modestly.
- SEC-GFD changes very little.

Interpretation:

- MLP invariance under graph-only stress is expected and validates the feature-only baseline.
- Feature camouflage affects all models because all models use features.
- Oracle heterophily is much harsher than non-oracle heterophily, especially for SEC-GFD.
- Relation camouflage likely needs stronger calibration because the audit shift is small relative to YelpChi density.
- These conclusions are for `train_clean_eval_all` only.

## Is Anything Illogical?

There is no fatal conceptual contradiction. The benchmark is logically coherent as a controlled stress benchmark.

The potentially illogical parts are mostly interpretation risks:

1. Calling oracle perturbations realistic attacks would be illogical.
2. Comparing severity values directly across families would be illogical.
3. Claiming relation-aware model robustness while evaluating only a homogeneous graph view would be too strong.
4. Treating one split as stable evidence of model ranking would be too strong.
5. Reporting current v3 results as if both protocols were run would be incorrect.
6. Interpreting no relation-camouflage degradation as proof of robustness would be weak unless the audit shows a meaningful neighborhood-composition shift.

## Reproducibility And Artifact Issues

The code design is reproducible, but the current local artifact state is not fully self-contained.

Observed issues:

- Existing `runs/gfd_robustness_benchmark_v3/results.csv` contains absolute paths from another checkout location.
- The local `runs/gfd_robustness_benchmark_v3/graphs/.../graph.bin` files are not present in this workspace.
- DGL is not installed in the current Python environment.

This means the current CSVs are useful for inspection, but the benchmark should be regenerated locally before final submission if you want full artifact traceability.

Recommended reproducibility improvement:

- Store graph paths relative to the run directory in future CSVs, or add a path-normalization helper in graph loading.

## Priority Fixes Before Final Report

1. Fix the `truncate_error_message` import in `benchmark/shift_stage.py`.
2. Use a Python 3.10 or 3.11 virtual environment with the DGL/PyTorch versions from the README.
3. Rerun `graphs` locally so graph binaries exist under the current workspace.
4. Run both protocols if the report claims both:
   - `train_on_variant`
   - `train_clean_eval_all`
5. Add at least one extra split config if time allows.
6. Calibrate relation camouflage so the audit metric shows a meaningful shift.
7. Disclose that PMP and SEC-GFD use reduced-cost benchmark hparams.
8. Keep oracle labels visible in every table and plot used in the report.

## Recommended Next Phase

### Phase 1: Make The Current Benchmark Defensible

Goal: produce a clean, reproducible final-project run.

Actions:

- Create a Python 3.11 environment.
- Install DGL/PyTorch as documented.
- Regenerate v3 graphs locally.
- Run `matrix` under `train_clean_eval_all`.
- Run `matrix` under `train_on_variant` if compute allows.
- Run plots with `--ci`.
- Confirm `missing_or_error_runs.csv` is empty or explain every missing/error row.

This phase is enough for a good final project.

### Phase 2: Strengthen Scientific Claims

Goal: reduce evaluation fragility.

Actions:

- Add a new config, not by editing frozen configs, with 3 data splits.
- Keep the same graph and training seeds.
- Compare whether model rankings change across splits.
- Report split-level and cross-split summaries.

This directly addresses the evaluation instability concerns from the Pitfalls paper.

### Phase 3: Improve Stress Calibration

Goal: make perturbations more interpretable.

Actions:

- Add a stronger relation-camouflage config.
- Try `camouflage_edges_per_node` values such as 5, 10, or degree-relative values.
- Try feature camouflage with partial blending, for example `gamma = 0.25, 0.5, 0.75, 1.0`.
- Add a density-normalized noise scenario such as random neighbors per node.
- Prefer conclusions based on realized audit metrics, not raw severity.

This would make the stress families more balanced.

### Phase 4: Optional Research Extensions

Only do these if the core project is already complete:

- Add CARE-GNN for camouflage-specific comparison.
- Add GAGA if preprocessing effort is manageable.
- Add Amazon as a second dataset with only clean and max-stress settings.
- Add temporal or pseudo-temporal splits if a dataset supports timestamps.
- Add robustness scores based on worst-case performance, not only area under curves.

## Suggested Final Report Thesis

A strong final report can argue:

> This work builds a reproducible static robustness benchmark for graph fraud detection. It shows that model robustness depends strongly on the type of graph stress and on whether the model is evaluated under test-time shift or retraining on stressed graphs. Feature camouflage affects both feature-only and graph models, while graph-only perturbations separate feature reliance from message-passing sensitivity. Oracle and non-oracle perturbations produce different levels of degradation, so robustness claims must disclose perturbation construction and realized audit strength.

This thesis is accurate, technically meaningful, and aligned with the implementation.

## Final Assessment

The project is feasible, meaningful, and close to report-ready. The benchmark is not fundamentally illogical. Its main value is controlled comparative robustness evaluation, not realism.

The key to making it academically strong is disciplined framing:

- Be precise about what is measured.
- Keep protocols separate.
- Use audit evidence.
- Disclose oracle perturbations.
- Avoid overclaiming from one split.
- Fix the small implementation issues before final runs.

If those points are handled, this is a solid final project with a clear connection to the reviewed research papers and enough engineering depth to be credible.
