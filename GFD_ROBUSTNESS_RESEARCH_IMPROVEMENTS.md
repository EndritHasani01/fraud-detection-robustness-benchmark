# Highest-Value Research Improvements and Implementation Record

## Executive decision

The R5 run does not need more models first. Its execution and provenance are already strong; its limiting factors are the validity of the statistical summaries, the interpretation of oracle graphs, the calibration of two stress scenarios, training-diagnostic visibility, and the absence of independent data splits.

The highest-value changes selected for implementation are:

1. Replace flat IID bootstrapping with seed-aware, paired, trajectory-level reporting.
2. Add explicit oracle claim scopes and exclude oracle rows from operational ranking by default.
3. Export direct protocol contrasts, worst-case performance, and clean-retention summaries.
4. Make relation camouflage degree-relative instead of a fixed `+2` edges per selected node.
5. Freeze the non-oracle feature partition across severity and allocate random noise proportionally to relation size in the prospective design.
6. Add a new three-split/five-training-seed v4 config without modifying frozen v1–v3 definitions.
7. Record early-stopping telemetry in every new successful result row.
8. Strengthen config validation so duplicate seeds, identifiers, invalid splits, and ambiguous scenario controls fail before expensive execution.
9. Synchronize the clean Kaggle notebook with repository modules, the proven `MPLBACKEND=Agg` hotfix, new report outputs, and an exact module-source test.

These changes are implemented in the repository. They do not retroactively manufacture new model results. The corrected reporting layer can reanalyze R5’s existing 504 raw rows without GPU training; the new v4 perturbation and split design requires a prospective run.

Large extensions—native relation-aware PMP evaluation, a second dataset, nested perturbation plans for every scenario, CARE-GNN/GAGA integration, and row-level cryptographic run-spec enforcement—remain valuable future work. They are not bundled into the immediate patch because each changes the scientific object or execution architecture enough to require a separately reviewed experiment revision.

## 1. Decision rubric

Candidates were scored using four questions:

- **Scientific value:** Does it change whether a conclusion is valid or interpretable?
- **Evidence:** Is the need visible in R5 rather than speculative?
- **Implementation confidence:** Can the change be specified and tested without silently changing frozen results?
- **Cost/risk:** Does it require a large rerun, upstream rewrite, or new confound?

| Priority | Proposal | Scientific value | Evidence in R5 | Cost/risk | Decision |
|---:|---|---|---|---|---|
| P0 | Seed-aware paired reporting | Critical | Exact-zero MLP effect gets nonzero legacy CI | Low; no GPU rerun | Implemented |
| P0 | Oracle claim-scope guards | Critical | SEC-GFD oracle retraining AP ≈ 1 | Low; no GPU rerun | Implemented |
| P0 | Protocol contrast and worst/retention tables | High | Retraining gains range from −0.124 to +0.841 AP | Low | Implemented |
| P1 | Degree-relative relation camouflage | High | R5 adds only one net edge per selected node | Moderate; next run needed | Implemented prospectively |
| P1 | Fixed non-oracle partition across severity | High | Pseudo-positive rate changes sharply with severity | Moderate; next run needed | Implemented prospectively |
| P1 | Proportional relation noise | High | Uniform allocation over-inflates small relations | Moderate; next run needed | Implemented prospectively |
| P1 | Multiple split replication | High | All cross-split tables have `n_splits=1` | Compute/storage | New v4 config implemented |
| P1 | Early-stopping telemetry | High | SAGE duration/AP correlation ≈ 0.808 | Low | Implemented |
| P1 | Strict config validation | High operational value | Duplicate axes would create ambiguous run keys | Low | Implemented |
| P1 | Notebook/source synchronization | High reproducibility value | Clean notebook missed the proven plotting hotfix | Low | Implemented |
| P2 | Strict nested perturbation prefixes | High | R5 severities redraw variants | High design change | Deferred to a dedicated revision |
| P2 | Native multi-relation evaluation | High | PMP receives only `_E` | High adapter/cache redesign | Deferred |
| P2 | Row-level run-spec hashes | Medium-high | Bundle provenance is strong, row provenance thinner | Cross-cutting schema work | Deferred |
| P3 | Add CARE-GNN/GAGA now | Low before validity fixes | No evidence another model fixes current inference | High integration risk | Rejected for this phase |
| P3 | Increase bootstrap resamples | None by itself | Dependence, not Monte Carlo count, is the defect | Low but misleading | Rejected |

## 2. Improvement 1: seed-aware statistical reporting

### Problem

R5 has a crossed random design. Each positive-severity point contains two graph seeds and three training seeds. Under the shift protocol, rows sharing a training seed also share one trained clean artifact. Severity points are repeated observations along a trajectory.

The legacy implementation flattened six rows as if they were independent, independently resampled clean and stress for drop intervals, and independently resampled each severity when estimating curve uncertainty. This destroys pairing and confuses optimization variation with perturbation-effect uncertainty.

The clearest symptom is the MLP. Its prediction is exactly unchanged by graph-only perturbations for every matching training seed, yet the legacy AP-drop interval is approximately `[-0.063, +0.063]` because independent resampling allows different clean seed mixtures on the two sides.

### Implemented design

`benchmark/plots_stage.py` now constructs one value per `(training_seed, graph_seed)` cell and exposes the design explicitly.

Point summaries:

- Collapse accidental duplicate ledger rows within a seed cell.
- Preserve empirical sample SD across seed cells.
- Bootstrap the training-seed and graph-seed axes, never the flattened row list.
- Export `n_training_seeds`, `n_graph_seeds`, `n_seed_cells`, and `ci_method`.

Clean-minus-stress drops:

- Aggregate the clean value by training seed.
- Pair it to each stressed graph realization with the same training seed.
- Calculate empirical paired-cell deltas.
- Resample training and graph axes together.
- Export `n_paired_seed_cells` and `n_paired_training_seeds`.

Curve summaries:

- Build a complete trajectory for each seed cell.
- Reuse the matching clean score at severity zero.
- Compute trapezoidal AUC and severity-normalized mean performance per trajectory.
- Aggregate and bootstrap those trajectory values, preserving dependence across severity.
- Exclude incomplete trajectories rather than combining mismatched points.

Cross-split summaries:

- Treat each split-level mean as the independent cross-split unit.
- Export a split-bootstrap method label.
- With only one split, treat the row as schema/descriptive output; it does not create inferential replication.

### New and expanded artifacts

- `plots/summary_curves.csv`
- `plots/performance_drop_max_stress.csv`
- `plots/robustness_scores.csv`
- Their cross-split equivalents

All now carry explicit claim and seed-design metadata. The R5 raw means remain unchanged; interval and SD semantics are corrected.

### Acceptance evidence

The disposable R5 reanalysis reconstructed all 504 rows with zero completeness errors. For MLP under every graph-only scenario and both protocols:

```text
paired AP drop mean = 0
paired AP drop SD   = 0
paired AP drop CI   = [0, 0]
```

This is the required mechanistic invariant.

## 3. Improvement 2: oracle claim-scope guards

### Problem

`oracle_labels=true` is necessary but insufficient metadata. Protocol determines how privileged information enters the model’s environment:

- In `train_clean_eval_all`, true labels construct only the evaluation variant. This is an oracle shift-sensitivity diagnostic.
- In `train_on_variant`, true labels—including test-node labels—construct the graph used for training. This is privileged-label training exposure.

SEC-GFD’s AP near 1.0 under oracle retraining is exactly the result most likely to be misreported as a robustness victory.

### Implemented design

Every report-facing row now contains:

- `claim_scope`
- `operational_ranking_eligible`

The scopes are:

| Oracle status | Protocol | Claim scope | Operational ranking? |
|---|---|---|---:|
| Non-oracle | Either | `non_oracle_controlled_stress` | Yes, within the static benchmark scope |
| Oracle | `train_clean_eval_all` | `oracle_shift_sensitivity_diagnostic` | No |
| Oracle | `train_on_variant` | `oracle_privileged_training_diagnostic` | No |
| Oracle protocol contrast | Both | `oracle_protocol_comparison_diagnostic` | No |

The data are not deleted or clipped. The perfect SEC-GFD result remains visible, but its allowed interpretation travels with the row.

### Acceptance evidence

R5 reanalysis assigns all oracle summary rows to one of the two diagnostic scopes and marks them `operational_ranking_eligible=False`. Non-oracle rows retain controlled-stress eligibility.

## 4. Improvement 3: direct protocol contrasts and decision-useful robustness tables

### Problem

Separate protocol plots make adaptation effects visually inferable but do not quantify them. “Smallest drop” tables also reward the graph-blind MLP and hide absolute detector quality.

The old curve AUC is an absolute average metric over the scenario-specific severity range. It is useful, but it mainly rewards a high clean baseline and should not be interpreted as a universal clean-normalized robustness score.

### Implemented artifacts

#### `plots/protocol_contrasts.csv`

For every matching dataset, split, scenario, severity, model, and metric:

```text
contrast = train_on_variant − train_clean_eval_all
```

The calculation pairs exact graph/training cells and uses the seed-aware bootstrap. The table includes both sides’ means, empirical contrast SD, interval, seed counts, oracle scope, and the literal contrast definition.

The R5 reanalysis produces 180 rows. Maximum-stress AP examples are:

| Scenario/model | Contrast | Seed-aware 95% descriptive interval |
|---|---:|---:|
| Oracle heterophily / SEC-GFD | +0.8410 | `[+0.8261, +0.8516]` |
| Oracle heterophily / SAGE | +0.2690 | `[+0.2380, +0.3018]` |
| Noise / SAGE | −0.1241 | `[−0.1946, −0.0093]` |
| Non-oracle rewiring / SAGE | −0.0692 | `[−0.1830, −0.0031]` |
| Non-oracle rewiring / PMP | +0.0199 | `[+0.0099, +0.0351]` |

These remain descriptive intervals from one split, not formal population-level significance results.

#### `plots/worst_case_performance.csv`

For every dataset, split, model, protocol, and metric, the reporting stage selects the lowest configured nonzero point mean and exports:

- scenario and severity of the selected worst point;
- worst absolute performance;
- clean performance;
- paired clean-to-worst drop;
- stressed/clean retention fraction;
- seed-aware uncertainty and seed counts;
- oracle claim scope and ranking eligibility;
- the explicit selection rule.

This table prevents a single robustness number from hiding its originating scenario. If the worst point is oracle, the row is visibly ineligible for operational ranking.

#### Interpretation rule

No report should choose a model from drop alone. A defensible comparison shows, side by side:

1. clean AP;
2. stressed AP;
3. absolute paired drop;
4. relative retention;
5. oracle/operational scope;
6. seed counts and split count.

## 5. Improvement 4: density-relative relation camouflage

### Problem

R5’s `camouflage_edges_per_node=2` and `remove_suspicious_ratio=0.5` add two edges and remove one for each selected fraud node. The mean selected node already has roughly 167 outgoing neighbors. A net `+1` edge is not a credible high-severity relation-camouflage test on this graph.

### Implemented scenario control

`add_relation_camouflage_edges` now supports an optional bounded degree-relative budget:

- `camouflage_edge_degree_ratio`
- `camouflage_min_edges_per_node`
- `camouflage_max_edges_per_node`

For a selected node with observed out-degree `d`, the requested addition budget is:

```text
ceil(d × camouflage_edge_degree_ratio)
```

then clamped to the configured minimum and maximum. The legacy fixed `camouflage_edges_per_node` path remains unchanged for v1–v3. Config validation rejects setting both policies simultaneously.

The audit info now records:

- budget mode (`fixed` or `degree_relative`);
- degree ratio and bounds;
- mean requested camouflage edges per selected node;
- requested/actual additions and removals;
- before/after local neighbor ratios and degree.

### Prospective v4 setting

The new v4 config uses:

```json
{
  "camouflage_edge_degree_ratio": 0.25,
  "camouflage_min_edges_per_node": 4,
  "camouflage_max_edges_per_node": 64,
  "remove_suspicious_ratio": 0.5
}
```

This is a preregistered prospective policy, not a post-hoc reinterpretation of R5. The graph stage must be run and its realized neighbor-composition shift inspected before expensive training begins.

## 6. Improvement 5: stable non-oracle partition and proportional noise

### 6.1 Severity-stable feature partition

R5 fits the two-means feature partition from a severity-dependent RNG stream. It therefore changes the latent partition as well as the rewire fraction.

The scenario now supports:

```json
"fixed_feature_partition_across_severity": true
```

When enabled, the two-means partition uses a dedicated RNG stream keyed by scenario, method, graph seed, and parameters but not severity. Rewiring still uses the historical severity-specific stream. The audit records whether the partition scope is `fixed_per_graph_seed` or `severity_specific`.

This removes the largest non-oracle curve confound while preserving v1–v3 deterministic behavior by default. Strictly nested edge selections across all severities remain deferred because rejection policies and vectorized target sampling require a first-class persisted perturbation plan, not only a seed change.

### 6.2 Relation-proportional noise

R5 chooses a relation uniformly for each new edge. If relations have unequal base sizes, equal expected additions distort their composition.

The noise scenario now supports:

```json
"relation_allocation": "proportional"
```

This samples relations according to existing edge counts. The legacy `uniform` path remains the default for v1–v3. The policy is recorded in scenario audit information and validated before graph generation.

## 7. Improvement 6: independent split replication config

### Problem

R5 has one split, so every cross-split summary has `n_splits=1`. The literature review explicitly warns that GNN rankings can change across data partitions.

### Implemented config

[configs/exp_yelpchi_v4_multisplit.json](configs/exp_yelpchi_v4_multisplit.json) adds three preregistered stratified splits:

| Split | Seed | Train/validation/test |
|---|---:|---|
| `s0` | 717 | 40% / 20% / 40% |
| `s1` | 1729 | 40% / 20% / 40% |
| `s2` | 3253 | 40% / 20% / 40% |

It uses five training seeds and one graph seed.

The one-graph-seed choice is deliberate. R5’s 31-row graph ledger consumes approximately 5.7 GiB after setup, while the measured Kaggle session has about 14.2 GiB free at that point. Three splits with two graph seeds would likely exceed the writable-disk budget because graph binaries duplicate masks and variants per split. The v4 config therefore prioritizes independent data partitions and more optimization repeats while keeping approximately the same number of unique cached graphs per three-split experiment as a feasible Kaggle run.

This trade-off must remain visible: v4 improves split uncertainty but weakens perturbation-realization uncertainty. If storage or sequential split processing is later implemented, add more graph seeds in a new config rather than mutating v4.

Expected v4 scale with current positive severities:

```text
3 splits × (1 clean + 5 scenarios × 2 positive severities)
× 5 training seeds × 4 models × 2 protocols
= 1,320 result rows
```

The graph ledger contains severity-zero references in addition to evaluated states.

## 8. Improvement 7: early-stopping telemetry

### Problem

R5 SAGE has a strong duration/performance split, but the result schema cannot distinguish a bad early stop from another failure mode.

### Implemented result fields

New successful rows record:

- `epochs_trained`
- `best_epoch`
- `best_validation_monitor`
- `validation_monitor`
- `stopping_reason`

The baseline, PMP, and SEC-GFD artifacts carry these values into both protocols. Under the shift protocol, all evaluation rows produced by one clean-trained artifact repeat the same training diagnostics, which correctly describes their shared training history.

Historical CSV migration remains backward compatible: old rows receive blank diagnostic fields. The new fields do not alter the scientific run key.

For future SAGE analysis, report at minimum:

- AP by training seed;
- epochs trained and best epoch;
- best validation ROC-AUC;
- stop reason;
- whether a collapsed seed repeats across scenarios or splits.

## 9. Improvement 8: fail-fast config validation

Expensive graph/model runs should not begin with an ambiguous experiment definition. Validation now rejects:

- duplicate training or graph seeds;
- duplicate dataset, split, model, or scenario identifiers;
- duplicate severity values;
- scenarios without explicit severity zero;
- non-integer or duplicate split seeds;
- nonpositive train/validation shares or train+validation ≥ 1;
- ambiguous fixed and degree-relative camouflage budgets;
- invalid camouflage bounds;
- invalid `relation_allocation` values;
- non-boolean severity-stable-partition controls.

All existing frozen configs continue to validate. The stricter checks protect run-key uniqueness and split independence before files are generated.

## 10. Improvement 9: notebook/source integrity

### Problem

The executed latest-working notebook contains a proven plotting hotfix that the clean source notebook did not: it temporarily sets `MPLBACKEND=Agg` for the benchmark plotting subprocess. The notebook also embeds 22 benchmark modules, but the previous test only checked that embedded hashes matched the notebook’s own hash dictionary; it did not prove equality with the repository modules.

### Implemented workflow

- [tools/sync_kaggle_notebook.py](tools/sync_kaggle_notebook.py) synchronizes every `benchmark/*.py` body into the clean notebook.
- It verifies that the embedded and local module sets are identical.
- It regenerates the notebook’s embedded hash contract.
- `--check` fails without writing if the notebook is stale.
- The notebook regression test now compares every embedded body exactly with the repository source.
- The proven noninteractive Matplotlib backend guard is carried into the clean revision.
- Required plot receipts include the new protocol-contrast and worst-case artifacts.

The executed R5 notebook and bundle remain historical evidence. The clean notebook is the next rerunnable source revision; it must not be confused with the already executed R5 outputs.

## 11. Verification strategy

The implementation follows a generate–evaluate–refine loop:

### Unit and regression checks

- Config schema and frozen-config validation.
- Degree-relative camouflage budgets and deterministic behavior.
- Severity-stable non-oracle feature partition.
- Proportional noise policy recording.
- Result-schema telemetry migration.
- Seed-aware point, paired-drop, paired-curve, and paired-protocol summaries.
- Oracle claim-scope propagation.
- Exact MLP graph-invariance interval.
- Notebook parse, embedded hash contract, and exact source equality.

### R5 reanalysis check

A disposable copy of the R5 raw artifacts is processed with the new plotting stage. Acceptance conditions are:

- 504/504 expected rows present;
- zero missing/error rows;
- 180 protocol contrasts;
- 24 global worst-case rows (one per model/protocol/metric);
- exact zero MLP graph-only paired effects;
- oracle rows marked ineligible for operational ranking;
- all report CSVs parse and contain explicit seed and CI metadata.

### Prospective v4 gate

Do not launch the full four-model matrix immediately. Run:

```powershell
py -m benchmark.run --config configs/exp_yelpchi_v4_multisplit.json --stage graphs
```

Then inspect, per split:

- requested versus realized counts;
- relation-camouflage local neighbor-ratio and degree shifts;
- non-oracle fixed-partition rate and true-label heterophily shift;
- per-relation noise growth;
- graph-cache disk usage;
- invariant checks and completeness.

Only after the graph audit is acceptable should both training protocols be run.

## 12. Prospective v4 interpretation contract

Before seeing model outcomes, the report should commit to these rules:

1. AP is primary; ROC-AUC and macro-F1 are corroborating.
2. Protocols are never averaged.
3. Oracle rows are diagnostics and never operational rankings.
4. Absolute stressed performance and relative retention are both reported.
5. Scenario families are not combined into one leaderboard.
6. Requested severity is interpreted only beside realized audit metrics.
7. Cross-split claims require all three splits; a partially completed run is labeled incomplete.
8. A failed or weak scenario realization is reported as invalid/inconclusive, not as model robustness.
9. Hyperparameters are frozen before stressed test outcomes are inspected.
10. New configs are created for any material design change.

## 13. Deferred high-value research

### 13.1 Strictly nested perturbation plans

The ideal design constructs one maximum-severity plan per scenario and graph seed, then applies prefixes at lower severities. This guarantees that severity changes only dose. Implementing it correctly requires persisted plans that include selected nodes/edges, target candidates, relation allocations, and rejection outcomes. Merely removing severity from the RNG seed is not sufficient because sampling algorithms and rejection rounds can still change earlier assignments.

This should be a dedicated v5 design with plan hashes in graph metadata and tests proving subset relations across every severity.

### 13.2 Native heterogeneous model views

PMP should ultimately receive YelpChi’s three relations. A rigorous design would cache one logical variant with synchronized heterogeneous and homogeneous physical views, record the view and relation count per result, and add an R-GCN-style relational baseline. This is a meaningful research extension but changes model fidelity and cache architecture enough to require a separate experiment.

Until then, report `pmp` and `secgfd` as integrated homogeneous/repaired adapters, not exact paper reproductions.

### 13.3 Row-level provenance and fail-closed resumability

The bundle manifest already has strong experiment-level provenance. A future schema can additionally record `run_spec_hash`, config hash, graph hash, model-recipe hash, adapter version, code fingerprint, run-attempt ID, split seed, and nullable perturbation seed in every row. Resumption should fail if those do not match the active request.

This is valuable before long-lived or multi-machine production runs; it is less urgent than correcting the scientific summaries for the current course project.

### 13.4 Second dataset

Amazon or another graph-fraud dataset would improve external validity. It should be added only after scenario calibration and model-view provenance are stable. Otherwise, a second dataset multiplies ambiguous results rather than strengthening the study.

## 14. Changes intentionally not made

- Frozen v1–v3 configs were not edited.
- The R5 result bundle was not overwritten; reanalysis uses a disposable derived directory.
- The SEC-GFD oracle result was not removed or clipped.
- CARE-GNN and GAGA were not added before validity fixes.
- No post-hoc significance test was added.
- Bootstrap count was not increased as a substitute for correct experimental units.
- Protocols, oracle/non-oracle scenarios, and scenario families are not averaged into one score.
- Raw severity values are not compared across families.
- Concurrent lane times are not promoted to model-speed benchmarks.

## 15. Recommended execution order

### Immediate, no GPU retraining

1. Run the full local test suite.
2. Run the clean notebook synchronization check.
3. Reanalyze R5 from raw `results.csv` and audit files with the corrected plots stage.
4. Use the corrected tables and this analysis in the final report.

### Next Kaggle run

1. Upload/run the synchronized clean notebook if reproducing the v3 matrix, or use the v4 config in the repository runner.
2. Execute only v4 graph generation first.
3. Review realized scenario strength and disk use.
4. Freeze the graph audit decision.
5. Run `train_on_variant` and `train_clean_eval_all` for all selected models.
6. Generate seed-aware plots with `--ci`.
7. Require empty `missing_or_error_runs.csv`.
8. Review early-stopping telemetry before interpreting SAGE.
9. Report split-level and cross-split results separately.

## 16. Definition of done

The implemented phase is complete when:

- all tests pass;
- the clean notebook parses and exactly embeds current benchmark sources;
- R5 reanalysis has 504 complete rows and corrected MLP invariant intervals;
- protocol contrast and worst/retention artifacts are generated;
- oracle claim scopes are present in every relevant summary;
- the new v4 config validates;
- legacy configs remain unchanged and valid;
- new scenario controls are deterministic and regression-tested;
- new result rows retain training diagnostics;
- the two research Markdown files distinguish executed evidence from prospective design.

The prospective v4 research run itself is not complete until its graph audit and all 1,320 expected training rows are executed. Code readiness is not reported as empirical confirmation.

## Final recommendation

The project should present R5 as a successful, complete controlled benchmark run with carefully bounded conclusions, use the corrected seed-aware reanalysis for uncertainty, and make the next expensive run a confirmatory multi-split experiment after graph-audit calibration.

That sequence has higher research value than adding another model: it turns the current result from a technically impressive single run into a methodologically explicit study where every performance claim is tied to a valid experimental unit, realized perturbation, protocol, and label-access scope.
