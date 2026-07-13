# Full Results Analysis: YelpChi Robustness Benchmark v3, Kaggle R6

## Executive verdict

This document is the report-facing analysis of the executed notebook [`KAGGLE_DUAL_T4_RESEARCH_RUN_with_outputs_latest_v4_working.ipynb`](KAGGLE_DUAL_T4_RESEARCH_RUN_with_outputs_latest_v4_working.ipynb) and the evidence bundle [`gfd-robustness-v3-report-r6/`](gfd-robustness-v3-report-r6/). It is intentionally narrower than a final paper: it establishes what was run, what the results show, which interpretations are defensible, and which conclusions must remain out of scope.

The main result is not a single universal robustness winner. It is a set of protocol- and scenario-specific findings:

1. **PMP is the strongest detector in absolute terms in this run.** It has the best clean ROC-AUC, Average Precision (AP), and macro-F1 under both protocols. At maximum configured stress it has the highest AP in 8 of the 10 protocol-by-scenario comparisons.
2. **The configured oracle-selected complete feature replacement is the broadest consistent failure mode.** Under clean-training shift, maximum oracle feature camouflage reduces AP by about 20.6% to 21.5% for every model. The degradation is also visible in ROC-AUC and macro-F1.
3. **Oracle heterophily exposes the largest protocol interaction.** At severity 0.30, SEC-GFD reaches essentially perfect AP when retrained on the label-constructed graph, but only 0.1570 AP when trained on the clean graph and evaluated on that variant. The paired protocol contrast is +0.8430 AP. This is a privileged-label topology diagnostic, not production robustness.
4. **PMP remains best on the non-oracle stresses, even when it is not the most invariant.** Under clean-training shift, its AP drops by 0.0783 for non-oracle rewiring and 0.0499 for edge noise, yet its stressed AP remains higher than every alternative.
5. **The tested relation-camouflage intervention is too weak to answer the intended question.** At maximum severity it changes only +2,003 net edges in an 8.05-million-edge graph and raises the selected fraud-to-normal neighbor ratio by about 0.007. Near-flat performance is therefore inconclusive, not evidence that the models resist relation camouflage.
6. **The non-oracle “heterophily” scenario is structurally strong but weak in its named target.** It rewires 2.42 million edges at severity 0.30, but true-label heterophily rises by only about 0.0051. Its conclusions concern feature-partition-driven rewiring, not a strong realized true-label heterophily shift.
7. **GraphSAGE retraining is unstable.** In the train-on-variant protocol, 20 early-stopped runs average 0.2223 AP, while 43 runs reaching 100 epochs average 0.4524 AP. Epoch count and AP correlate at 0.809. This is a strong diagnostic association, not proof that early stopping alone caused the weak mode.
8. **The run is operationally complete but inferentially small.** All 504 expected rows succeeded, and the corrected summaries respect training-seed/graph-seed pairing. However, there is only one YelpChi split. The intervals are descriptive within-split uncertainty bands, not population-level confidence or formal significance evidence.

> **Report-safe one-sentence conclusion:** On one fixed YelpChi split, PMP was the strongest clean model and usually retained the highest absolute AP under the configured stresses; the configured oracle-selected complete feature replacement caused the most consistent degradation, while oracle heterophily revealed a large adaptation-versus-shift interaction whose near-perfect retrained SEC-GFD result is a privileged diagnostic rather than an operational robustness result.

## 1. Scope, evidence hierarchy, and naming clarification

### 1.1 Primary evidence

The analysis uses the following artifacts in descending order of authority:

1. [`results.csv`](gfd-robustness-v3-report-r6/results.csv): one canonical row per scientific run key.
2. [`variant_audit.csv`](gfd-robustness-v3-report-r6/variant_audit.csv) and [`graph_variants.csv`](gfd-robustness-v3-report-r6/graph_variants.csv): requested and realized perturbation evidence.
3. Seed-aware, protocol-aware derived tables in [`plots/`](gfd-robustness-v3-report-r6/plots/), especially:
   - `summary_curves.csv`;
   - `performance_drop_max_stress.csv`;
   - `robustness_scores.csv`;
   - `protocol_contrasts.csv`;
   - `worst_case_performance.csv`;
   - `performance_audit_join.csv`;
   - `missing_or_error_runs.csv`.
4. [`kaggle_run_manifest.json`](gfd-robustness-v3-report-r6/kaggle_run_manifest.json), [`run_fingerprint.json`](gfd-robustness-v3-report-r6/run_fingerprint.json), training logs, notebook outputs, and artifact receipts.
5. Presentation PNGs, which visualize—but do not replace—the CSV evidence.

All numerical claims below were recomputed or checked against the raw result and audit tables. Protocols, oracle and non-oracle scenarios, and scenario families are never pooled into one leaderboard.

### 1.2 The executed run is R6 of the v3 experiment, not the prospective v4 design

The notebook filename contains `latest_v4_working`, but its execution metadata and config identify:

- notebook revision: `r6`;
- implementation ID: `single-source-v3-r6-2026-07-12`;
- experiment name: `gfd_kaggle_notebook_v3_final_r6`;
- one data split, `s0`, with split seed 717;
- graph seeds 0 and 1;
- training seeds 0, 1, and 2.

Therefore, this analysis calls the run **v3/R6**. It must not be described as the prospective multi-split v4 experiment. The repository’s v4 config is a future confirmatory design and is not evidence from this run.

The final limitations cell also contains the phrase “R5 severity variants.” That is a cosmetic carry-over in the prose; the executed implementation, outputs, and manifest are R6.

### 1.3 Research questions answered by this analysis

The run supports four bounded questions:

1. Which integrated model has the strongest clean performance on the fixed YelpChi split?
2. How does each model behave as each configured stress increases?
3. How different is matched-distribution retraining from clean-training deployment shift?
4. Did each perturbation actually create the type and magnitude of stress its name suggests?

It does not answer whether one model is universally robust, whether the transformations are realistic adaptive attacks, or whether the ranking generalizes to new splits, datasets, institutions, or time periods.

## 2. Execution integrity, completeness, and provenance

### 2.1 Notebook execution evidence

The executed notebook contains 90 cells: 66 code cells and 24 Markdown cells. All 66 code cells have unique, sequential execution counts from 1 through 66. No cell contains an error output, and the final packaging cell printed `FINAL_STATUS=complete`.

The manifest run attempt began at `2026-07-12T13:19:57.622327Z` and completed at `2026-07-12T14:14:08.224049Z`, an elapsed time of 54 minutes 10.6 seconds. Measured from the first code-cell input through the final shell reply, the full notebook took 54 minutes 11.95 seconds, including setup, graph preparation, training, reporting, and packaging.

All 22 embedded `benchmark/*.py` notebook bodies match the hashes recorded in the run fingerprint. The setup recorded 13 fresh receipts, the embedded tests passed 15/15, and both GPU adapter semantic probes succeeded. The completed report pipeline recorded 11 stage receipts.

### 2.2 Reconstructed experiment matrix

The scientific run key is:

```text
(dataset_id, split_id, scenario_id, severity, graph_seed,
 training_seed, model_id, protocol)
```

The expected matrix is reconstructed as follows:

- one clean graph state;
- five scenarios;
- two positive severities per scenario;
- two graph seeds per positive-severity state;
- therefore `1 + (5 × 2 × 2) = 21` evaluated graph states;
- three training seeds, giving `21 × 3 = 63` rows per model/protocol;
- four models, giving `63 × 4 = 252` rows per protocol;
- two protocols, giving `252 × 2 = 504` result rows.

The graph ledger includes severity-zero references for every scenario and graph seed, so it has:

```text
1 clean + (5 scenarios × 3 severities × 2 graph seeds) = 31 rows
```

Observed integrity checks:

| Check | Result |
|---|---:|
| Canonical result rows | 504 |
| Duplicate scientific keys | 0 |
| `status=ok` rows | 504 |
| Missing/error audit rows | 0 |
| Non-finite ROC-AUC/AP/macro-F1 values | 0 |
| Rows per model/protocol | 63 |
| Rows per protocol | 252 |
| Graph ledger rows | 31 |
| Variant-audit rows | 31 |
| Generated plot PNGs | 98 |
| Presentation figures | 3 |

The clean-training shift protocol contains only 12 actual training artifacts—four models times three training seeds—because each clean-trained artifact is evaluated over all 21 graph states. Its 252 result rows are evaluations, not 252 independent model fits.

The train-on-variant protocol fits one model per row, so the run contains `252 + 12 = 264` actual model fits in total.

### 2.3 Artifact dimensions

| Artifact | Rows × columns | Purpose |
|---|---:|---|
| `results.csv` | 504 × 32 | Raw metrics, run identity, graph statistics, and training telemetry |
| `graph_variants.csv` | 31 × 18 | Cached-graph ledger |
| `variant_audit.csv` | 31 × 63 | Requested/realized scenario evidence |
| `summary_curves.csv` | 360 × 20 | Seed-aware metric points and intervals |
| `performance_drop_max_stress.csv` | 120 × 28 | Paired clean-minus-maximum-stress deltas |
| `robustness_scores.csv` | 120 × 22 | Complete-trajectory integrals and average curve performance |
| `protocol_contrasts.csv` | 180 × 25 | Paired train-on-variant minus clean-train contrasts |
| `worst_case_performance.csv` | 24 × 31 | Discrete configured worst point, drop, and retention |
| `performance_audit_join.csv` | 504 × 38 | Row-level performance joined to realized stress |
| `missing_or_error_runs.csv` | 0 × 9 | Completeness diagnostic |

### 2.4 Fingerprints and bundle verification

| Item | SHA-256 |
|---|---|
| Executed notebook | `fb4c726d6924b69e0c81cf12ae89cee6aaaf319ab36cff8fee4b3a746b64c690` |
| Frozen inline config | `867ce9ae1685b69a69381026001ab562ce2865a2bc805d7688f883badf16eb9f` |
| Canonical `results.csv` | `0053f8bfe1cf7d35164aefc658ce0d67151fda8c626eb5465229b39e12fc6035` |
| Canonical run fingerprint | `a65576e0be6ef425aa22e8c0f4c4dbeb0bd2ea2aaf49080a89086d976a2c781f` |
| Local R6 report ZIP | `62a4b1e6df161dfeb69b6e0fbad7e283cc318b5b2ffdc0bf73f23aac757c368b` |

Recorded source-data receipts are:

| Dataset artifact | SHA-256 |
|---|---|
| `yelp.zip` | `3a31296b951e6c8158dddb783ff2c62709fa987d106bbe21e59f8e119053cec3` |
| `YelpChi.mat` | `fedb35a8fa539b27866244d3515a47a76b20080cdacb33112da3458fd2487b42` |
| DGL Yelp graph cache | `4343d6544b2532b2330fd35a3b9e72fbbaff66fa35fcb40131e84f8824f49bc9` |

All 20 artifact hashes recorded by the report receipt and all three presentation-figure hashes match the extracted files. The ZIP has 131 members, passes `testzip()`, and every non-manifest ZIP member matches the corresponding extracted file byte-for-byte.

The extracted folder contains the deliberately internal bundle manifest. The external post-ZIP manifest, which would contain the ZIP size and hash without self-reference, was not supplied; the ZIP hash above was independently calculated locally.

### 2.5 What the light bundle excludes

The report bundle intentionally excludes the 5.32 GiB graph cache, raw dataset cache, isolated Python environment, lane scratch directories, and upstream repository clones. Before training, the notebook validated all 21 cached evaluated graphs against their per-graph byte-size and SHA-256 metadata. Both the graph binaries and those metadata files are absent from the light bundle, so a third party can regenerate the logical graphs but cannot compare their byte identity with the executed R6 cache. Absolute `/kaggle/working/...` graph paths in the CSVs are historical provenance, not portable local paths.

## 3. Dataset, task, and evaluation metrics

### 3.1 YelpChi profile

The dataset was loaded from `dgl.data.FraudDataset(name="yelp")`.

| Property | Value |
|---|---:|
| Nodes | 45,954 |
| Homogeneous directed edges | 8,051,348 |
| Feature columns | 32 |
| Fraud nodes | 6,677 |
| Normal nodes | 39,277 |
| Fraud prevalence | 14.5297% |
| Train nodes | 18,380 (40%) |
| Validation nodes | 9,190 (20%) |
| Test nodes | 18,384 (40%) |
| Split | `s0`, seed 717 |

This is a transductive node-classification experiment: the full graph structure and features are available, while masks control which node labels are used for training, validation, and test evaluation.

The 40/20/40 masks are regenerated deterministically with split seed 717 using an approximately stratified, class-wise split. Each class’s partition counts are floored separately and then shuffled within the partitioning procedure.

### 3.2 Why AP is the primary metric

At 14.53% prevalence, a random ranking has expected AP near 0.1453. AP is therefore more revealing than accuracy and often more discriminating than ROC-AUC under class imbalance. This analysis treats:

- **Average Precision** as the primary ranking metric;
- **ROC-AUC** as a threshold-free corroborating metric;
- **macro-F1** as a threshold-sensitive, class-balanced operational metric.

The classification threshold is chosen using the validation set’s best macro-F1 and then applied to the test set. The test labels are not used to select that threshold. In the clean-training shift protocol, the clean validation threshold remains attached to the clean-trained artifact while it is evaluated on each stressed graph.

## 4. System and experiment design as actually run

### 4.1 Shared graph view

Relation-aware perturbations are generated from YelpChi’s three source relations (`net_rsr`, `net_rtr`, and `net_rur`), but all four models consume the same graph after `dgl.to_homogeneous`. This controls the physical input surface across models, but it collapses relation identity during downstream learning.

That tradeoff is especially important for PMP: the adapter sees one `_E` relation, not the native three-relation YelpChi heterograph. These results evaluate a homogeneous integrated adapter, not the full relation-aware PMP setup from the paper.

### 4.2 Integrated models

| Model | Role and effective configuration |
|---|---|
| MLP | Graph-blind negative control; two 128-unit hidden layers, dropout 0.5, Adam at 0.001, weight decay 0.0005, weighted cross-entropy, up to 100 epochs, patience 10 |
| GraphSAGE | Full-graph/full-batch two-layer mean-aggregation baseline; hidden size 64, dropout 0.5, Adam at 0.01, weight decay 0.0005, weighted cross-entropy, up to 100 epochs, patience 10 |
| PMP | Pinned `LASAGE_S` integration; hidden size 48, one message-passing layer, one homogeneous relation, incoming-neighbor sampling with mean aggregation, fanout 10, batch 512, dropout 0, Adam at 0.01, weight decay 0, 100 epochs, patience 10, `num_workers=0`; only train labels exposed through `label_unk` |
| SEC-GFD | Hidden size 32, spectral order 2, high-order term 1, class-weighted cross-entropy plus 0.2× the repaired contrastive-like loss, Adam at 0.01, weight decay 0, 50 epochs, patience 10 |

PMP is pinned to upstream commit `3f7629f6c180891a0bc1bba3c66d94d288a1ddae`. Its compatibility patch replaces unavailable PyG normalization classes with identity modules. SEC-GFD is pinned to `97faa51145ed1fbcbdc67cc5d399490da9a9797a`; its compatibility patch permits zero-in-degree nodes. The SEC-GFD adapter also repairs the auxiliary-loss training indices and maps cosine similarity from `[-1, 1]` to `[0, 1]` before its positive log-ratio. These are disclosed correctness/integration changes, so the results must not be presented as exact paper reproductions.

The PMP adapter row-normalizes features, uses unweighted cross-entropy in accordance with the inherited effective setting, and encodes validation/test labels as unknown in `label_unk`. Its fallback from GraphNorm/GraphSizeNorm to identity is material and must remain visible in the methods section of the final report.

Every training artifact monitors validation ROC-AUC, restores the best state, and derives macro-F1’s threshold only from validation data. Every result row now records `epochs_trained`, `best_epoch`, `best_validation_monitor`, `validation_monitor`, and `stopping_reason`.

### 4.3 Protocols

| Protocol | Training graph | Evaluation graph | Scientific interpretation |
|---|---|---|---|
| `train_on_variant` | The same variant being evaluated | Same variant | Matched-distribution adaptation/retraining |
| `train_clean_eval_all` | Clean base graph only | Clean and all variants | Deployment-like sensitivity to unseen graph/feature shift |

These protocols answer different questions and must never be averaged. In oracle scenarios, `train_on_variant` additionally lets the learner train on topology or features constructed from labels across the entire graph. It is therefore marked `oracle_privileged_training_diagnostic` and excluded from operational rankings. Oracle `train_clean_eval_all` is marked `oracle_shift_sensitivity_diagnostic`.

The two non-oracle scenario families use `claim_scope=non_oracle_controlled_stress` and are the only rows with `operational_ranking_eligible=true`. That flag means eligible for comparison within this synthetic benchmark; it does not imply that the perturbation is a realistic attacker model.

### 4.4 Stress tests

| Scenario | Nonzero severities | Construction | Oracle? | Intended target |
|---|---|---|---:|---|
| `heterophily_rewire_oracle` | 0.15, 0.30 | Rewire selected destinations to opposite true-label nodes | Yes | Controlled true-label heterophily |
| `heterophily_rewire_nonoracle` | 0.15, 0.30 | Two-means feature partition, then rewire toward pseudo-opposite nodes | No | Label-free structural disagreement proxy |
| `camouflage_feature_oracle` | 0.15, 0.30 | Replace selected fraud features with sampled normal features, `gamma=1` | Yes | Feature camouflage |
| `camouflage_relation_oracle` | 0.15, 0.30 | Add two normal-looking edges and remove one suspicious edge per selected fraud node, within existing relation types | Yes | Relation camouflage |
| `noise_edges_uniform` | 0.10, 0.20 | Add directed random edges approximately uniformly across source relations | No | Density/noisy-neighborhood stress |

Nominal severity has different units and causal meaning in every family. A value of 0.30 in feature camouflage cannot be treated as equivalent to 0.30 in rewiring. All comparisons must remain within a scenario and be paired with the realized audit.

Every topology sampler in this run rejects self-loops, already-existing edges, and duplicate proposed edges. All requested changes were satisfied without exhausting the sampler.

## 5. Statistical estimands and uncertainty

### 5.1 Experimental units

Each positive-severity point has a crossed design of three training seeds and two graph seeds, giving six seed cells. The clean graph has three true training-seed observations. For paired comparisons, each clean training-seed score is matched to both stressed graph seeds without pretending that the duplicated clean value is an independent clean training run.

The R6 summaries improve over a flat six-row bootstrap:

- point intervals resample the training-seed and graph-seed axes;
- clean-minus-stress deltas pair on training seed and retain graph clusters;
- protocol contrasts pair identical training/graph cells;
- curve summaries construct complete per-seed severity trajectories before integration;
- retention is computed as a paired stressed/clean ratio;
- reported `std` is empirical seed-cell variability, not the standard deviation of bootstrap means.

The plot stage uses 1,000 bootstrap resamples with fixed bootstrap seed 0. Its intervals are descriptive 95% bands.

### 5.2 Definitions

For metric value `M`:

```text
absolute drop = M_clean - M_stress
ratio-of-means retention = mean(M_stress) / mean(M_clean)
paired retention = mean(M_stress_cell / M_clean_matched_training_seed)
protocol contrast = M_train_on_variant - M_train_clean_eval_all
curve integral = trapezoidal integral of M over configured severity
curve-average metric = curve integral / configured severity range
```

A positive drop means degradation. A negative drop means the stressed score is higher than its clean reference. A positive protocol contrast means matched-variant retraining scored higher than clean-training shift. The maximum-stress overview in Section 8 shows the ratio of reported point means; the configured-worst tables use the exported mean of paired seed-cell ratios.

The curve-average metric is not a clean-normalized robustness score. It strongly rewards a high clean baseline and is comparable only within the same scenario, protocol, metric, and severity grid.

### 5.3 One split remains the dominant limit

There are three training seeds and two graph seeds, but only one independently generated split. The `*_cross_split.csv` files contain `n_splits=1`, standard deviation zero, and point-width intervals. They are schema/bookkeeping outputs, not empirical cross-split uncertainty evidence, and should not be used in the final report’s substantive claims.

The within-split bootstrap bands are useful for showing seed sensitivity, particularly the GraphSAGE instability, but they do not justify formal statistical significance language.

## 6. Did the perturbations realize meaningful stress?

### 6.1 Maximum-severity audit

| Scenario | Requested and realized change | Realized target effect | Validity assessment |
|---|---:|---:|---|
| Oracle heterophily | 2,415,404 rewired edges | Heterophily 0.22688 → 0.45874, shift +0.23186 | Strong controlled oracle intervention |
| Non-oracle rewiring | 2,415,404 rewired edges | True-label heterophily 0.22688 → 0.23198, shift +0.00510 | Strong structural edit, weak named target |
| Feature camouflage | 2,003 fraud nodes replaced | Cosine to sampled normal 0.83979 → 1.00000; mean L2 change ≈2.241 | Strong feature intervention |
| Relation camouflage | 2,003 fraud nodes; +4,006 and −2,003 edges | Local fraud-to-normal ratio 0.81385 → 0.82088, shift +0.00703 | Underpowered for broad robustness claims |
| Uniform noise | 1,610,269 added directed edges | Edge count +20%; heterophily 0.22688 → 0.23041 | Strong density intervention, weak heterophily change |

All requested counts were realized exactly. Exact count realization, however, is not the same as construct validity.

### 6.2 Oracle heterophily

At severity 0.15, 1,207,702 edges are rewired and heterophily rises by about 0.11604. At severity 0.30, 2,415,404 edges are rewired and heterophily rises by about 0.23186. The two graph seeds realize nearly identical aggregate effects.

This is the strongest graph intervention in its stated target. It is also explicitly oracle: destination selection reads the true labels of all nodes, including test nodes. That is acceptable as a controlled sensitivity probe only when the report preserves the oracle qualifier.

### 6.3 Non-oracle feature-partition rewiring

The non-oracle scenario changes the same number of edges as the oracle scenario but produces only about 2.2% of its maximum true-label heterophily shift (`0.00510 / 0.23186`). The feature-derived pseudo-positive rate also changes from an average of 0.5943 at severity 0.15 to 0.3896 at severity 0.30.

The frozen config seeds the feature partition with severity, so the higher-severity graph is not a nested extension of the lower-severity graph. Severity changes both the number of rewired edges and the underlying pseudo-label partition. Non-monotonic model curves are therefore not clean dose-response evidence.

The accurate interpretation is **feature-partition-driven rewiring stress**, not successful strong true-label heterophily stress.

### 6.4 Oracle-selected complete feature camouflage

At severity 0.15, 1,001 fraud nodes are changed; at 0.30, 2,003 are changed. With `gamma=1`, each selected fraud feature vector is completely replaced by a sampled normal feature vector, which explains the post-change cosine similarity of exactly 1.0 to that sampled normal.

This is a deliberately strong, artificial oracle perturbation. It cleanly tests dependence on fraud-feature distinctiveness, but it is not a model of gradual or behaviorally constrained evasion.

### 6.5 Relation camouflage

Each selected fraud node receives two camouflage edges and loses one suspicious edge on average. At maximum severity this yields only +2,003 net edges, or about 0.0249% of the 8.05-million-edge base graph. Selected nodes have mean out-degree about 167.73 before and 168.73 after; the intervention adds only one net edge against that large neighborhood.

The local fraud-to-normal neighbor ratio rises by about 0.7 percentage points. The global heterophily ratio rises by only 0.00044. This scenario was deterministic and correctly executed, but its budget was too small relative to graph density. A null performance response cannot establish relation-camouflage robustness.

### 6.6 Uniform edge noise

The noise scenario adds 805,134 directed edges at severity 0.10 and 1,610,269 at 0.20. At maximum severity the three relations each receive roughly 536,000 added edges because allocation is approximately uniform by relation, not proportional to original relation size.

The graph becomes 20% denser, while global heterophily rises by only about 0.00353. This is primarily a density/information-flooding test, not a heterophily test. It does not model realistic fraud behavior or preserve relation composition.

## 7. Clean performance

Mean ± empirical sample standard deviation over three training seeds:

| Protocol | Model | ROC-AUC | Average Precision | macro-F1 |
|---|---|---:|---:|---:|
| Train on variant | MLP | 0.7813 ± 0.0288 | 0.3994 ± 0.0547 | 0.6532 ± 0.0236 |
| Train on variant | GraphSAGE | 0.7917 ± 0.0073 | 0.4295 ± 0.0155 | 0.6708 ± 0.0053 |
| Train on variant | PMP | **0.8492 ± 0.0038** | **0.5677 ± 0.0134** | **0.7197 ± 0.0020** |
| Train on variant | SEC-GFD | 0.8125 ± 0.0033 | 0.4709 ± 0.0047 | 0.6877 ± 0.0049 |
| Train clean, evaluate all | MLP | 0.7813 ± 0.0288 | 0.3994 ± 0.0547 | 0.6532 ± 0.0236 |
| Train clean, evaluate all | GraphSAGE | 0.7910 ± 0.0069 | 0.4271 ± 0.0134 | 0.6697 ± 0.0050 |
| Train clean, evaluate all | PMP | **0.8492 ± 0.0038** | **0.5677 ± 0.0134** | **0.7197 ± 0.0020** |
| Train clean, evaluate all | SEC-GFD | 0.8141 ± 0.0038 | 0.4755 ± 0.0006 | 0.6897 ± 0.0041 |

PMP leads every clean metric under both protocols. SEC-GFD is second in AP, GraphSAGE third, and MLP fourth.

The clean-training-shift descriptive AP intervals are `[0.3362, 0.4321]` for MLP, `[0.4118, 0.4365]` for GraphSAGE, `[0.5522, 0.5757]` for PMP, and `[0.4751, 0.4762]` for SEC-GFD. With only three training seeds, these bands primarily show the observed seed range rather than precise population uncertainty.

Relative to the 0.1453 fraud prevalence, the clean shift AP values are approximately 3.91× prevalence for PMP, 3.27× for SEC-GFD, 2.94× for GraphSAGE, and 2.75× for MLP.

MLP’s large clean AP standard deviation comes mainly from training seed 0, which stopped after 17 epochs at 0.3362 AP; seeds 1 and 2 reached 100 epochs and scored about 0.4298 and 0.4321. Clean performance is therefore not equally stable across models.

The clean case is trained separately in the two protocol lanes. MLP and PMP are bit-identical across those lanes. GraphSAGE differs by at most 0.00737 AP, and SEC-GFD by at most 0.00728 AP across matching training seeds. These small clean mismatches slightly contaminate very small protocol contrasts and are consistent with the run’s stated lack of bitwise CUDA determinism.

## 8. Maximum-stress Average Precision

Each cell reports:

```text
maximum-stress AP (clean minus stress drop; mean stressed AP / mean clean AP)
```

### 8.1 Train on each variant

| Scenario | MLP | GraphSAGE | PMP | SEC-GFD |
|---|---:|---:|---:|---:|
| Oracle heterophily | 0.3994 (+0.0000; 100.0%) | 0.6519 (−0.2223; 151.8%) | 0.6018 (−0.0341; 106.0%) | **1.0000 (−0.5291; 212.3%)** |
| Non-oracle rewiring | 0.3994 (+0.0000; 100.0%) | 0.3469 (+0.0826; 80.8%) | **0.5094 (+0.0583; 89.7%)** | 0.4842 (−0.0132; 102.8%) |
| Feature camouflage | 0.3213 (+0.0781; 80.5%) | 0.3119 (+0.1176; 72.6%) | **0.4174 (+0.1503; 73.5%)** | 0.3936 (+0.0773; 83.6%) |
| Relation camouflage | 0.3994 (+0.0000; 100.0%) | 0.3906 (+0.0390; 90.9%) | **0.5491 (+0.0186; 96.7%)** | 0.4720 (−0.0011; 100.2%) |
| Uniform edge noise | 0.3994 (+0.0000; 100.0%) | 0.2973 (+0.1322; 69.2%) | **0.5228 (+0.0450; 92.1%)** | 0.4783 (−0.0073; 101.6%) |

### 8.2 Train clean, evaluate all variants

| Scenario | MLP | GraphSAGE | PMP | SEC-GFD |
|---|---:|---:|---:|---:|
| Oracle heterophily | **0.3994 (+0.0000; 100.0%)** | 0.3838 (+0.0433; 89.9%) | 0.3859 (+0.1818; 68.0%) | 0.1570 (+0.3185; 33.0%) |
| Non-oracle rewiring | 0.3994 (+0.0000; 100.0%) | 0.4134 (+0.0137; 96.8%) | **0.4895 (+0.0783; 86.2%)** | 0.4586 (+0.0169; 96.4%) |
| Feature camouflage | 0.3170 (+0.0824; 79.4%) | 0.3352 (+0.0919; 78.5%) | **0.4471 (+0.1206; 78.8%)** | 0.3732 (+0.1023; 78.5%) |
| Relation camouflage | 0.3994 (+0.0000; 100.0%) | 0.4268 (+0.0003; 99.9%) | **0.5684 (−0.0007; 100.1%)** | 0.4760 (−0.0005; 100.1%) |
| Uniform edge noise | 0.3994 (+0.0000; 100.0%) | 0.4186 (+0.0085; 98.0%) | **0.5178 (+0.0499; 91.2%)** | 0.4651 (+0.0105; 97.8%) |

PMP has the highest maximum-stress AP in eight of ten rows. The two exceptions need careful interpretation:

- Under oracle retraining, SEC-GFD learns the label-derived topology and reaches AP ≈1.
- Under oracle clean-training shift, the graph-blind MLP “wins” because edge rewiring cannot alter its predictions. It still has weaker clean AP than PMP and SEC-GFD, so this is an invariance control rather than evidence that MLP is the best fraud detector.

## 9. Scenario-by-scenario interpretation

### 9.1 Oracle heterophily: a topology-regime and protocol diagnostic

At severity 0.30:

| Model | Retrained AP | Clean-trained AP | Retrained − shift AP | Paired 95% descriptive interval |
|---|---:|---:|---:|---:|
| MLP | 0.3994 | 0.3994 | 0.0000 | [0.0000, 0.0000] |
| GraphSAGE | 0.6519 | 0.3838 | +0.2681 | [0.2396, 0.2985] |
| PMP | 0.6018 | 0.3859 | +0.2159 | [0.1508, 0.2611] |
| SEC-GFD | 1.0000 | 0.1570 | +0.8430 | [0.8272, 0.8641] |

SEC-GFD’s clean-trained maximum-stress ROC-AUC/AP/macro-F1 are approximately `0.5237 / 0.1570 / 0.5030`. ROC-AUC and AP are near their random-ranking references, while macro-F1 is weak but has no universal random baseline because it depends on prevalence, predictions, and the selected threshold. After retraining the three metrics are essentially `1.0000 / 1.0000 / 0.9998`.

The oracle graph is not merely “more difficult.” Rewiring every selected edge toward an opposite-label destination creates a highly regular label-dependent topology. A model trained on that topology can exploit it; a clean-trained model encounters a major structural regime change. The result is scientifically useful as an adaptation/shift diagnostic, but it cannot be interpreted as realistic adversarial robustness or as a fair operational victory for SEC-GFD.

### 9.2 Oracle-selected complete feature replacement: the most consistent degradation

Under clean-training shift, the maximum configured oracle feature-camouflage diagnostic (`gamma=1`) produces remarkably similar relative AP losses:

| Model | Absolute AP drop | Relative AP loss | Paired drop interval |
|---|---:|---:|---:|
| MLP | 0.0824 | 20.6% | [0.0677, 0.0983] |
| GraphSAGE | 0.0919 | 21.5% | [0.0827, 0.1006] |
| PMP | 0.1206 | 21.2% | [0.1120, 0.1285] |
| SEC-GFD | 0.1023 | 21.5% | [0.0974, 0.1074] |

Under matched retraining, the relative AP losses are 19.5% for MLP, 27.4% for GraphSAGE, 26.5% for PMP, and 16.4% for SEC-GFD. SEC-GFD retains the largest fraction of its own clean AP under retraining, but PMP still has the highest absolute stressed AP: 0.4174 versus 0.3936 for SEC-GFD.

The fact that MLP also degrades is an important internal check: feature camouflage edits exactly the input it consumes. Graph structure cannot fully compensate for the loss of discriminative features in any integrated model.

### 9.3 Non-oracle rewiring: PMP is strongest, but the stress is not strong true-label heterophily

Under clean-training shift at severity 0.30:

- GraphSAGE loses 0.0137 AP, interval `[0.0123, 0.0158]`;
- PMP loses 0.0783 AP, interval `[0.0699, 0.0854]`;
- SEC-GFD loses 0.0169 AP, interval `[0.0088, 0.0227]`;
- MLP is exactly unchanged.

PMP is less invariant in absolute drop than GraphSAGE or SEC-GFD, yet it remains the strongest detector at 0.4895 AP. This is the clearest example of why drop and absolute performance must be reported together.

With retraining, PMP and SEC-GFD score 0.5094 and 0.4842 AP, while GraphSAGE falls to 0.3469 with high seed variability. Because the pseudo-label partition itself changes across severity, the small improvement in GraphSAGE from 0.3196 at severity 0.15 to 0.3469 at 0.30 is not evidence that more stress helps.

### 9.4 Relation camouflage: numerically tiny changes are scientifically inconclusive

Clean-training shift is nearly flat:

- MLP drop: exactly 0;
- GraphSAGE drop: +0.00026;
- PMP drop: −0.00071;
- SEC-GFD drop: −0.00046.

Some tiny paired intervals exclude zero because the evaluation is deterministic enough to resolve very small differences. That does not make the intervention scientifically important. Construct strength is the limiting issue: one net edge per selected node against mean degree around 168 is too small to test broad relation-camouflage resilience.

The train-on-variant GraphSAGE curve is also non-monotonic and unstable: AP is 0.3292 at severity 0.15 and 0.3906 at 0.30. Its maximum-severity drop interval spans roughly −0.0013 to 0.1477. This is training instability superimposed on a weak perturbation.

### 9.5 Uniform noise: density shift matters most for PMP under deployment and GraphSAGE under retraining

At 20% added edges under clean-training shift:

- PMP loses 0.0499 AP but remains best at 0.5178;
- GraphSAGE loses 0.0085 AP;
- SEC-GFD loses 0.0105 AP;
- MLP is unchanged.

The graph-aware models’ paired shift-drop intervals are all above zero, but PMP’s absolute loss is largest. That is consistent with its greater use of neighborhood information, while its stronger clean baseline preserves the best absolute score.

Under retraining, GraphSAGE is much worse: the configured worst point is severity 0.10, AP 0.2947, and retention 68.3%. Its AP is 0.2973 at severity 0.20. Both points contain multiple early-stopped low-performance runs, so the result is better described as fragile adaptation than as a smooth noise-response curve.

## 10. ROC-AUC and macro-F1 corroboration

AP is primary, but the other metrics support the main interpretation.

### 10.1 Maximum-stress ROC-AUC

Cells are `stressed ROC-AUC (clean minus stress drop)`.

| Protocol/scenario | MLP | GraphSAGE | PMP | SEC-GFD |
|---|---:|---:|---:|---:|
| Variant / oracle heterophily | 0.7813 (+0.0000) | 0.8976 (−0.1059) | 0.8877 (−0.0385) | **1.0000 (−0.1875)** |
| Variant / non-oracle rewiring | 0.7813 (+0.0000) | 0.7307 (+0.0610) | **0.8314 (+0.0178)** | 0.8198 (−0.0072) |
| Variant / feature camouflage | 0.6999 (+0.0814) | 0.6875 (+0.1042) | **0.7598 (+0.0894)** | 0.7480 (+0.0646) |
| Variant / relation camouflage | 0.7813 (+0.0000) | 0.7619 (+0.0298) | **0.8460 (+0.0032)** | 0.8135 (−0.0010) |
| Variant / uniform noise | 0.7813 (+0.0000) | 0.6947 (+0.0970) | **0.8376 (+0.0116)** | 0.8173 (−0.0048) |
| Shift / oracle heterophily | **0.7813 (+0.0000)** | 0.7683 (+0.0226) | 0.7635 (+0.0857) | 0.5237 (+0.2904) |
| Shift / non-oracle rewiring | 0.7813 (+0.0000) | 0.7850 (+0.0060) | **0.8168 (+0.0324)** | 0.8075 (+0.0066) |
| Shift / feature camouflage | 0.6986 (+0.0826) | 0.7072 (+0.0838) | **0.7631 (+0.0861)** | 0.7237 (+0.0904) |
| Shift / relation camouflage | 0.7813 (+0.0000) | 0.7908 (+0.0002) | **0.8499 (−0.0007)** | 0.8143 (−0.0003) |
| Shift / uniform noise | 0.7813 (+0.0000) | 0.7877 (+0.0032) | **0.8331 (+0.0160)** | 0.8104 (+0.0037) |

### 10.2 Maximum-stress macro-F1

| Protocol/scenario | MLP | GraphSAGE | PMP | SEC-GFD |
|---|---:|---:|---:|---:|
| Variant / oracle heterophily | 0.6532 (+0.0000) | 0.7681 (−0.0973) | 0.7496 (−0.0299) | **0.9998 (−0.3121)** |
| Variant / non-oracle rewiring | 0.6532 (+0.0000) | 0.6249 (+0.0460) | **0.7018 (+0.0178)** | 0.6964 (−0.0086) |
| Variant / feature camouflage | 0.6166 (+0.0366) | 0.6118 (+0.0590) | **0.6612 (+0.0585)** | 0.6513 (+0.0364) |
| Variant / relation camouflage | 0.6532 (+0.0000) | 0.6493 (+0.0216) | **0.7176 (+0.0021)** | 0.6885 (−0.0008) |
| Variant / uniform noise | 0.6532 (+0.0000) | 0.6021 (+0.0687) | **0.7088 (+0.0109)** | 0.6929 (−0.0052) |
| Shift / oracle heterophily | **0.6532 (+0.0000)** | 0.6493 (+0.0204) | 0.6415 (+0.0781) | 0.5030 (+0.1866) |
| Shift / non-oracle rewiring | 0.6532 (+0.0000) | 0.6627 (+0.0070) | 0.6828 (+0.0368) | **0.6840 (+0.0057)** |
| Shift / feature camouflage | 0.6125 (+0.0407) | 0.6249 (+0.0448) | **0.6652 (+0.0544)** | 0.6425 (+0.0472) |
| Shift / relation camouflage | 0.6532 (+0.0000) | 0.6695 (+0.0002) | **0.7191 (+0.0006)** | 0.6903 (−0.0006) |
| Shift / uniform noise | 0.6532 (+0.0000) | 0.6643 (+0.0055) | **0.7011 (+0.0186)** | 0.6872 (+0.0025) |

The threshold-free and thresholded metrics tell the same broad story: the configured oracle-selected complete feature replacement degrades every model; PMP is usually strongest in absolute performance; and SEC-GFD’s oracle heterophily result reverses across protocols.

## 11. Direct paired protocol contrasts

The maximum-severity AP contrast is defined as:

```text
train_on_variant AP - train_clean_eval_all AP
```

| Scenario | MLP | GraphSAGE | PMP | SEC-GFD |
|---|---:|---:|---:|---:|
| Oracle heterophily | +0.0000 | **+0.2681** | **+0.2159** | **+0.8430** |
| Non-oracle rewiring | +0.0000 | −0.0664 | +0.0199 | +0.0256 |
| Feature camouflage | +0.0043 | −0.0233 | −0.0297 | +0.0204 |
| Relation camouflage | +0.0000 | −0.0362 | −0.0194 | −0.0040 |
| Uniform edge noise | +0.0000 | **−0.1212** | +0.0050 | +0.0132 |

Key paired intervals:

- Oracle heterophily: GraphSAGE `[0.2396, 0.2985]`, PMP `[0.1508, 0.2611]`, SEC-GFD `[0.8272, 0.8641]`.
- Non-oracle rewiring: PMP `[0.0099, 0.0351]`, SEC-GFD `[0.0150, 0.0319]`; GraphSAGE `[-0.1830, 0.0020]`.
- Uniform noise: GraphSAGE `[-0.1947, -0.0039]`; PMP and SEC-GFD intervals cross zero.
- Feature camouflage: PMP `[-0.0490, -0.0156]`; the other model intervals cross zero.

Retraining is not uniformly beneficial. GraphSAGE’s negative noise contrast is largely associated with its low early-stopped train-on-variant runs. The paired contrast describes what this pipeline produced; it does not prove that retraining on noisy graphs is intrinsically harmful.

## 12. Severity-curve summaries and configured worst cases

### 12.1 Mean AP over each configured severity curve

This is the trapezoidal curve integral divided by the scenario’s configured severity range. It rewards absolute performance and must be compared only within one scenario.

| Train-on-variant scenario | MLP | GraphSAGE | PMP | SEC-GFD |
|---|---:|---:|---:|---:|
| Oracle heterophily | 0.3994 | 0.5063 | 0.5831 | **0.8628** |
| Non-oracle rewiring | 0.3994 | 0.3539 | **0.5334** | 0.4756 |
| Feature camouflage | 0.3587 | 0.3626 | **0.4962** | 0.4284 |
| Relation camouflage | 0.3994 | 0.3696 | **0.5568** | 0.4707 |
| Uniform edge noise | 0.3994 | 0.3291 | **0.5408** | 0.4744 |

| Clean-training-shift scenario | MLP | GraphSAGE | PMP | SEC-GFD |
|---|---:|---:|---:|---:|
| Oracle heterophily | 0.3994 | 0.4013 | **0.4495** | 0.2966 |
| Non-oracle rewiring | 0.3994 | 0.4195 | **0.5228** | 0.4662 |
| Feature camouflage | 0.3597 | 0.3832 | **0.5101** | 0.4270 |
| Relation camouflage | 0.3994 | 0.4270 | **0.5666** | 0.4757 |
| Uniform edge noise | 0.3994 | 0.4223 | **0.5382** | 0.4695 |

PMP has the best curve-average AP in all five shift scenarios and four of five retraining scenarios. SEC-GFD leads only the oracle-retraining curve.

### 12.2 Global configured worst AP

`worst_case_performance.csv` selects the minimum mean among the configured nonzero points; it is not an adversarial optimization over continuous perturbations.

| Protocol | Model | Selected scenario/severity | Worst AP | Clean AP | Drop | Retention |
|---|---|---|---:|---:|---:|---:|
| Train on variant | MLP | Feature camouflage / 0.30 | 0.3213 | 0.3994 | 0.0781 | 81.1% |
| Train on variant | GraphSAGE | Uniform noise / 0.10 | 0.2947 | 0.4295 | 0.1348 | 68.3% |
| Train on variant | PMP | Feature camouflage / 0.30 | 0.4174 | 0.5677 | 0.1503 | 73.6% |
| Train on variant | SEC-GFD | Feature camouflage / 0.30 | 0.3936 | 0.4709 | 0.0773 | 83.6% |
| Train clean, evaluate all | MLP | Feature camouflage / 0.30 | 0.3170 | 0.3994 | 0.0824 | 79.5% |
| Train clean, evaluate all | GraphSAGE | Feature camouflage / 0.30 | 0.3352 | 0.4271 | 0.0919 | 78.5% |
| Train clean, evaluate all | PMP | Oracle heterophily / 0.30 | 0.3859 | 0.5677 | 0.1818 | 68.0% |
| Train clean, evaluate all | SEC-GFD | Oracle heterophily / 0.30 | 0.1570 | 0.4755 | 0.3185 | 33.0% |

Twenty-two of the 24 global worst rows across all three metrics select an oracle scenario and are marked ineligible for operational ranking. The only eligible rows are GraphSAGE’s train-on-variant AP and macro-F1 worst points under noise. The global table is therefore a descriptive stress envelope, not a production leaderboard.

### 12.3 Non-oracle operational stress envelope

Restricting the envelope to the two non-oracle families gives a more operational—but still synthetic—comparison:

| Protocol | Model | Worst non-oracle point | AP | Retention |
|---|---|---|---:|---:|
| Train on variant | MLP | Non-oracle rewiring / 0.15 | 0.3994 | 100.0% |
| Train on variant | GraphSAGE | Uniform noise / 0.10 | 0.2947 | 68.3% |
| Train on variant | PMP | Non-oracle rewiring / 0.30 | **0.5094** | 89.7% |
| Train on variant | SEC-GFD | Non-oracle rewiring / 0.15 | 0.4737 | 100.6% |
| Train clean, evaluate all | MLP | Non-oracle rewiring / 0.15 | 0.3994 | 100.0% |
| Train clean, evaluate all | GraphSAGE | Non-oracle rewiring / 0.30 | 0.4134 | 96.8% |
| Train clean, evaluate all | PMP | Non-oracle rewiring / 0.30 | **0.4895** | 86.2% |
| Train clean, evaluate all | SEC-GFD | Non-oracle rewiring / 0.30 | 0.4586 | 96.4% |

PMP has the highest absolute AP in both non-oracle envelopes. MLP has perfect graph-only retention by construction but a weaker score.

## 13. Model-specific assessment

### 13.1 MLP: essential control, not a default robustness winner

Strengths:

- Simple and reproducible.
- Exact invariance under every graph-only test-time transformation.
- Detects the expected loss under feature camouflage.

Limitations:

- Lowest clean AP.
- Cannot exploit relational evidence.
- A zero graph-stress drop is structural blindness, not proof of superior fraud detection.

The MLP establishes that edge-only variants did not accidentally modify features or corrupt evaluation pairing. Its exact zero drop and `[0, 0]` paired intervals are a strong internal-validity check.

Across the four graph-only clean-training shift families, its maximum absolute difference from clean is exactly zero for AP, ROC-AUC, and macro-F1.

### 13.2 GraphSAGE: competitive clean baseline, unstable matched retraining

GraphSAGE improves on MLP clean AP, but its train-on-variant results are bimodal:

| Stopping group | Runs | Mean AP | Mean ROC-AUC | Mean epochs | Mean duration |
|---|---:|---:|---:|---:|---:|
| Early stopped | 20 | 0.2223 | 0.6388 | 26.0 | 1.12 s |
| Reached max epochs | 43 | 0.4524 | 0.7978 | 100.0 | 3.47 s |

Across its 63 variant-trained rows:

- epochs versus AP correlation: 0.809;
- duration versus AP correlation: 0.811;
- best validation ROC-AUC versus test AP correlation: 0.983.

The worst instability appears under noise and non-oracle/weak relation variants. For example, maximum-noise AP has empirical SD about 0.100. The telemetry makes the failure visible, but association does not identify a single cause. The next run should test longer warm-up, different patience/min-delta rules, learning-rate sensitivity, and a fixed-epoch control.

Training seed 1 accounts for 12 of the 14 GraphSAGE train-on-variant runs below 0.25 AP, which further shows that the weakness is not solely a scenario-level effect.

### 13.3 PMP: strongest absolute detector, meaningful but not perfect graph sensitivity

PMP’s clean AP of 0.5677 is substantially above SEC-GFD, GraphSAGE, and MLP. It remains the maximum-stress AP leader in every non-oracle row and in both feature-camouflage rows.

Its larger shift drops under non-oracle rewiring and noise show that it uses graph structure rather than merely ignoring it. The key practical result is that it retains enough absolute margin to stay best. That is stronger evidence than either clean score or smallest drop alone.

Limitations include the homogeneous one-relation adapter, feasible rather than paper-native hyperparameters, and PMP’s dominant compute cost. The result supports this integrated adapter on this split; it does not reproduce all claims of the original PMP paper.

### 13.4 SEC-GFD: strong second model and extreme oracle interaction

SEC-GFD is the second-best clean AP model and shows small non-oracle shift degradation. Under feature-camouflage retraining it retains the largest fraction of its clean AP among the graph-aware models.

Its defining result is the oracle contrast. At 0.30 oracle heterophily all six retrained rows reach essentially perfect ranking, and the runs stop after roughly 12–14 epochs because validation AUC is already 1.0. Those successful early stops drive the model’s negative overall epoch/AP correlation; unlike GraphSAGE, they are not a collapse mode.

The result is inseparable from label-constructed topology and the repaired integrated adapter. It should be shown because it is scientifically revealing, but captioned as privileged diagnostic evidence.

## 14. Training telemetry and decision-threshold behavior

All 504 rows contain complete telemetry, and every validation monitor is ROC-AUC. Both `early_stopping` and `max_epochs` are represented.

For train-on-variant runs:

| Model | Mean epochs | Min–max epochs | Early-stopped rows | Mean row duration |
|---|---:|---:|---:|---:|
| MLP | 73.7 | 17–100 | 20/63 | 1.56 s |
| GraphSAGE | 76.5 | 24–100 | 20/63 | 2.73 s |
| PMP | 76.9 | 20–100 | 57/63 | 42.06 s |
| SEC-GFD | 45.1 | 12–50 | 9/63 | 4.59 s |

For clean-training shift, telemetry is repeated across the 21 evaluations associated with each clean-trained artifact. It must be deduplicated by model and training seed before interpreting training behavior. In particular, 63 shift rows per model do not represent 63 independent fits.

The large threshold differences between separately trained clean GraphSAGE and SEC-GFD lanes—up to 0.099 and 0.078—show why macro-F1 is more variable and why AP/ROC-AUC deserve primary emphasis.

## 15. R5-to-R6 repeatability check

The R5 and R6 configs are identical after removing the experiment name, and their 504 scientific keys align exactly. R6 is nevertheless an implementation-revision/GPU rerun, not a guaranteed bitwise replication.

| Model | Exact AP matches out of 126 | Mean absolute AP difference | Maximum absolute AP difference |
|---|---:|---:|---:|
| MLP | 126 | 0.000000 | 0.000000 |
| PMP | 126 | 0.000000 | 0.000000 |
| GraphSAGE | 0 | 0.001781 | 0.009606 |
| SEC-GFD | 7 | 0.003610 | 0.024559 |

Across all models, the mean absolute R5–R6 difference is 0.00135 AP; the largest ROC-AUC difference is 0.0501 and the largest macro-F1 difference is 0.0195. All ten maximum-stress AP winners are unchanged between R5 and R6.

This supports the stability of the high-level story while also demonstrating why seeded GPU runs should not be described as universally bitwise deterministic. Because the implementation ID changed, the comparison cannot isolate GPU nondeterminism as the only source of GraphSAGE/SEC-GFD differences.

## 16. Runtime and resource interpretation

The isolated environment used Python 3.11.15, PyTorch 2.1.0+cu118, DGL 1.1.3+cu118, and NumPy 1.26.4 on two Tesla T4 GPUs with 15,360 MiB each. The host exposed about 31.35 GiB RAM.

Non-training stage evidence:

| Stage/resource | Observed value |
|---|---:|
| Graph-generation benchmark | 134.67 s |
| Graph generation plus validation cell | 152.03 s |
| CUDA smoke matrix | 8.88 s |
| Plot/report generation | 78.87 s |
| Free disk at start | 19.502 GiB |
| Free disk after isolated setup | 14.157 GiB |
| Free disk after graph generation | 8.485 GiB |
| Graph cache | 5,714,978,057 bytes (5.323 GiB) |

| Lane | Protocol | Duration |
|---|---|---:|
| PMP | Train on variant | 2,657.99 s (44.30 min) |
| SEC-GFD | Train on variant | 296.31 s (4.94 min) |
| MLP + GraphSAGE | Train on variant | 278.65 s (4.64 min) |
| PMP | Train clean, evaluate all | 199.46 s (3.32 min) |
| MLP + GraphSAGE | Train clean, evaluate all | 69.72 s (1.16 min) |
| SEC-GFD | Train clean, evaluate all | 46.77 s (0.78 min) |

PMP is the operational bottleneck. The shift lanes are much shorter because each model trains only three times and reuses those artifacts across variants. Lane durations are not fair model-speed benchmarks: tasks ran concurrently and shared CPU, RAM, disk, and GPU orchestration resources.

The row-level `duration_sec` field also changes meaning by protocol. A train-on-variant row includes fitting and evaluation; a clean shift row includes the shared clean fit and clean evaluation; a stressed shift row is evaluation-only. Those row durations cannot be compared as if every row represented a fresh training run.

## 17. Relation to the reviewed research

The correct comparison to the papers in [`Paper_Summary.md`](Paper_Summary.md) is mechanistic, not numerical.

### CARE-GNN

CARE-GNN motivates the distinction between feature and relation camouflage. R6 shows that every evaluated model degrades under the configured oracle-selected complete feature replacement. It does not establish vulnerability to every form of feature camouflage and does not answer whether CARE-GNN would solve the problem because CARE-GNN was not integrated. The weak realized relation intervention also means the CARE-GNN-style relation-camouflage question remains open.

### PMP

PMP separates neighbor roles to reduce harmful mixing under imbalance, heterophily, and noisy neighborhoods. Its strong absolute performance in R6 is qualitatively consistent with that motivation. It is not a numeric confirmation of the paper because the benchmark uses a homogeneous one-relation view, three seeds, one split, and a reduced feasible adapter setup.

### SEC-GFD

SEC-GFD’s spectral and environmental-constraint design targets heterophily. The R6 oracle interaction shows that the integrated model is highly sensitive to whether a label-derived structural regime is available during training. That result neither proves nor refutes the paper’s robustness claims: the oracle graph is a benchmark construction that makes topology itself label-informative.

### GAGA

GAGA is relevant as an alternative grouping/attention mechanism for low-homophily graphs, but it was not evaluated. No empirical GAGA comparison is supported.

### Pitfalls of GNN Evaluation

The run’s seed-aware reporting responds to initialization and perturbation variability, but it still uses one split. The main lesson from evaluation-pitfall research therefore remains: within-split seeds do not establish a stable model ranking across data partitions.

Direct cross-paper numeric comparisons would be invalid because the papers differ in dataset versions, graph relations, split ratios, trial counts, tuning, metrics, and model implementations.

## 18. Threats to validity

### 18.1 Internal validity

Strengths:

- Exact 504-key completion with no errors or duplicates.
- Decoupled graph and training seeds.
- Fixed split masks.
- Validation-only model selection and F1 thresholding.
- Paired, seed-aware derived summaries.
- Realized perturbation audit and row-level performance/audit join.
- Exact MLP graph-only invariance.
- Source, dataset, config, upstream, patch, and artifact hashes.

Risks:

- CUDA kernels are seeded but not guaranteed bitwise deterministic.
- The two protocols separately retrain the clean model, creating small baseline mismatches for GraphSAGE and SEC-GFD.
- Clean rows store split seed 717 in the legacy `graph_seed` field, conflating two seed concepts.
- Oracle construction uses all true labels, including test-node labels. `no_test_leakage=true` is accurate for training/threshold selection, not for oracle perturbation construction.

### 18.2 Construct validity

- Feature camouflage is a complete replacement (`gamma=1`), not gradual plausible evasion.
- Relation camouflage is underpowered relative to graph degree and size.
- Non-oracle rewiring weakly changes true-label heterophily and changes its feature partition across severity.
- Uniform noise changes density but allocates edges uniformly across relations and is not behaviorally realistic.
- Severity values are family-specific and variants are not guaranteed nested.
- Oracle heterophily may inject a learnable label signal rather than monotonically corrupt the task.

### 18.3 Statistical conclusion validity

- Only one split exists.
- There are only three training seeds and two graph seeds.
- Bootstrap intervals are discrete, descriptive, and based on very small axes.
- Worst-case selection is post hoc over configured points and does not account for selection uncertainty.
- Multiple scenarios, models, severities, protocols, and metrics are inspected; the run does not support uncorrected significance claims.
- Cross-split files with `n_splits=1` are not inferential evidence.

### 18.4 External validity

- One YelpChi version, one split, and one static transductive graph.
- No Amazon or second institutional dataset.
- No temporal drift, delayed labels, adaptive attacker, or business-cost simulation.
- Homogenization removes native relation-specific downstream behavior.
- PMP and SEC-GFD are integrated, patched adapters rather than exact paper pipelines.

### 18.5 Reproducibility boundary

The light bundle provides strong logical provenance but omits physical graph binaries, their per-graph metadata, and the exact external post-ZIP manifest. The notebook verified all 21 executed graph files against byte-size/SHA metadata before training, but an external auditor cannot compare regenerated bytes with that absent R6 cache. Reproduction therefore requires downloading the pinned YelpChi source, rebuilding logical graphs with the embedded code/config, and accepting that CUDA may not be bit-identical.

## 19. Supported and unsupported claims

### 19.1 Supported

- R6 completed all 504 configured evaluations without missing or error rows.
- PMP had the best clean ROC-AUC, AP, and macro-F1 in this fixed-split run.
- PMP had the highest maximum-stress AP in 8 of 10 protocol/scenario comparisons.
- The configured oracle-selected complete feature replacement consistently reduced all three metrics for all four models.
- PMP remained the highest-AP model under every non-oracle maximum-stress comparison.
- Oracle heterophily produced a very large adaptation-versus-shift interaction, especially for SEC-GFD.
- GraphSAGE’s matched-retraining performance was unstable and strongly associated with stopping behavior.
- The tested relation-camouflage configuration was too weak to support a broad robustness conclusion.
- The non-oracle rewiring scenario did not realize a large true-label heterophily shift.
- The corrected paired summaries repair the false uncertainty that a flat, independent clean/stress bootstrap would assign to exact MLP invariants.

### 19.2 Unsupported

- PMP, SEC-GFD, GraphSAGE, or MLP is universally “the most robust model.”
- The descriptive bootstrap intervals establish formal statistical significance.
- Near-perfect oracle-retrained SEC-GFD demonstrates production robustness.
- A near-zero relation-camouflage response proves robustness to camouflage.
- The non-oracle scenario is a strong realized true-label heterophily attack.
- MLP is the best detector because its graph-only drop is zero.
- Retraining always helps, or GraphSAGE inherently cannot learn from noisy graphs.
- The results reproduce the original PMP or SEC-GFD paper numbers.
- The rankings generalize to other YelpChi splits, datasets, time periods, or organizations.
- Concurrent lane time establishes model runtime superiority.
- The executed notebook is the multi-split v4 experiment.

## 20. Lessons learned and prioritized future work

### Priority 1: execute a genuinely independent multi-split confirmation

The highest-value next run is the prospective multi-split design, not a denser severity grid or more models. At least three independently seeded splits and five training seeds would directly address the dominant inference limitation. The final report should clearly distinguish this R6 evidence from any future v4 results.

### Priority 2: gate scenarios on realized construct strength

Before expensive training, each scenario should have a predeclared acceptance contract:

- primary realized effect metric;
- expected direction;
- minimum target/tolerance;
- `valid_for_claim` flag.

Relation camouflage should use a degree-relative or target-ratio budget. A scenario that changes local neighbor composition by only 0.007 should be automatically marked underpowered.

### Priority 3: make severity trajectories nested and semantically stable

Generate one perturbation plan per scenario/graph seed and take nested prefixes as severity increases. Fit the non-oracle feature partition once per graph seed. This would turn severity curves into clearer dose-response evidence.

### Priority 4: diagnose GraphSAGE training stability

Run controlled ablations for:

- fixed 100-epoch training versus early stopping;
- longer patience and warm-up;
- learning-rate and weight-decay sensitivity;
- checkpoint selection by AP instead of ROC-AUC;
- identical clean artifact reuse across protocols.

The objective is to separate actual structural sensitivity from optimizer/checkpoint instability.

### Priority 5: restore native relation-aware model views

Cache one logical perturbation with both heterogeneous and homogeneous physical views. Let PMP consume the native relations, add a relational baseline such as R-GCN, and record `input_graph_view`, relation count, adapter hash, and effective hyperparameter hash per row.

### Priority 6: improve realism and operational relevance

- use partial feature blending rather than only `gamma=1` replacement;
- use degree-preserving or relation-preserving rewiring controls;
- allocate ordinary noise proportional to relation size;
- add temporal or pseudo-temporal splits;
- test calibration, cost-sensitive errors, recall at fixed alert budget, and threshold transfer;
- eventually model adaptive but plausible fraud behavior.

### Priority 7: expand only after validity is stronger

A second dataset, CARE-GNN, and GAGA are valuable extensions, but they should follow—not precede—multi-split validation, stronger perturbation contracts, and graph-view fidelity. More models cannot repair a weak stress test or a one-split inference design.

## 21. Final-report-ready asset map

| Final report need | Recommended R6 source |
|---|---|
| Exact methods/config | `config.json`, notebook methods cells, run fingerprint |
| Dataset and resources | `kaggle_run_manifest.json` dataset profile/runtime sections |
| Raw result traceability | `results.csv` |
| Perturbation definitions and realized effects | `variant_audit.csv`, `graph_variants.csv` |
| Performance curves with descriptive intervals | `plots/summary_curves.csv` |
| Maximum-stress clean-minus-stress results | `plots/performance_drop_max_stress.csv` |
| Protocol adaptation/shift interaction | `plots/protocol_contrasts.csv` |
| Configured worst AP and retention | `plots/worst_case_performance.csv` |
| Performance-to-stress linkage | `plots/performance_audit_join.csv` |
| Completeness evidence | `plots/missing_or_error_runs.csv`, manifest receipts |
| Main AP figure | `presentation/figure_1_ap_curves.png` |
| Drop overview | `presentation/figure_2_ap_drop_heatmap.png` |
| Realized perturbation figure | `presentation/figure_3_realized_audit.png` |

Recommended final-report figure captions must state the protocol, oracle status, fixed split, seed counts, and that intervals are descriptive. The three presentation figures are useful overview graphics, but exact claims should cite the CSV values.

Do not use the `*_cross_split.csv` tables for evidence in this report; with one split, they are placeholders.

## 22. Suggested final-report narrative

The most defensible narrative is:

1. Fraud detection graphs are imbalanced and structurally messy; ordinary neighborhood aggregation can fail under heterophily, camouflage, and density.
2. Existing methods attack different parts of this problem, so clean-paper scores do not form a shared robustness benchmark.
3. This project contributes a common, auditable harness with deterministic stress generation, two protocol semantics, four integrated models, realized-stress auditing, and unified metrics.
4. PMP is the strongest absolute detector in the observed run, but robustness depends on whether one values absolute stressed performance, retention, matched adaptation, or unseen-shift stability.
5. The configured oracle-selected complete feature replacement is the clearest shared failure mode.
6. Oracle heterophily demonstrates why protocol and label-access scope fundamentally change interpretation.
7. Weak relation camouflage and feature-partition rewiring show why perturbation audits are as important as model metrics.
8. One split and homogeneous adapters constrain generalization; the next contribution should be confirmatory multi-split and native-view evaluation rather than a broader but less valid model zoo.

## 23. Conclusion

R6 is a complete, traceable, and methodologically improved execution of the v3 YelpChi robustness benchmark. Its strongest empirical conclusion is that PMP combines the best clean performance with the best absolute AP under most configured stresses. Its strongest failure-mode conclusion is that the configured oracle-selected complete feature replacement harms every model. Its most revealing diagnostic is the SEC-GFD oracle heterophily reversal: nearly perfect matched retraining and near-random ranking by ROC-AUC/AP under clean-training shift.

Equally important, the audit prevents overclaiming. Relation camouflage was too weak, non-oracle rewiring barely changed true-label heterophily, oracle topology used labels across the full graph, and one split cannot establish general rankings. The final report should therefore present R6 as controlled fixed-split evidence and a demonstration of why robustness benchmarks need protocol separation, perturbation-fidelity audits, seed-aware pairing, and explicit claim scope.

The result is useful precisely because it distinguishes three things that are often conflated: **a strong detector, an invariant detector, and a detector that can adapt after seeing the shifted regime.** PMP, MLP, and oracle-retrained SEC-GFD respectively illustrate why those are different properties.
