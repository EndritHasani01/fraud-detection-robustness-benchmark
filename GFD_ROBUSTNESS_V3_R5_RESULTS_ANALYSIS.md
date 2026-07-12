# Full Results Analysis: YelpChi Robustness Benchmark v3, Kaggle R5

## Executive verdict

The R5 experiment is a complete, internally consistent, and unusually well-provenanced course-project run. It contains all 504 expected successful model evaluations, covers both evaluation protocols, has no duplicate scientific keys, has no missing or failed rows, and links every performance row to a deterministic graph variant and a perturbation audit. The notebook executed without a visible error output, and the report bundle, upstream commits, compatibility patches, package environment, dataset files, and generated artifacts are all fingerprinted.

The scientific result is more nuanced than a single “most robust model” ranking:

1. **PMP is the strongest detector in absolute terms in this run.** It leads all three clean metrics and has the highest maximum-stress Average Precision (AP) in 8 of the 10 protocol-by-scenario comparisons.
2. **Feature camouflage is the only stress that consistently damages all four models.** At maximum test-time shift, every model loses approximately 21% of its clean AP.
3. **Oracle heterophily is a privileged-label diagnostic, not a realistic attack.** SEC-GFD reaches essentially perfect AP when trained on the label-constructed oracle graph, but collapses to AP 0.159 when trained clean and evaluated on that graph. The same topology can therefore become either a highly learnable label code or a severe distribution shift, depending on protocol.
4. **The tested relation-camouflage stress is underpowered.** It gives each selected fraud node only one net extra edge on a neighborhood of roughly 167 edges and changes the local fraud-to-normal neighbor ratio by only about 0.7 percentage points. Near-zero performance movement does not establish relation-camouflage robustness.
5. **The non-oracle rewiring path changes many edges but barely changes true-label heterophily.** At nominal severity 0.30 it rewires 2.415 million edges, yet heterophily moves only from 0.22688 to 0.23198. Its results describe feature-partition cross-rewiring, not a strong true-label heterophily intervention.
6. **The original R5 confidence intervals are not valid for the crossed seed design.** They flatten graph and training seeds and independently resample clean and stressed rows. An exact MLP graph-invariance effect of zero consequently receives a spurious AP-drop interval of approximately `[-0.063, +0.063]`. The point estimates are correct; the legacy uncertainty columns must be treated as superseded descriptive outputs.
7. **The evidence is limited to one dataset and one split.** Graph and training seeds quantify useful within-split variation, but they do not establish ranking stability across independently sampled train/validation/test partitions.

The defensible headline is therefore:

> On the fixed YelpChi split and integrated homogeneous adapters evaluated here, PMP had the strongest clean and usually the strongest stressed AP. Feature camouflage caused the broadest consistent degradation. Oracle heterophily exposed a very large protocol interaction—especially for SEC-GFD—while the relation-camouflage and non-oracle heterophily configurations did not realize equally strong stress. These are controlled, within-run robustness observations rather than universal model or deployment claims.

## 1. Scope and sources

This analysis is derived from the executed notebook and its latest complete artifact bundle:

- [Executed R5 notebook](KAGGLE_DUAL_T4_RESEARCH_RUN_with_outputs_latest_working.ipynb)
- [Frozen run config](gfd-robustness-v3-report-r5/config.json)
- [Unified raw results](gfd-robustness-v3-report-r5/results.csv)
- [Graph variant ledger](gfd-robustness-v3-report-r5/graph_variants.csv)
- [Perturbation audit](gfd-robustness-v3-report-r5/variant_audit.csv)
- [Kaggle run manifest](gfd-robustness-v3-report-r5/kaggle_run_manifest.json)
- [Run fingerprint](gfd-robustness-v3-report-r5/run_fingerprint.json)
- [Performance/audit join](gfd-robustness-v3-report-r5/plots/performance_audit_join.csv)
- [Maximum-stress summaries](gfd-robustness-v3-report-r5/plots/performance_drop_max_stress.csv)
- [Severity-curve summaries](gfd-robustness-v3-report-r5/plots/summary_curves.csv)
- [Robustness curve integrals](gfd-robustness-v3-report-r5/plots/robustness_scores.csv)
- [Missing/error diagnostic](gfd-robustness-v3-report-r5/plots/missing_or_error_runs.csv)

The interpretation is anchored in the intended methodology in [FINAL_PROJECT_GUIDE.md](FINAL_PROJECT_GUIDE.md), the model and evaluation cautions in [Paper_Summary.md](Paper_Summary.md), and the repository’s v3 design. AP is treated as the primary metric because YelpChi is imbalanced: fraud prevalence is approximately 14.53%, so a random ranking has expected AP close to 0.1453. ROC-AUC and macro-F1 remain important corroborating metrics, but macro-F1 also depends on the validation-selected threshold.

No value in this report is taken from an unexecuted hypothetical experiment. Proposed v4 changes are discussed separately in [GFD_ROBUSTNESS_RESEARCH_IMPROVEMENTS.md](GFD_ROBUSTNESS_RESEARCH_IMPROVEMENTS.md).

## 2. Integrity, completeness, and provenance

### 2.1 Notebook execution evidence

The executed notebook contains 90 cells: 66 code cells and 24 Markdown cells. Every code cell has an execution count, and no output has Jupyter’s `error` type. Execution count 56 is absent, which most likely means a cell was executed and later removed or reordered. This is a minor whole-notebook provenance imperfection, not evidence of a failed benchmark stage, because the later scientific-key, artifact, and hash gates all passed.

The notebook SHA-256 calculated from the checked-in executed file is:

```text
dbc164946f37bd31d6730dd6f4e94ba7abd214be985d6f80fa8fdabcdc0fd0e8
```

The R5 manifest does not record this whole-notebook hash. It does, however, record hashes for all 22 embedded benchmark modules, the inline config, the upstream patches, the package environment, the result table, and the report artifacts.

### 2.2 Reconstructed experiment matrix

The expected matrix can be derived independently from the config:

```text
1 clean state
+ 5 scenarios × 2 positive severities × 2 graph seeds
= 21 evaluated graph states

21 states × 3 training seeds = 63 rows per model/protocol
63 × 4 models = 252 rows per protocol
252 × 2 protocols = 504 total result rows
```

The graph ledger has 31 rows rather than 21 because it also retains explicit severity-zero references for each scenario and graph seed:

```text
1 clean ledger row + 5 scenarios × 3 severities × 2 graph seeds = 31
```

Observed checks:

| Check | Result |
|---|---:|
| Raw result rows | 504 |
| Successful rows | 504 |
| Error rows | 0 |
| Duplicate eight-field run keys | 0 |
| Missing expected keys | 0 |
| Unexpected keys | 0 |
| Rows per model/protocol | 63 |
| Missing/error diagnostic rows | 0 |
| `train_on_variant` rows trained on evaluated graph | 252/252 |
| `train_clean_eval_all` rows trained on clean graph | 252/252 |

The run key is correctly protocol-aware:

```text
(dataset_id, split_id, scenario_id, severity, graph_seed,
 training_seed, model_id, protocol)
```

This is important because the two protocols answer different questions and must never be averaged together.

### 2.3 Artifact fingerprints

Key hashes recorded or independently verified are:

| Object | SHA-256 |
|---|---|
| Config | `7e6b85461df2d7cbe7ba9704c252a741ffc6edf507773a33b1d5636f9b55ad43` |
| `results.csv` | `de73cbd331c7988eb2533a431b34c9f5e2c0a3ab5e3c8d06b16b93a9e19c7bd8` |
| Run fingerprint | `cd0fb3ee405665dc72f945e8a76345ae7421e3e992286322362aa8509a7539f2` |
| Report ZIP | `5cd78caa7b373350aa3b736434afe125a1a4c4611b5442e974c82779ed1edbcf` |

The ZIP contains 129 members, passes `ZipFile.testzip()`, and matches the extracted report directory member-for-member. The manifest pins the two active upstream revisions:

- PMP: `3f7629f6c180891a0bc1bba3c66d94d288a1ddae`
- SEC-GFD: `97faa51145ed1fbcbdc67cc5d399490da9a9797a`

It also fingerprints the compatibility patches and records the actual CUDA libraries loaded by DGL. This makes the execution evidence much stronger than a typical notebook-only submission.

### 2.4 What is not bundled

The 5.3 GiB graph cache is intentionally excluded from the lightweight report ZIP. Consequently, the bundle proves how to regenerate the variants but does not allow byte-for-byte verification of every graph binary without rerunning the notebook. Paths stored in result rows are absolute Kaggle paths and are dead outside the original session. Dataset file hashes and deterministic construction make regeneration credible, but the result bundle is not a self-contained graph archive.

## 3. Dataset and experimental design

### 3.1 YelpChi profile

| Property | Value |
|---|---:|
| Nodes | 45,954 |
| Homogeneous edges | 8,051,348 |
| Feature dimensions | 32 |
| Fraud nodes | 6,677 |
| Normal nodes | 39,277 |
| Fraud prevalence | 0.145297 |
| Train nodes | 18,380 |
| Validation nodes | 9,190 |
| Test nodes | 18,384 |
| Split | `s0`, seed 717 |
| Graph seeds | 0, 1 |
| Training seeds | 0, 1, 2 |

The split is stratified and fixed across all scenarios and severities. That is necessary for fair within-split stress comparisons, but it means the experiment cannot quantify split-to-split variation.

### 3.2 Models as actually evaluated

The model labels refer to integrated benchmark adapters, not unqualified reproductions of every paper configuration:

- **MLP** consumes node features only and is the graph-blind negative control.
- **GraphSAGE** is the standard message-passing baseline.
- **PMP** uses the upstream `LASAGE_S` implementation with sampled fanout 10, batch size 512, 100 epochs, patience 10, and one homogeneous `_E` relation.
- **SEC-GFD** uses hidden dimension 32, polynomial order 2, one high-order component, and the disclosed benchmark repairs for training-index correctness and finite cosine-domain loss.

YelpChi begins as a three-relation heterograph, and relation-aware perturbations are generated there. All four downstream adapters, however, consume the same homogeneous graph. This gives a common evaluation surface but removes the native relation-specific behavior central to PMP. “PMP” below should therefore be read as the homogeneous, reduced-cost PMP adapter in this benchmark.

### 3.3 Protocol semantics

`train_on_variant` trains and selects a validation threshold on each stressed graph. It measures how well an algorithm can fit or adapt to a changed environment when the stressed structure is available during training.

`train_clean_eval_all` trains once on the clean graph for each training seed, chooses its threshold using clean validation data, and evaluates that fixed artifact on every stressed graph. It measures test-time distribution-shift sensitivity.

For threshold-free AP and ROC-AUC, the protocol gap mainly reflects retraining/adaptation. For macro-F1, it also reflects the fact that the variant protocol may select a different validation threshold. Oracle `train_on_variant` has an additional and crucial distinction: the training graph was constructed with true labels from all graph partitions, including test-node labels. Those rows are privileged-label diagnostics, not operational rankings.

## 4. Did the perturbations realize meaningful stress?

Requested-count completion is excellent: every requested node or edge count was realized exactly. Scientific stress strength is not equal across scenarios, though.

| Scenario | Severity | Requested and realized | Principal measured effect |
|---|---:|---:|---|
| Oracle heterophily | 0.15 | 1,207,702 rewires | heterophily `0.22688 → 0.34292` (`+0.11604`) |
| Oracle heterophily | 0.30 | 2,415,404 rewires | heterophily `0.22688 → 0.45874` (`+0.23186`) |
| Non-oracle rewiring | 0.15 | 1,207,702 rewires | true-label heterophily approximately `+0.00300` |
| Non-oracle rewiring | 0.30 | 2,415,404 rewires | true-label heterophily approximately `+0.00510` |
| Feature camouflage | 0.15 | 1,001 fraud nodes | sampled-normal cosine approximately `+0.15951` |
| Feature camouflage | 0.30 | 2,003 fraud nodes | sampled-normal cosine approximately `+0.16021` |
| Relation camouflage | 0.15 | 1,001 selected fraud nodes | local normal-neighbor ratio approximately `+0.00715`; net `+1,001` edges |
| Relation camouflage | 0.30 | 2,003 selected fraud nodes | local normal-neighbor ratio approximately `+0.00703`; net `+2,003` edges |
| Uniform noise | 0.10 | 805,134 added edges | heterophily approximately `+0.00192` |
| Uniform noise | 0.20 | 1,610,269 added edges | heterophily approximately `+0.00353` |

### 4.1 Oracle heterophily

This is a strong structural intervention. At maximum severity it rewires exactly 30% of the 8.05 million edges and approximately doubles the cross-label edge ratio. It uses true labels to choose opposite-label destinations, so it is a controlled oracle probe. The resulting graph does not merely contain “more noise”: it contains a highly systematic class-partition pattern.

That distinction explains why a model trained on the oracle graph can improve dramatically. A heterophily-aware model can learn that opposite-label neighborhoods are predictive. A model trained on the clean regime may instead interpret the reversed structure incorrectly.

### 4.2 Non-oracle feature-partition rewiring

The same number of edges is rewired as in the oracle case, but the feature-derived two-means partition is only weakly aligned with the true fraud partition. The maximum true-label heterophily movement is roughly 2.2% of the oracle heterophily movement at the same nominal severity.

There is a second confound in R5: severity is part of the perturbation RNG seed, so the two-means partition is refit differently at each severity. Pseudo-positive rates change from approximately 0.57–0.61 at severity 0.15 to 0.386–0.393 at severity 0.30. The curve therefore changes both the number of rewired edges and the latent partition used to choose targets. A non-monotonic performance curve cannot be read as a pure dose response.

### 4.3 Feature camouflage

At maximum severity, 2,003 of 6,677 fraud nodes—30.00%—receive an exact normal-node feature vector because `gamma=1.0`. Their mean cosine similarity to their sampled normal donor becomes 1.0 by construction, from approximately 0.840 before replacement. This is strong and intentionally artificial. It cleanly tests reliance on node attributes and is the only scenario that should affect the MLP.

The mean per-changed-node cosine shift is similar at 0.15 and 0.30 because severity changes coverage, not replacement strength. A plot of the mean cosine shift alone hides that twice as many fraud nodes are affected at severity 0.30; the coverage count must be shown alongside it.

### 4.4 Relation camouflage

At maximum severity the scenario selects 2,003 fraud nodes, adds 4,006 fraud-to-normal edges, removes 2,003 suspicious edges, and therefore adds only 2,003 net edges to an 8.05-million-edge graph—0.0249%. The mean selected-node out-degree moves from roughly 167–169 to roughly 168–170: one net edge per selected node.

The local fraud-to-normal neighbor ratio moves by only about 0.007. This configuration is useful as a weak-stress calibration point, but it does not create enough evidence to conclude that relation camouflage is harmless. The correct conclusion is “inconclusive because realized stress was weak.”

### 4.5 Uniform random-edge noise

The graph grows by 10% and 20%, increasing mean out-degree from 175.20 to 210.25 at maximum severity. Because the added endpoints are random, global heterophily changes only slightly. This is still a meaningful density and irrelevant-neighborhood stress.

R5 assigns additions approximately uniformly across the three relations even though their original sizes differ sharply. This over-inflates the smallest relation relative to its baseline. The main results therefore combine density stress with a relation-composition shift before homogenization.

## 5. Clean performance

Means and sample standard deviations below are over three training seeds. The protocols retrained their clean cases separately.

| Protocol | Model | ROC-AUC | Average Precision | Macro-F1 |
|---|---|---:|---:|---:|
| Clean-train shift | MLP | 0.7813 ± 0.0288 | 0.3994 ± 0.0547 | 0.6532 ± 0.0236 |
| Clean-train shift | SAGE | 0.7921 ± 0.0078 | 0.4299 ± 0.0159 | 0.6708 ± 0.0058 |
| Clean-train shift | PMP | **0.8492 ± 0.0038** | **0.5677 ± 0.0134** | **0.7197 ± 0.0020** |
| Clean-train shift | SEC-GFD | 0.8137 ± 0.0033 | 0.4736 ± 0.0030 | 0.6891 ± 0.0034 |
| Train on variant | MLP | 0.7813 ± 0.0288 | 0.3994 ± 0.0547 | 0.6532 ± 0.0236 |
| Train on variant | SAGE | 0.7920 ± 0.0075 | 0.4293 ± 0.0153 | 0.6709 ± 0.0048 |
| Train on variant | PMP | **0.8492 ± 0.0038** | **0.5677 ± 0.0134** | **0.7197 ± 0.0020** |
| Train on variant | SEC-GFD | 0.8122 ± 0.0028 | 0.4704 ± 0.0022 | 0.6870 ± 0.0035 |

PMP is the clean leader on all three metrics. SEC-GFD is second, GraphSAGE third, and MLP fourth on AP.

The clean PMP and MLP rows are bit-identical between protocols. GraphSAGE and SEC-GFD have small clean discrepancies despite matching seeds and data: maximum per-seed AP discrepancies are about 0.0049 and 0.0045, respectively. CUDA was seeded but not forced into bitwise deterministic execution. Small protocol gaps near this scale should not be overinterpreted.

## 6. Maximum-stress Average Precision

Each table cell reports:

```text
stressed AP (clean AP − stressed AP)
```

A positive parenthesized value is degradation. A negative value is improvement relative to the separately trained clean baseline for that protocol.

### 6.1 Train on every variant

| Scenario | MLP | SAGE | PMP | SEC-GFD |
|---|---:|---:|---:|---:|
| Oracle heterophily | 0.3994 (+0.0000) | 0.6532 (−0.2239) | 0.6018 (−0.0341) | **1.0000 (−0.5296)** |
| Non-oracle rewiring | 0.3994 (+0.0000) | 0.3465 (+0.0828) | **0.5094 (+0.0583)** | 0.4838 (−0.0133) |
| Feature camouflage | 0.3213 (+0.0781) | 0.3118 (+0.1175) | **0.4174 (+0.1503)** | 0.3949 (+0.0756) |
| Relation camouflage | 0.3994 (+0.0000) | 0.3902 (+0.0391) | **0.5491 (+0.0186)** | 0.4707 (−0.0002) |
| Uniform noise | 0.3994 (+0.0000) | 0.2970 (+0.1323) | **0.5228 (+0.0450)** | 0.4765 (−0.0061) |

### 6.2 Train clean, evaluate every variant

| Scenario | MLP | SAGE | PMP | SEC-GFD |
|---|---:|---:|---:|---:|
| Oracle heterophily | **0.3994 (+0.0000)** | 0.3842 (+0.0457) | 0.3859 (+0.1818) | 0.1590 (+0.3145) |
| Non-oracle rewiring | 0.3994 (+0.0000) | 0.4157 (+0.0143) | **0.4895 (+0.0783)** | 0.4582 (+0.0153) |
| Feature camouflage | 0.3170 (+0.0824) | 0.3372 (+0.0927) | **0.4471 (+0.1206)** | 0.3714 (+0.1021) |
| Relation camouflage | 0.3994 (+0.0000) | 0.4297 (+0.0003) | **0.5684 (−0.0007)** | 0.4741 (−0.0005) |
| Uniform noise | 0.3994 (+0.0000) | 0.4211 (+0.0088) | **0.5178 (+0.0499)** | 0.4646 (+0.0090) |

PMP has the highest maximum-stress AP in eight of ten cells. The exceptions are not ordinary losses:

- Under oracle heterophily with retraining, SEC-GFD exploits the privileged label-derived graph and becomes nearly perfect.
- Under oracle heterophily shift, the graph-blind MLP retains 0.3994 AP and narrowly exceeds the graph models because it ignores the reversed topology.

This is why “smallest drop” and “best detector under stress” must be reported separately. The MLP has zero drop under graph-only changes by construction, but it starts below PMP and SEC-GFD on clean data.

## 7. Scenario-by-scenario interpretation

### 7.1 Oracle heterophily: the decisive protocol interaction

At severity 0.30, SEC-GFD’s outcomes are:

| Protocol | ROC-AUC | AP | Macro-F1 |
|---|---:|---:|---:|
| Train on oracle variant | approximately 1.000 | approximately 1.000 | approximately 1.000 |
| Train clean, evaluate oracle variant | 0.5347 | 0.1590 | 0.5046 |

Random-ranking AP is approximately 0.1453, so clean-trained SEC-GFD is only slightly above random AP. The paired protocol difference in maximum-stress AP is approximately `+0.841` in favor of retraining.

The result is scientifically interesting precisely because it is not a normal robustness victory. Oracle rewiring uses the true class partition, including test labels, to create opposite-label edges. A model trained on this graph can learn a near-deterministic heterophilous rule. A model trained on the clean graph expects the original structure and encounters a regime reversal.

GraphSAGE and PMP show the same direction but smaller magnitudes: maximum-stress retraining gains are about +0.269 and +0.216 AP. The MLP is unchanged. This validates that the interaction lives in the graph topology, not in node features.

Claim scope:

- `train_clean_eval_all` is a valid oracle sensitivity diagnostic: “How badly does a fixed clean model fail under a deliberately label-informed structural shift?”
- `train_on_variant` is a privileged-label training diagnostic: “Can the adapter learn the synthetic class-coded topology?”
- Neither supports a production adversarial-robustness claim.

### 7.2 Feature camouflage: the most consistent failure mode

At maximum clean-trained shift, relative AP declines are:

| Model | Relative AP decline |
|---|---:|
| MLP | 20.6% |
| SAGE | 21.6% |
| PMP | 21.2% |
| SEC-GFD | 21.6% |

This tight grouping is substantive. All four models consume node features, and exact feature replacement attacks that shared input channel. Message passing does not compensate enough to prevent a roughly one-fifth relative AP loss.

Under retraining, SEC-GFD retains the largest fraction of its clean AP, but PMP remains the best absolute detector: AP 0.417 versus 0.395 for SEC-GFD. This illustrates an important reporting distinction:

- SEC-GFD has the smaller relative degradation.
- PMP still makes the stronger stressed ranking.

Both facts can be true, and neither alone defines a universal winner.

### 7.3 Non-oracle rewiring: modest shift, unstable adaptation

Under clean-trained shift, maximum relative AP declines are modest for GraphSAGE and SEC-GFD—approximately 3.3% and 3.2%—but larger for PMP at about 13.8%. PMP nevertheless retains the highest absolute stressed AP, 0.4895.

Retraining is not universally helpful. Relative to clean-trained evaluation on the same maximum variant:

- SAGE loses another 0.069 AP when retrained.
- PMP gains about 0.020.
- SEC-GFD gains about 0.026.

Because the realized true-label heterophily shift is only about 0.005, the scenario should not be described as strong non-oracle heterophily. The R5 severity-dependent feature partition also prevents a clean monotonic interpretation.

### 7.4 Relation camouflage: inconclusive, not robust

Clean-trained AP is essentially flat for all four models. That is consistent with the audit: a net one-edge change per selected fraud node is tiny relative to the node’s existing neighborhood.

The variant-trained SAGE and PMP means are lower by approximately 0.039 and 0.019 AP, while SEC-GFD and MLP remain nearly unchanged. With only two graph seeds and three training seeds—and a weak realized intervention—these differences cannot establish a general model ordering for relation camouflage.

The safe result sentence is:

> The R5 relation-camouflage configuration did not materially move clean-trained performance, but its audit showed only a 0.7-percentage-point local neighbor-composition shift, so this is an underpowered stress result rather than evidence that the models resist relation camouflage.

### 7.5 Uniform noise: PMP remains strongest; SAGE adaptation is fragile

Under clean-trained shift, maximum AP losses are small for SAGE (0.0088) and SEC-GFD (0.0090), moderate for PMP (0.0499), and exactly zero for MLP. PMP still has the highest absolute stressed AP at 0.5178.

Variant-trained SAGE falls to AP 0.2970, producing an adaptation gap of approximately −0.124 AP relative to the clean-trained SAGE evaluated on the same noisy graph. PMP and SEC-GFD change little between protocols.

This SAGE behavior appears related to optimization instability rather than a smooth structural response, as discussed below.

## 8. Direct protocol contrasts

Maximum-stress paired AP contrast is defined as:

```text
train_on_variant AP − train_clean_eval_all AP
```

| Scenario | MLP | SAGE | PMP | SEC-GFD |
|---|---:|---:|---:|---:|
| Feature camouflage | +0.0043 | −0.0254 | −0.0297 | +0.0234 |
| Relation camouflage | +0.0000 | −0.0394 | −0.0194 | −0.0034 |
| Non-oracle rewiring | +0.0000 | −0.0692 | +0.0199 | +0.0255 |
| Oracle heterophily | +0.0000 | +0.2690 | +0.2159 | +0.8410 |
| Uniform noise | +0.0000 | −0.1241 | +0.0050 | +0.0119 |

Three conclusions follow:

1. Adaptation is scenario- and model-dependent; it is not automatically beneficial.
2. Oracle heterophily is qualitatively different from the other stresses because the variant graph encodes privileged class structure.
3. SAGE’s negative adaptation gaps under non-oracle rewiring and noise are too large to ignore and are connected to training instability.

Small nonzero MLP feature-camouflage gaps and small clean-model gaps can arise because the protocols independently retrain. For graph-only stresses, the MLP is exactly invariant within each training seed.

## 9. Negative control and internal validity

The MLP is an excellent mechanistic control:

- Its AP is exactly unchanged under every graph-only variant when matched by training seed.
- It changes under feature camouflage because that scenario edits its actual input.
- Its maximum feature-camouflage AP decline is approximately 0.082 in the shift protocol.

This confirms that graph-only transformations did not accidentally alter labels, masks, or features. It also exposes the original reporting bug: a paired effect that is exactly zero should have zero empirical paired SD and a zero-width paired interval. The R5 flat independent bootstrap instead attributed clean-training variation to the graph perturbation effect.

The MLP should not be called the most robust graph fraud model merely because it has zero graph-only drop. It is a graph-blind baseline with lower clean AP. Its role is to separate graph sensitivity from feature sensitivity.

## 10. Seed behavior and uncertainty

### 10.1 Experimental units

Positive-severity cells contain a crossed design:

```text
2 graph realizations × 3 training realizations = 6 rows
```

The clean graph has one physical graph state and three training realizations. Under `train_clean_eval_all`, every stressed graph with the same training seed is evaluated using the same trained artifact. Rows are therefore dependent across graph variants and severity values.

The correct descriptive analysis must preserve those identities:

- Pair clean and stressed performance by training seed.
- Resample training-seed and graph-seed axes hierarchically, not six flattened rows.
- Use the same sampled seed identities across the entire severity trajectory.
- Calculate worst case, curve integral, and retention on each matched trajectory before aggregation.

### 10.2 Why the legacy R5 intervals are superseded

The original R5 plotting code independently bootstraps clean and stressed values and independently resamples each severity. Its `drop_std` is the standard deviation of bootstrap mean differences—closer to a bootstrap standard error—while `clean_std` is an empirical sample SD. The identically named `std` concept is therefore inconsistent across columns.

Concrete failure:

```text
MLP graph-only paired AP drops: [0, 0, 0, 0, 0, 0]
Correct paired mean / SD / CI:   0 / 0 / [0, 0]
Legacy independent CI:          approximately [-0.063, +0.063]
```

The exported point means and ordinary cell sample SDs reconcile exactly to the raw results and remain usable. The legacy bootstrap drop and robustness intervals must not be used as significance evidence. The implemented seed-aware reporting replacement is described in the improvement document.

### 10.3 One split remains the main inferential limit

Every R5 `*_cross_split.csv` row has `n_splits=1`. Those files are schema-compatible single-split rollups, not additional replication. A zero cross-split SD or point interval with one split is not evidence of certainty.

The analysis is properly labeled `fixed_split_descriptive`. More bootstrap resamples cannot solve the absence of independent splits.

## 11. GraphSAGE instability

The most important training anomaly is in `train_on_variant` SAGE:

- 20 of 63 rows finish within two seconds and have mean AP approximately 0.222.
- The other 43 rows take more than three seconds and have mean AP approximately 0.452.
- Duration and AP have correlation approximately 0.808.
- Maximum-stress AP SD is around 0.10 for relation camouflage, non-oracle rewiring, and noise.
- Training seed 1 frequently belongs to the collapsed mode.

The pattern is consistent with early stopping after a poor validation trajectory, but R5 does not record epoch count, best epoch, best validation monitor, or stop reason. Duration is only an indirect proxy, so the cause cannot be proven from the bundle.

This anomaly affects interpretation in two ways:

1. SAGE’s poor variant-trained noise and non-oracle results may be optimization failures rather than inevitable limits of the architecture.
2. Means based on three training seeds are sensitive to one collapsed seed, which is another reason not to claim a stable model ranking.

The implemented next-run result schema now records explicit training diagnostics so this failure mode can be audited directly.

## 12. Model-specific assessment

### 12.1 MLP

Strengths:

- Clean AP 0.399 is meaningfully above the 0.145 prevalence baseline.
- Exact graph-only invariance validates the experiment mechanics.
- It becomes the best absolute model under maximum oracle test-time shift because every graph model is affected by the reversed topology.

Limits:

- It is the weakest or second-weakest clean detector.
- Zero structural drop is not evidence of graph robustness; it never reads the graph.
- It loses about one-fifth of AP under feature camouflage.

### 12.2 GraphSAGE

Strengths:

- Improves over MLP on clean data.
- Has modest clean-trained degradation under non-oracle rewiring and noise.
- Can learn the oracle heterophilous topology when retrained.

Limits:

- Variant-trained outcomes are bimodal and strongly associated with short runs.
- Retraining hurts substantially under maximum non-oracle rewiring and noise.
- With only three training seeds, its mean is particularly fragile.

### 12.3 PMP

Strengths:

- Best clean ROC-AUC, AP, and macro-F1.
- Highest maximum-stress AP in eight of ten protocol/scenario cells.
- Highest average AP over all five shift curves and four of five variant-training curves.
- Remains the best absolute model under feature camouflage despite a nontrivial relative loss.

Limits:

- Loses more AP than SAGE or SEC-GFD under non-oracle and noise shift, even while remaining strongest absolutely.
- Dominates runtime: the variant-training lane takes approximately 42.9 minutes.
- The homogeneous one-relation, fanout-10 adapter is not the paper’s native multi-relation setup.

### 12.4 SEC-GFD

Strengths:

- Second-best clean AP.
- Small clean-trained changes under non-oracle rewiring and uniform noise.
- Best relative feature-camouflage retention under variant retraining.
- Very strong capacity to exploit deliberately heterophilous structure.

Limits:

- Catastrophic oracle test-time shift: AP falls to 0.159.
- Near-perfect oracle retraining is a privileged-label diagnostic, not deployment evidence.
- The adapter contains necessary benchmark repairs and reduced-cost hyperparameters, so claims apply to this integration.

## 13. Runtime interpretation

| Lane | Wall time (seconds) |
|---|---:|
| Variant PMP | 2,570.95 |
| Variant SEC-GFD | 310.37 |
| Variant MLP/SAGE | 278.33 |
| Shift PMP | 193.21 |
| Shift MLP/SAGE | 67.67 |
| Shift SEC-GFD | 46.73 |

PMP is clearly the computational bottleneck in this implementation. These are lane wall times from a concurrent dual-GPU schedule, not controlled model-speed measurements. Processes share CPU, RAM, disk, and data-loading resources, so the numbers support capacity planning only.

The sum of per-row recorded duration is about 0.94 hours, while the end-to-end notebook includes environment setup, dataset and graph construction, smoke tests, concurrent training lanes, reporting, and packaging. Neither measure should be presented as a fair architecture benchmark.

## 14. Threats to validity

### Internal validity

Strong controls:

- Fixed masks across severity.
- Separate graph and training seeds.
- Validation-only early stopping and threshold selection.
- No test-metric tuning.
- Deterministic graph cache and perturbation audit.
- Exact scientific-key completeness checks.
- MLP graph-only invariant.

Remaining threats:

- Oracle construction reads all labels, including test-node labels; this is a disclosed controlled exception to ordinary no-test-label use.
- CUDA is seeded but not bitwise deterministic.
- R5 severity realizations are not nested.
- SAGE stopping telemetry is absent.
- The original uncertainty procedure breaks pairing.

### Construct validity

- The benchmark measures robustness to static synthetic graph/feature transformations, not adaptive fraud behavior.
- Nominal severity is family-specific and cannot be compared across scenarios.
- Relation camouflage and non-oracle true-label heterophily are weakly realized in R5.
- Oracle heterophily may make labels easier to infer under retraining rather than simply “damage” the graph.

### External validity

- One YelpChi version.
- One stratified split.
- No second dataset.
- Homogeneous evaluation suppresses native relation semantics.
- Integrated PMP and SEC-GFD adapters use feasible rather than exact paper settings.

### Statistical conclusion validity

- Three training seeds and two graph seeds are a small descriptive sample.
- The six stress rows are not iid.
- No independent split replication exists.
- Original bootstrap intervals are superseded.
- No formal significance or stable-ranking claim is justified.

## 15. Claims that are and are not supported

### Supported

- PMP had the strongest clean performance in this run.
- PMP usually retained the strongest absolute stressed AP.
- Feature camouflage consistently reduced AP for all four integrated models.
- Oracle heterophily created a large protocol interaction, especially for SEC-GFD.
- MLP was exactly invariant to graph-only changes, as expected.
- Non-oracle rewiring and random noise caused modest clean-trained changes for SAGE and SEC-GFD, while PMP remained strongest absolutely.
- Variant-trained SAGE showed substantial training instability.
- The tested relation-camouflage configuration was too weak for a general robustness conclusion.

### Not supported

- A single universal “most robust” model.
- Production adversarial robustness.
- Realistic benefits from oracle-label perturbations.
- A claim that SEC-GFD’s oracle AP near 1 proves robustness.
- A claim that relation camouflage generally has no effect.
- Formal significance from the legacy R5 intervals.
- Stable rankings across data splits or datasets.
- Exact reproduction of the PMP or SEC-GFD papers.
- Runtime superiority from concurrent lane durations.

## 16. Recommended report narrative

A coherent final-project narrative is:

1. The project contributes a reproducible, common robustness harness rather than another detector.
2. Clean performance alone is insufficient: model responses depend strongly on both stress family and protocol.
3. Absolute stressed performance and relative degradation answer different questions; both must be shown.
4. Feature camouflage is the clearest shared failure mode.
5. Oracle heterophily demonstrates that label-informed topology can be either a powerful learnable signal or a severe shift, which is why oracle status and training exposure must remain visible.
6. Auditing realized stress is essential: equal requested counts do not create equal scientific interventions.
7. The main remaining research need is independent split replication after scenario calibration, not additional models.

## 17. Final conclusion

R5 successfully answers the engineering question: the benchmark can run four integrated models, five audited stress scenarios, two protocols, crossed graph/training seeds, and a complete report pipeline on dual T4 GPUs with strong completion evidence.

It answers the scientific question more carefully:

- PMP is the best absolute detector in most conditions tested.
- Feature camouflage is a broad and consistent weakness.
- Graph-only robustness cannot be summarized by one drop number because the MLP is invariant by construction and because oracle topology can become a learnable label code.
- SEC-GFD is highly sensitive to unseen oracle heterophily but highly effective when allowed to learn that privileged graph.
- The weak relation-camouflage and non-oracle heterophily realizations limit claims for those scenarios.
- The single split and original flat bootstrap prevent inferential overstatement.

The run is therefore valuable and report-worthy, but its strongest contribution is a traceable controlled comparison and the methodological lesson that **robustness claims are only as meaningful as the protocol, perturbation realization, statistical unit, and label-access scope attached to them**.
