# V3 Benchmark Reference

This document is the authoritative reference for the current Iteration 3 benchmark. It explains the benchmark as implemented in the repository after the v3 scenario, audit, testing, and reporting work.

Use this file when you need to answer any of the following questions:

- What does each scenario actually do?
- Which scenarios are oracle and which are non-oracle?
- What do the two evaluation protocols mean?
- How do `graph_variants.csv`, `variant_audit.csv`, `results.csv`, and the plot outputs relate to each other?
- What conclusions are safe, and what would over-claim the benchmark?

## 1. What The Benchmark Is

This project is a reproducible static robustness benchmark for graph-based fraud detection on YelpChi. It starts from a fixed dataset and split definition, generates deterministic stressed graph variants, evaluates multiple models under two protocols, and exports both performance tables and perturbation audits.

The benchmark is strongest when it is described as a controlled synthetic stress-test system. It is not a temporal simulation, not an attacker-policy benchmark, and not a realistic operational fraud environment.

The current frozen v3 configs are:

- `configs/exp_yelpchi_v3_fast.json`: smoke-test run that exercises the full v3 scenario surface with minimal seeds and coarse severities.
- `configs/exp_yelpchi_v3.json`: the main report-facing v3 benchmark.
- `configs/exp_yelpchi_v3_full.json`: denser severity curves and broader seed coverage for fuller robustness analysis.

## 2. Current V3 Workflow

The standard v3 workflow is:

```powershell
py -m benchmark.run --config configs/exp_yelpchi_v3.json --stage graphs
py -m benchmark.run --config configs/exp_yelpchi_v3.json --stage matrix
py -m benchmark.run --config configs/exp_yelpchi_v3.json --stage matrix --protocol train_clean_eval_all
py -m benchmark.run --config configs/exp_yelpchi_v3.json --stage plots --ci
```

`graphs` builds the clean graph and stressed variants, `matrix` runs the configured model set under one protocol, and `plots` generates report-ready summaries plus audit-oriented outputs.

## 3. Scenario Catalog

All v3 scenarios are declared in frozen configs. The main v3 config uses the following scenario IDs.

| Scenario ID | Family | Oracle | Graph view mode | Severity parameter |
| --- | --- | --- | --- | --- |
| `heterophily_rewire_oracle` | `heterophily` | Yes | `heterograph_aware_generation` | `p_rewire` |
| `heterophily_rewire_nonoracle` | `heterophily` | No | `heterograph_aware_generation` | `p_rewire` |
| `camouflage_feature_oracle` | `camouflage_feature` | Yes | `canonical` | `p_cam_feat` |
| `camouflage_relation_oracle` | `camouflage_relation` | Yes | `heterograph_aware_generation` | `p_cam_rel` |
| `noise_edges_uniform` | `noise` | No | `heterograph_aware_generation` | `edge_noise_rate` |

The YelpChi relation filter used in the frozen v3 configs is `net_rsr`, `net_rtr`, and `net_rur`. Relation-aware scenarios are generated on the source heterograph and then converted back to the canonical evaluation view configured for the benchmark.

### 3.1 `heterophily_rewire_oracle`

Mechanism:
Select a fraction of eligible edges, keep each source node fixed, and rewire the destination toward a true opposite-label target while respecting the configured edge-sampling policy.

What severity controls:
`p_rewire` is the requested fraction of eligible selected edges. Requested severity is not the same thing as realized change. The realized count can be lower if the edge-sampling policy rejects too many candidates.

Audit evidence to read:

- `n_rewired_edges_selected`
- `n_rewired_edges_actual`
- `heterophily_ratio_before`
- `heterophily_ratio_after`
- `requested_change` and `realized_change`
- `sampling_policy_json`

Interpretation:
This is an oracle structural stress. It is useful for benchmarking sensitivity to misleading neighborhood labels, but it is not a realistic live attack construction because it uses true labels.

### 3.2 `heterophily_rewire_nonoracle`

Mechanism:
Use feature-derived pseudo partitions instead of true labels when choosing rewiring targets. The implementation uses a simple two-means style split in feature space, then rewires toward pseudo opposite partitions under the same edge-sampling policy used by the oracle version.

What severity controls:
The same `p_rewire` meaning as the oracle version. The non-oracle path still reports selected versus realized rewires because policy rejections can reduce the realized change.

Audit evidence to read:

- `n_rewired_edges_selected`
- `n_rewired_edges_actual`
- `heterophily_ratio_before`
- `heterophily_ratio_after`
- `pseudo_label_pos_rate`
- `requested_change` and `realized_change`

Interpretation:
This is the non-oracle heterophily reference. It avoids true labels during target construction, but it is still a synthetic benchmark perturbation rather than a learned attacker model.

### 3.3 `camouflage_feature_oracle`

Mechanism:
Select a severity-controlled fraction of fraud nodes and replace or blend their features with sampled normal-node features. In the frozen v3 configs, `gamma = 1.0`, so the selected fraud-node features are fully replaced by the sampled normal-node features.

What severity controls:
`p_cam_feat` is the requested fraction of fraud nodes selected for feature editing.

Audit evidence to read:

- `n_changed_nodes_requested`
- `n_changed_nodes_actual`
- `mean_cosine_to_sampled_normal_before`
- `mean_cosine_to_sampled_normal_after`
- `mean_l2_delta`
- `gamma`

Interpretation:
This is feature-only camouflage. It edits node features and does not change graph structure. It is oracle because true labels are used to identify which nodes are fraud nodes.

### 3.4 `camouflage_relation_oracle`

Mechanism:
Select a severity-controlled fraction of fraud nodes, add new edges from them to sampled normal nodes on the selected relations, and optionally remove a fraction of suspicious fraud-heavy edges. The frozen v3 configs use `camouflage_edges_per_node = 2` and `remove_suspicious_ratio = 0.5`.

What severity controls:
`p_cam_rel` controls the fraction of fraud nodes selected for relation camouflage. The net graph-size change depends on how many camouflage additions succeed and how many suspicious edges are removed.

Audit evidence to read:

- `n_camouflaged_nodes`
- `n_camouflaged_edges_requested`
- `n_camouflaged_edges_added`
- `n_suspicious_edges_removed_requested`
- `n_suspicious_edges_removed`
- `fraud_to_normal_neighbor_ratio_before`
- `fraud_to_normal_neighbor_ratio_after`
- `fraud_to_fraud_neighbor_ratio_before`
- `fraud_to_fraud_neighbor_ratio_after`
- `mean_selected_out_degree_before`
- `mean_selected_out_degree_after`

Interpretation:
This is structural camouflage. It is the main v3 addition that closes the earlier feature-only gap. It remains oracle because true labels are used to identify the fraud nodes to camouflage and the normal nodes used as benign structural targets.

### 3.5 `noise_edges_uniform`

Mechanism:
Sample new relation-aware edges uniformly on the selected relations and add them to the graph under the shared edge-sampling policy. The frozen v3 configs keep `undirected = false`, so the inserted edges are directed.

What severity controls:
`edge_noise_rate` controls the requested number of added edge pairs relative to the current eligible edge count. The realized number of added unique edges can be lower if self-loops, duplicates, or already-existing edges are rejected and the sampling budget is exhausted.

Audit evidence to read:

- `n_added_edge_pairs_requested`
- `n_added_edge_pairs_actual`
- `n_added_edges_requested`
- `n_added_edges_actual`
- `requested_change` and `realized_change`
- `sampling_policy_json`

Interpretation:
This is random density stress, not a targeted attacker. It is useful for comparing robustness to neighborhood clutter and graph densification.

## 4. Shared Edge-Sampling Policy

The default v3 policy is intentionally explicit and shared by rewiring and edge-addition scenarios:

- self-loops are disallowed unless a scenario explicitly enables them
- already-existing edges are rejected
- duplicate sampled edges are rejected
- candidates are resampled up to a bounded attempt budget

This has two important consequences.

First, the requested perturbation strength and realized perturbation strength can differ. For that reason, `variant_audit.csv` always matters more than the raw severity number when you interpret how strongly a graph was actually perturbed.

Second, rewiring and noise scenarios record both selected and realized changes. If the realized count is noticeably below the requested count, treat that as part of the benchmark evidence rather than an implementation nuisance.

## 5. Graph View Modes

The benchmark still evaluates all models on one canonical graph view for comparability. In the frozen v3 configs, the canonical view is homogeneous.

`graph_view_mode` tells you how the perturbation was generated before evaluation:

- `canonical`: the scenario edits the already-canonical evaluation graph
- `heterograph_aware_generation`: the scenario edits the source heterograph first and then converts back to the canonical evaluation graph

This distinction improves perturbation semantics without changing the shared downstream model interface. Relation-aware generation does not mean the downstream evaluation itself is heterograph-native.

## 6. Evaluation Protocols

The benchmark supports two protocols, and they answer different questions.

### 6.1 `train_on_variant`

Train the model separately on each stressed graph and evaluate on that same variant.

This answers:
How well can the model adapt when training data and evaluation data are already stressed in the same way?

Use this protocol when the claim is about performance under stressed training conditions.

### 6.2 `train_clean_eval_all`

Train the model once on the clean graph and evaluate that trained model on all stressed variants.

This answers:
How much performance is lost under test-time distribution shift when the model was trained under normal conditions?

Use this protocol when the claim is about shift robustness rather than adaptation under stressed retraining.

Do not average results across the two protocols. The same metric curve means something different under each one.

## 7. Artifact Reference And Traceability

The main run directory is `runs/<experiment_name>/`.

### 7.1 `graph_variants.csv`

This is the variant ledger. It records which clean and stressed graph files exist, which scenario and severity they correspond to, whether a scenario was actually applied, and basic graph statistics.

Use it to answer:
Which graph variants exist, and what canonical graph statistics do they have?

### 7.2 `variant_audit.csv`

This is the main perturbation audit artifact. It is keyed by:

`(dataset_id, split_id, scenario_id, severity, graph_seed)`

It records:

- requested severity
- requested versus realized change
- before/after graph statistics
- scenario-specific realized-effect metrics
- graph-view mode
- oracle status

Use it to answer:
What was requested, what actually changed, and how should that scenario be interpreted?

### 7.3 `results.csv`

This remains the single unified evaluation table. Its run key is:

`(dataset_id, split_id, scenario_id, severity, graph_seed, training_seed, model_id, protocol)`

Use it to answer:
How did a model perform under one exact evaluation run?

Do not duplicate audit columns into `results.csv`. The v3 design keeps performance and perturbation evidence separate but joinable.

### 7.4 `plots/performance_audit_join.csv`

This is the report-stage join between `results.csv`, `graph_variants.csv`, and `variant_audit.csv`. It is useful when you want one table that shows performance metrics alongside the realized perturbation evidence used to justify the stress claim.

### 7.5 Plot-ready summary outputs

The plots stage emits:

- `plots/summary_curves.csv`
- `plots/performance_drop_max_stress.csv`
- `plots/robustness_scores.csv`
- `plots/summary_curves_cross_split.csv`
- `plots/performance_drop_max_stress_cross_split.csv`
- `plots/robustness_scores_cross_split.csv`
- `plots/audit_curves.csv`
- `plots/audit_curves_cross_split.csv`
- `plots/missing_or_error_runs.csv`

The audit CSVs expose realized perturbation behavior, while the performance CSVs expose model degradation. Both keep `protocol` and `oracle_labels` visible.

PNG files live under:

`plots/<protocol>/<dataset>/<split>/`

and across-split PNGs live under:

`plots/<protocol>/<dataset>/across_splits/`

## 8. How To Read Severity Responsibly

Severity is family-specific. A severity value of `0.3` in one family is not calibrated to mean the same realized stress as `0.3` in another family.

Use severity in the following order of trust:

1. within-family comparisons of the same scenario
2. realized audit metrics from `variant_audit.csv` or `plots/audit_curves.csv`
3. protocol-aware model comparisons under the same scenario and severity grid

Avoid direct statements like:
"Camouflage at 0.3 is stronger than heterophily at 0.3."

If you need a cross-family statement, base it on realized audit evidence, not on the raw severity parameter.

## 9. Oracle Disclosure Rules

Oracle scenarios use privileged information during perturbation construction. In the current v3 benchmark:

- `heterophily_rewire_oracle` is oracle
- `camouflage_feature_oracle` is oracle
- `camouflage_relation_oracle` is oracle
- `heterophily_rewire_nonoracle` is non-oracle
- `noise_edges_uniform` is non-oracle

Oracle scenarios are acceptable in this benchmark because the benchmark is a controlled stress-test system, not a deployable attack generator. The disclosure rule is simple:

- keep `oracle_labels` visible in downstream tables and plots
- mention oracle status explicitly in any report section that interprets those scenarios
- do not describe oracle scenarios as realistic operational attacks

## 10. Interpretation Boundaries

The benchmark is designed to support controlled comparative robustness analysis. It is not designed to support claims about full adversarial realism.

Safe claims include:

- one model degraded less than another under the same scenario family, severity, and protocol
- a scenario increased the realized stress proxy it was supposed to affect
- a conclusion was stable or unstable across graph seeds, training seeds, splits, or protocols

Claims to avoid include:

- this benchmark simulates realistic fraudster behavior end to end
- oracle perturbations represent deployable attacker behavior
- severity values are directly comparable across scenario families
- a model that is strong here is automatically production-ready

The current benchmark does not simulate:

- temporal fraud evolution
- attacker policy learning
- adaptive defender loops
- business-rule interventions
- online retraining systems
- manual review processes

## 11. Recommended Reporting Workflow

When writing results, use both the performance and audit artifacts together.

1. Use `results.csv` and the summary tables to identify which models degrade under which stress.
2. Use `variant_audit.csv` or `plots/audit_curves.csv` to confirm that the scenario actually changed the graph or features in the claimed direction.
3. Keep `protocol` explicit in every table, figure, and conclusion.
4. Keep `oracle_labels` explicit whenever an oracle scenario appears.
5. Use within-family trend language unless you have a realized audit metric that justifies a stronger statement.

## 12. Final Framing

The most accurate one-sentence description of the v3 system is:

> A reproducible, report-ready static robustness benchmark for graph-based fraud detection with relation-aware synthetic stress tests, direct perturbation audits, and protocol-aware evaluation.

That is the intended scope. The benchmark becomes weaker, not stronger, when it is described as a realistic adversarial simulator.
