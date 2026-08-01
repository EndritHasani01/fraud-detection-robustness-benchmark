# Benchmark Scenario Implementation Explainer

Date: 2026-04-24  
Repository: `fraud-detection-robustness-benchmark`

This document explains, in simple terms, how the benchmark stress scenarios are implemented in code, what they are intended to mean, and what must be checked before trusting the results.

The main implementation is in `benchmark/scenarios.py`. The graph-generation stage that calls these scenarios is in `benchmark/run.py`.

## Short Summary

The benchmark starts with the real YelpChi graph from DGL. It then creates synthetic stressed versions of the graph. Each scenario changes either edges, node features, or both. The models are then evaluated on those variants.

The scenarios are feasible and mostly logically valid for a controlled robustness benchmark. They are not realistic fraud simulations. They are better understood as lab stress tests:

- What happens if fraud features look more normal?
- What happens if edges connect more often to opposite-label nodes?
- What happens if fraud nodes connect to more normal nodes?
- What happens if random noise edges are added?

The most important rule is this:

> Do not trust the raw severity number alone. Always check `variant_audit.csv` to see what actually changed.

## How Scenario Generation Works Overall

The graph stage is implemented in `benchmark/run.py`.

For every dataset and split, it does this:

1. Load YelpChi from `dgl.data.FraudDataset`.
2. Cast features and labels to stable types.
3. Create deterministic train/validation/test masks.
4. Save the clean base graph.
5. For every configured scenario, severity, and graph seed:
   - choose whether to perturb the homogeneous graph or the source heterograph,
   - call `apply_scenario(...)`,
   - save the resulting graph,
   - write `graph_variants.csv`,
   - write `variant_audit.csv`.

The main dispatcher is `apply_scenario(...)` in `benchmark/scenarios.py`. It reads the scenario method from the config and calls the matching implementation.

The code also checks basic invariants after a successful perturbation:

- node count must not change,
- labels must not change,
- train/validation/test masks must not change,
- feature tensor shape must not change.

That is good. It means the scenarios stress the graph/features without changing the supervised task definition.

## Shared Edge-Sampling Rules

Several scenarios add or rewire edges. They use a shared edge-sampling policy from `benchmark/relation_utils.py`.

In the main v3 config:

- self-loops are rejected,
- existing edges are rejected,
- duplicate sampled edges are rejected,
- sampling retries are bounded by `max_attempt_multiplier`.

This matters because the requested number of changes may not equal the realized number of changes. For example, if the graph is dense, many candidate edges already exist and get rejected.

Before trusting a run, check these audit fields:

- `requested_change`
- `realized_change`
- `n_rewired_edges_selected`
- `n_rewired_edges_actual`
- `n_added_edge_pairs_requested`
- `n_added_edge_pairs_actual`
- rejection counters in `extra_info_json`

## Scenario 1: `heterophily_rewire_oracle`

### Plain Meaning

This scenario makes the graph less label-consistent. It rewires some edges so that a node connects to a node with the opposite true label.

In fraud terms:

- normal nodes get more fraud neighbors,
- fraud nodes get more normal neighbors,
- message passing becomes more misleading.

This directly tests sensitivity to heterophily.

### How It Is Implemented

Code path:

- `apply_scenario(...)`
- `_rewire_edge_dst_to_opposite_label(...)`
- `_apply_partition_based_rewire(...)`
- `_rewire_relation_edges(...)`

The implementation:

1. Read the true labels from `g.ndata["label"]`.
2. Select the configured relations, usually `net_rsr`, `net_rtr`, and `net_rur`.
3. Count eligible edges.
4. Select `int(p_rewire * edge_count)` edges.
5. Keep each selected edge source node unchanged.
6. Replace the destination with a randomly sampled node from the opposite true-label class.
7. Reject invalid candidate destinations:
   - same as old destination,
   - self-loop,
   - already-existing edge,
   - duplicate sampled edge.
8. Rebuild the graph with the modified edge list.
9. Record before/after heterophily ratio and selected/actual rewired edge counts.

Example:

```text
Original edge: fraud node -> fraud node
Rewired edge:  fraud node -> normal node
```

### Is It Feasible?

Yes. This is computationally feasible and conceptually clear. It is a valid controlled perturbation.

### Is It Logically Valid?

Yes, with one major disclosure: it is oracle-based. It uses true labels to build the stress.

That is acceptable for a benchmark if described as:

> an oracle stress test for label-disagreement neighborhoods.

It is not valid to describe it as:

> a realistic fraudster attack.

Real attackers would not know all true labels.

### What To Verify

Check `variant_audit.csv`:

- `heterophily_ratio_after` should be greater than `heterophily_ratio_before`.
- `n_rewired_edges_actual` should be close to `n_rewired_edges_selected`.
- rejection counts should not dominate the run.

If heterophily does not increase, the scenario did not work as intended.

### Weaknesses

- Uses privileged true labels.
- Does not preserve exact degree distribution for destination nodes.
- Does not model realistic fraud behavior.
- In dense graphs, replacement candidates may be rejected often.

## Scenario 2: `heterophily_rewire_nonoracle`

### Plain Meaning

This is a non-oracle version of heterophily rewiring. It tries to make edges cross between two feature-based groups instead of true label groups.

In simple terms:

> The code clusters nodes into two rough feature groups, then rewires edges toward the opposite feature group.

This avoids using true labels for construction.

### How It Is Implemented

Code path:

- `apply_scenario(...)`
- `_rewire_edge_dst_to_feature_pseudo_opposite_label(...)`
- `_two_means_binary_labels(...)`
- `_apply_partition_based_rewire(...)`

The implementation:

1. Read node features from `g.ndata["feature"]`.
2. Run a simple two-means style clustering:
   - choose two random feature vectors as initial centers,
   - assign nodes to nearest center,
   - update centers,
   - repeat a small fixed number of times.
3. Treat the two clusters as pseudo-labels.
4. Select edges according to severity.
5. Keep the edge source fixed.
6. Replace the destination with a node from the opposite pseudo-label group.
7. Use true labels only for auditing heterophily before/after, not for target selection.

### Is It Feasible?

Yes. It is simple and fast enough for this benchmark.

### Is It Logically Valid?

Mostly yes, but it is weaker than the oracle version.

It is logically valid as:

> a feature-proxy structural stress test.

It is not a strong attacker model. The two-means partition may or may not align with fraud/normal labels.

### What To Verify

Check:

- `pseudo_label_pos_rate`
- `heterophily_ratio_before`
- `heterophily_ratio_after`
- `n_rewired_edges_actual`

Important:

If `heterophily_ratio_after` barely changes, the non-oracle scenario may be too weak or the pseudo partitions may not correspond to label disagreement.

### Weaknesses

- Two-means clustering is very simple.
- It can produce partitions that do not match fraud semantics.
- It may stress the graph less than the oracle version.
- Results depend on feature geometry.

## Scenario 3: `camouflage_feature_oracle`

### Plain Meaning

This scenario makes selected fraud nodes look normal in feature space.

In fraud terms:

> Some fraud reviews/accounts copy the feature pattern of normal reviews/accounts.

This tests how much models rely on node features.

### How It Is Implemented

Code path:

- `apply_scenario(...)`
- `_feature_camouflage(...)`

The implementation:

1. Read true labels.
2. Find fraud nodes: `label == 1`.
3. Find normal nodes: `label == 0`.
4. Select `int(p_cam_feat * number_of_fraud_nodes)` fraud nodes.
5. For each selected fraud node, sample one normal node.
6. Replace or blend the fraud feature vector:

```text
new_fraud_feature = (1 - gamma) * old_fraud_feature + gamma * sampled_normal_feature
```

In the main v3 config, `gamma = 1.0`, so selected fraud features are fully replaced by normal-node features.

7. Keep the graph structure unchanged.
8. Record cosine similarity before and after.

### Is It Feasible?

Yes. This is one of the simplest and most feasible scenarios. It only edits node features.

### Is It Logically Valid?

Yes, as a controlled feature camouflage stress test.

It directly matches the idea from camouflage literature: fraudsters may modify attributes to look benign.

But it is oracle-based because the code uses true labels to choose fraud nodes and normal feature donors.

### What To Verify

Check:

- `n_changed_nodes_requested`
- `n_changed_nodes_actual`
- `mean_cosine_to_sampled_normal_before`
- `mean_cosine_to_sampled_normal_after`
- `mean_l2_delta`

Expected pattern:

- cosine similarity to sampled normal features should increase,
- with `gamma = 1.0`, `mean_cosine_to_sampled_normal_after` may be exactly or nearly `1.0`.

### Weaknesses

- Uses true labels.
- Full replacement with `gamma = 1.0` is strong and artificial.
- Randomly copying a normal feature vector may create unrealistic duplicate-like features.
- It does not model gradual or constrained feature changes.

Recommended future check:

- Add another config with `gamma = 0.25`, `0.5`, `0.75`, and `1.0`.

## Scenario 4: `camouflage_relation_oracle`

### Plain Meaning

This scenario changes fraud-node neighborhoods so fraud nodes connect more to normal nodes and optionally lose some suspicious fraud-to-fraud edges.

In fraud terms:

> Fraud nodes try to hide by creating benign-looking relationships.

This tests structural camouflage.

### How It Is Implemented

Code path:

- `apply_scenario(...)`
- `_relation_camouflage(...)`

The implementation:

1. Read true labels.
2. Find fraud nodes and normal nodes.
3. Select `int(p_cam_rel * number_of_fraud_nodes)` fraud nodes.
4. For each selected fraud node:
   - choose one configured relation,
   - sample a normal destination node,
   - add a new edge from fraud node to normal node,
   - repeat `camouflage_edges_per_node` times.
5. In the main v3 config, `camouflage_edges_per_node = 2`.
6. Optionally remove suspicious edges:
   - find edges from selected fraud nodes to other fraud nodes,
   - remove up to `remove_suspicious_ratio * added_edges_actual`.
7. In the main v3 config, `remove_suspicious_ratio = 0.5`.
8. Rebuild the graph.
9. Record neighborhood composition before and after.

### Is It Feasible?

Yes. It is feasible, but the effect may be weak on a dense graph like YelpChi.

YelpChi in the current run has about 8 million edges and average degree around 175. Adding only two edges per selected fraud node may barely change the overall neighborhood.

### Is It Logically Valid?

Yes, as a structural camouflage stress test.

It is more realistic than pure random noise because it specifically changes fraud-node neighborhoods toward normal nodes. But it is still oracle-based and static.

### What To Verify

Check:

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

Expected pattern:

- fraud-to-normal neighbor ratio should increase,
- fraud-to-fraud neighbor ratio should decrease if removals happen,
- selected fraud nodes should have slightly changed out-degree.

### Weaknesses

- Uses true labels.
- May be too weak because YelpChi is dense.
- Adds a fixed number of edges per selected node instead of degree-relative changes.
- Does not check whether the added relation is semantically plausible.
- Does not model temporal trust-building behavior.

Important interpretation warning:

If model performance barely changes under relation camouflage, that does not automatically prove robustness. It may mean the perturbation did not change the graph enough.

Recommended future check:

- Try stronger configs with `camouflage_edges_per_node = 5` or `10`.
- Try degree-relative camouflage, for example adding edges equal to 10 percent of the selected node's current degree.

## Scenario 5: `noise_edges_uniform`

### Plain Meaning

This scenario adds random edges to the graph.

In fraud terms:

> The neighborhood becomes more cluttered and less informative.

This tests whether message-passing models are harmed by random graph density/noise.

### How It Is Implemented

Code path:

- `apply_scenario(...)`
- `_add_random_edges(...)`

The implementation:

1. Select configured relations.
2. Count existing eligible edges.
3. Compute:

```text
requested_edge_pairs = int(edge_noise_rate * existing_edge_count)
```

4. For each requested edge pair:
   - choose a relation,
   - sample random source node,
   - sample random destination node,
   - reject self-loops, existing edges, or duplicates,
   - add accepted edge.
5. In the main v3 config, `undirected = false`, so only one directed edge is added per accepted pair.
6. Record requested and actual added edges.

### Is It Feasible?

Yes. It is straightforward and feasible.

### Is It Logically Valid?

Yes, as random density stress.

It is not targeted fraud behavior. It is a general noise/clutter test.

### What To Verify

Check:

- `n_added_edge_pairs_requested`
- `n_added_edge_pairs_actual`
- `n_added_edges_requested`
- `n_added_edges_actual`
- `n_edges_before`
- `n_edges_after`
- rejection counters in `extra_info_json`

Expected pattern:

- total edge count should increase,
- realized additions should be reasonably close to requested additions unless many candidates are rejected.

### Weaknesses

- Random edges are not realistic fraud behavior.
- Uniform node sampling may create edges that would not make sense in the real Yelp relation schema.
- It may be too weak or too strong depending on graph density.
- It does not specifically target model weaknesses.

## Why MLP Results Are Useful

The MLP model uses node features only. It ignores graph edges.

Therefore:

- If only edges change, MLP performance should stay the same under `train_clean_eval_all`.
- If features change, MLP performance can change.

This makes MLP a sanity check.

Expected behavior:

- MLP unchanged under heterophily rewiring.
- MLP unchanged under relation camouflage.
- MLP unchanged under random edge noise.
- MLP degraded under feature camouflage.

If MLP changes under graph-only stress in the clean-train evaluation protocol, something may be wrong with the benchmark or result grouping.

## How Reliable Are The Results?

The results are moderately reliable for controlled stress testing if the run is complete and the audit file confirms the perturbations worked.

They are not reliable for broad claims about real-world fraud robustness.

### Reliable For

The benchmark can support claims like:

- Under this controlled scenario, model A degraded more than model B.
- Feature camouflage harmed all feature-using models.
- Oracle heterophily was stronger than non-oracle heterophily in this run.
- A graph-only perturbation did not affect the MLP, as expected.
- The realized perturbation strength was X according to `variant_audit.csv`.

### Not Reliable For

The benchmark should not be used to claim:

- This model is production-ready.
- This simulates real fraudsters.
- Oracle perturbations are realistic attacks.
- A severity of `0.3` means the same thing across all scenarios.
- Model rankings are stable across all datasets and splits.
- Relation camouflage has no effect in general if this specific relation-camouflage setup caused little degradation.

## Important Assumptions

The benchmark assumes:

- DGL's YelpChi version is the intended dataset.
- Labels are correct and can be used for oracle stress tests.
- Train/validation/test masks are fixed and fair.
- The homogeneous graph view is acceptable for comparing all models.
- Validation threshold selection is enough to avoid F1 leakage.
- Graph seeds and training seeds capture enough variance for a final project.
- Static graph perturbations are a meaningful proxy for robustness.

Each assumption is reasonable for a course project, but each should be disclosed.

## Main Weaknesses To Mention In The Report

1. Most strong scenarios are oracle-based.
2. Only one data split is used in the main v3 config.
3. The evaluation graph is homogeneous, even though YelpChi is relation-rich.
4. Relation camouflage may be too weak on a dense graph.
5. Feature camouflage with `gamma = 1.0` is artificial.
6. Random noise edges are not realistic fraud behavior.
7. Results depend on the DGL YelpChi dataset version.
8. PMP and SEC-GFD use benchmark-side reduced-cost hyperparameters.
9. Current generated CSVs may not be enough if graph binaries are missing locally.
10. Passing unit tests does not prove that the full DGL benchmark can run in the current environment.

## Checklist Before Trusting A Benchmark Run

Before using results in the final paper/report, verify this checklist.

### Environment

- Use Python 3.10 or 3.11, not Python 3.13.
- Confirm DGL imports successfully.
- Confirm PyTorch and DGL versions match the README.

### Graph Generation

- Run the graph stage locally.
- Confirm graph binaries exist under `runs/<experiment_name>/graphs`.
- Confirm `graph_variants.csv` exists.
- Confirm `variant_audit.csv` exists.

### Scenario Audit

For heterophily:

- `heterophily_ratio_after > heterophily_ratio_before`
- `n_rewired_edges_actual > 0`

For feature camouflage:

- `n_changed_nodes_actual > 0`
- `mean_cosine_to_sampled_normal_after > mean_cosine_to_sampled_normal_before`

For relation camouflage:

- `n_camouflaged_edges_added > 0`
- `fraud_to_normal_neighbor_ratio_after > fraud_to_normal_neighbor_ratio_before`
- if the shift is tiny, do not make strong robustness claims

For noise:

- `n_added_edge_pairs_actual > 0`
- `n_edges_after > n_edges_before`

### Results

- Confirm `missing_or_error_runs.csv` is empty, or explain every missing/error row.
- Confirm both protocols were run if both are discussed.
- Confirm `protocol` is visible in plots and tables.
- Confirm `oracle_labels` is visible in plots and tables.
- Confirm MLP behaves as expected:
  - unchanged under graph-only shift,
  - degraded under feature camouflage.

### Reproducibility

- Save the exact config used.
- Do not edit frozen configs.
- If changing scenario strength, create a new config file.
- Prefer relative graph paths in future outputs or regenerate locally before submission.

## Practical Interpretation Guide

When looking at a result curve, ask these questions in order:

1. Which protocol is this?
   - `train_on_variant` or `train_clean_eval_all`

2. Is this scenario oracle?
   - If yes, describe it as controlled oracle stress.

3. Did the audit confirm that the graph or features really changed?
   - If no, do not interpret performance changes strongly.

4. Does MLP behave as expected?
   - If no, investigate before trusting other models.

5. Is the performance change larger than seed variance?
   - If no, phrase the conclusion cautiously.

6. Is this only one split?
   - If yes, avoid strong model-ranking claims.

## Final Judgment

The scenario implementations are feasible and mostly logically valid for a controlled benchmark. They are strong enough for a faculty project if explained honestly.

The most trustworthy scenarios are:

- feature camouflage, because it directly changes fraud-node features and has clear audit metrics,
- oracle heterophily, because it directly increases label-disagreement edges and has clear audit metrics.

The scenarios that need the most careful interpretation are:

- non-oracle heterophily, because feature clusters may not match fraud labels,
- relation camouflage, because the current perturbation may be weak relative to YelpChi density,
- random noise, because it is a generic clutter test rather than fraud behavior.

The benchmark should be trusted only after verifying the audit file, environment, run completeness, and protocol separation.
