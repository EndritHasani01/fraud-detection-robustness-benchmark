# Stress-Test Trustworthiness Review

This review captures the trustworthiness analysis that motivated the v3 documentation and scenario refinements. It analyzes the earlier v2 benchmark state, not the final implemented v3 contract.

For the current v3 scenario definitions, audit artifacts, disclosure rules, and interpretation boundaries, use [docs/V3_BENCHMARK_REFERENCE.md](V3_BENCHMARK_REFERENCE.md).

This note answers a practical question about the current v2 benchmark:

> Are the implemented `camouflage`, `heterophily`, and `noise` scenarios trustworthy enough to support real conclusions, and are they implemented in the right way?

Short answer:

- Yes, they are useful as controlled benchmark stress tests.
- No, they are not all equally strong as methodological claims.
- `heterophily` is the most trustworthy of the three.
- `noise` is acceptable as a coarse density / neighborhood-flooding stress.
- `camouflage` is only partially implemented, because the current benchmark covers feature camouflage but not relation camouflage.
- None of them should be presented as realistic attacker simulations.

## Why this question matters

A benchmark perturbation can be "correct" in two different senses:

1. **Implementation correctness**
   - The code actually performs the intended graph or feature modification.
2. **Methodological trustworthiness**
   - The modification is a meaningful proxy for the robustness problem we claim to study.

The current benchmark is stronger on implementation correctness than on full methodological coverage.

## What the current code actually does

All three scenario families are implemented in [benchmark/scenarios.py](</d:/1.Revolucion/1. Fakulltet/4.1 Semestri/6. Theoretical Graphs/GitHub/fraud-detection-robustness-benchmark/benchmark/scenarios.py>):

- `_rewire_edge_dst_to_opposite_label`
- `_feature_camouflage`
- `_add_random_edges`

The v2 config wires the following scenarios ([configs/exp_yelpchi_v2.json](</d:/1.Revolucion/1. Fakulltet/4.1 Semestri/6. Theoretical Graphs/GitHub/fraud-detection-robustness-benchmark/configs/exp_yelpchi_v2.json>)):

- `heterophily_rewire_oracle`
- `camouflage_feature_oracle`
- `noise_edges`

Important setup detail:

- the benchmark converts YelpChi to a **homogeneous** graph before applying perturbations
- relation types are merged into one graph view

So the scenarios act on a single merged graph, not on the original heterogeneous relation structure.

## What the project guide expected

The design in [FINAL_PROJECT_GUIDE.md](</d:/1.Revolucion/1. Fakulltet/4.1 Semestri/6. Theoretical Graphs/GitHub/fraud-detection-robustness-benchmark/FINAL_PROJECT_GUIDE.md>) is broadly aligned with the current implementation:

- heterophily: rewire edges toward opposite-label endpoints
- camouflage:
  - feature camouflage
  - relation camouflage
- noise: add random edges / increase neighborhood density

So the current code is not conceptually off-track. The bigger question is whether it covers enough of each robustness concept to justify strong conclusions.

## Heterophily: trustworthy as a controlled oracle stress

### What the code does

`_rewire_edge_dst_to_opposite_label`:

- samples a fraction `p_rewire` of existing edges
- keeps each source node fixed
- replaces the destination with a node of the opposite label
- preserves node count and edge count
- leaves labels and split masks unchanged

This is very close to the guide's suggested pseudocode.

### What I verified from the generated graphs

In the current `v2` run:

- clean heterophily ratio is about `0.2269`
- stressed heterophily ratio rises to about `0.4588`
- edge count stays the same

That is exactly the directional behavior this scenario is supposed to produce.

### Why this is useful

This perturbation isolates a very specific failure mode:

- message passing becomes less reliable because neighbors are less label-consistent

That makes it a good benchmark stress for comparing models that claim robustness to heterophily.

### Limitations

There are still important caveats:

- It is explicitly **oracle** because it uses true labels to construct opposite-label edges.
- It is not a realistic fraudster simulation; it is a synthetic stress test.
- It operates on the homogeneous graph, so it ignores relation-type semantics.
- It is only "degree-preserving-ish":
  - source-side degree is fixed
  - destination-side degree distribution can shift
- The logged `n_rewired_edges` counts edges selected for rewiring, not necessarily edges whose destination actually changed.

That last point is a minor audit issue, not a fatal flaw.

### Trust assessment

**Verdict:** trustworthy for benchmarking, provided it is described as an oracle structural stress test.

### Safe claim

You can safely say:

> We used a controlled oracle heterophily stress that rewires a fraction of edges toward opposite-label destinations, increasing overall edge heterophily while keeping graph size fixed.

You should not say:

> This simulates realistic adversarial fraud behavior.

## Camouflage: useful, but only partially implemented

### What the code does

`_feature_camouflage`:

- selects a fraction `p_cam_feat` of fraud nodes
- samples normal nodes
- replaces or blends fraud features with normal features according to `gamma`

In the current v2 config:

- `gamma = 1.0`
- so selected fraud nodes fully copy sampled normal-node features

### What I verified from the generated graphs

For the current `0.3` feature-camouflage variant:

- exactly `2003` nodes changed
- all `2003` changed nodes are fraud nodes
- zero normal nodes were changed
- `2003` is exactly `30%` of the fraud nodes

So the implementation is mechanically doing what the config says.

### Why this is useful

It is a meaningful test of:

- feature-based disguising of fraud nodes
- robustness to fraud nodes becoming less separable in feature space

This is especially useful because all current models consume node features.

### The biggest limitation

The guide explicitly separates camouflage into:

1. feature camouflage
2. relation camouflage

But the benchmark currently implements only feature camouflage.

That means the current scenario name can easily be over-read. It does **not** test full camouflage robustness in the CARE-GNN sense. It tests only one side of camouflage.

### Additional caveats

- It is also oracle in the sense that it uses true labels to identify fraud nodes.
- It changes only node features, not local neighborhood composition.
- It runs on the homogeneous graph, so there is no relation-aware camouflage pattern.
- The proxy logs are informative:
  - number of changed fraud nodes
  - cosine similarity before and after
  - feature delta magnitude
  but those values are currently stored only in per-graph metadata, not in the main CSVs.

### Trust assessment

**Verdict:** trustworthy as a **feature camouflage** stress, not trustworthy as a complete camouflage benchmark.

### Safe claim

You can safely say:

> We evaluated feature camouflage by replacing a fraction of fraud-node features with sampled normal-node features.

You should not say:

> We fully benchmarked model robustness to camouflage.

## Noise: acceptable as a coarse neighborhood-flooding stress

### What the code does

`_add_random_edges`:

- samples random source-destination pairs uniformly
- appends them to the graph
- can optionally mirror them if `undirected=true`

This matches the guide's idea of random edge noise and degree inflation.

### What I verified from the generated graphs

For the current `noise_edges` severity `0.2` variants:

- target added edges: `1,610,269`
- edge count rises from about `8.05M` to `9.66M`
- almost all sampled additions are genuinely new edges
- there are small artifacts:
  - about `5.9k` sampled edges already existed
  - about `650` duplicates appear within the newly sampled set
  - about `34-37` self-loops appear

These artifacts are tiny relative to the total number of added edges.

### Why this is useful

This is a reasonable proxy for:

- denser neighborhoods
- irrelevant edge flooding
- reduced local signal-to-noise ratio

That makes it useful for testing whether models are robust to neighborhood expansion and random clutter.

### Limitations

- It is not adversarially targeted noise.
- It may create self-loops and duplicate edges.
- It does not preserve degree distributions in any realistic way.
- It does not model fraud-specific camouflage or targeted misleading neighbors.
- On the homogeneous graph, it ignores relation-specific noise patterns.

So this scenario is best understood as **random density stress**, not as a realistic attacker.

### Trust assessment

**Verdict:** trustworthy as a coarse random-noise / density stress.

### Safe claim

You can safely say:

> We added uniformly random edges to increase neighborhood density and evaluate robustness to graph clutter.

You should not say:

> We simulated realistic adversarial neighborhood manipulation.

## Cross-cutting concerns

## 1. Homogeneous conversion weakens semantic realism

The benchmark intentionally converts the original DGL fraud graph to a homogeneous view before perturbation.

This is practical for cross-model comparability, but it has a cost:

- relation semantics are erased
- per-relation perturbations are no longer possible
- relation camouflage cannot really be represented faithfully

This does not make the benchmark invalid. It just means the scenarios are graph-level stress tests, not relation-aware fraud-behavior simulations.

## 2. Oracle perturbations are acceptable, but must be disclosed

Both heterophily rewiring and feature camouflage use true labels.

That is acceptable in this project because the guide already frames these as controlled oracle benchmark perturbations. But it must be stated clearly in any report section that uses them.

## 3. Severity values are not calibrated across families

The severity knobs mean different things:

- `p_rewire = 0.3`
- `p_cam_feat = 0.3`
- `edge_noise_rate = 0.2`

These are not directly comparable quantities. A "0.3" camouflage stress and a "0.3" heterophily stress do not represent equal perturbation strength.

So the benchmark supports:

- within-family trend analysis
- cross-model comparison under the same family

It does **not** support naive cross-family severity comparisons.

## 4. Auditability is weaker than it should be

The perturbation functions compute useful proxy logs, for example:

- `heterophily_ratio_before` / `after`
- `n_camouflaged_nodes`
- cosine-similarity shift
- `n_added_edges`

These are saved in per-variant metadata via `scenario_info`, but not exported into the main `graph_variants.csv` or `results.csv` schemas.

That means:

- the information exists
- but it is harder to analyze and plot systematically

This is a benchmark-quality issue, not a conceptual issue.

## 5. There is very little direct semantic test coverage

I did not find dedicated tests that directly validate the semantics of:

- edge rewiring correctness
- camouflage correctness
- noise-edge properties

Most existing tests focus on pipeline behavior, configs, summaries, and stage execution.

That means the perturbations are not currently untrusted, but they are under-tested relative to how central they are to the benchmark's claims.

## What insight is genuinely supported by the current benchmark

The current scenarios do support meaningful insights of the following kind:

- Is a model more sensitive to feature corruption or to structural corruption?
- Does a graph-agnostic baseline behave as expected under graph-only perturbations?
- Which models degrade most when neighbor labels become less aligned?
- Which models stay stable when the graph gets denser and noisier?

These are legitimate robustness questions.

The current benchmark is especially good for:

- **comparative robustness under controlled stress**
- **sanity checks on model behavior**
- **surfacing failure modes that deserve deeper follow-up**

It is weaker for:

- realistic attacker modeling
- relation-aware fraud behavior
- full camouflage claims
- strong causal claims about real-world fraud dynamics

## Practical conclusion by scenario

### Heterophily

- Implemented in the right general way: **yes**
- Methodologically useful: **yes**
- Trust level: **high for oracle benchmarking**

### Camouflage

- Implemented in the right general way: **yes, for feature camouflage**
- Methodologically complete: **no**
- Trust level: **medium**

### Noise

- Implemented in the right general way: **yes**
- Methodologically realistic: **no**
- Trust level: **medium to high as a coarse density stress**

## Recommended wording for the report

If you want wording that is careful but still confident, this is safe:

> The v2 perturbations should be interpreted as controlled stress tests rather than realistic attack simulations. The heterophily scenario is the strongest methodological component because it directly increases disagreeing-label edges while preserving graph size. The camouflage scenario currently captures feature camouflage only, not the full relation-aware camouflage discussed in CARE-GNN. The noise scenario is best interpreted as random neighborhood flooding or density stress. Together, these scenarios are useful for comparative robustness analysis, but they should not be over-claimed as faithful models of real adversarial fraud behavior.

## Recommended improvements

If this benchmark will be used for a final report or defended as a robustness study, these are the most valuable upgrades:

1. Add **relation camouflage** as a fourth scenario or as a second camouflage branch.
2. Export `scenario_info` fields into a tabular artifact so perturbation strength is easy to audit.
3. Add direct unit tests for each perturbation function.
4. Optionally filter out:
   - duplicate sampled noise edges
   - self-loops
   - rewiring cases that accidentally preserve the original destination
5. Consider a heterograph-aware perturbation mode if relation semantics become important to the write-up.

## Bottom line

The current perturbations are good enough to learn something real, but only if the conclusions are scoped correctly.

- `heterophily` is genuinely informative and well aligned with the stated robustness target.
- `noise` is a reasonable coarse stress for density and neighborhood clutter.
- `camouflage` is only partially covered and should be described more narrowly as feature camouflage.

So the benchmark is already useful, but its strongest claim is:

> controlled comparative stress testing

not:

> realistic adversarial fraud simulation
