# What, Why, And How

This file is a direct explanation template for the main parts of the repository. It is written so you can answer three questions repeatedly: what is this, why did we use it, and how did we use it in the implementation. This is often the clearest way to explain a technical project because it connects concepts to design decisions and then to actual code.

## The Project

What this is: this project is a reproducible robustness benchmark for graph-based fraud detection. It is not just one model and one score. It is a small experiment system that loads a real graph fraud dataset, creates controlled stressed versions of the graph, evaluates multiple fraud detection models, and exports results and plots for comparison.

Why we used it: graph fraud detection models can look good on one clean dataset but fail when graph structure becomes misleading, noisy, or camouflaged. A robustness benchmark lets us study those failures in a controlled way. Instead of only asking which model has the best clean ROC-AUC, the project asks which model degrades under heterophily, camouflage, and noise, and whether the degradation is visible across seeds and protocols.

How we used it: the implementation is organized as a staged runner in `benchmark/run.py`. The `graphs` stage creates and caches clean and stressed graph variants. The model stages train and evaluate `mlp`, `sage`, `pmp`, and `secgfd`. The `plots` stage reads the unified result table and creates summary CSVs and figures. The stages are connected through CSV artifacts such as `graph_variants.csv`, `variant_audit.csv`, and `results.csv`.

## YelpChi

What this is: YelpChi is the fraud detection dataset used by the current benchmark configs. In this repository it is loaded through `dgl.data.FraudDataset` with the DGL source name `yelp`. It provides a graph, node features, and binary labels for fraud detection.

Why we used it: YelpChi is a common benchmark dataset in graph fraud detection papers, and it is supported by DGL. That makes it practical for integrating graph neural networks and specialized research methods. It also gives the project a real fraud graph instead of a toy graph, which makes robustness results more meaningful.

How we used it: the loader is in `benchmark/data.py`, specifically `load_dgl_fraud_dataset()`. The graph stage calls `_make_source_graph()` in `benchmark/run.py`, which loads YelpChi, casts features to `float32`, casts labels to `int64`, and overwrites train, validation, and test masks with deterministic benchmark-controlled masks. The dataset is cached under the run output directory so repeated runs do not need to reload everything from scratch.

## DGL

What this is: DGL is the graph deep learning library used to represent graphs, load the fraud dataset, convert graph views, save graph binaries, and run graph neural network layers.

Why we used it: the benchmark needs a common graph representation that works for graph manipulation and model training. DGL is already aligned with YelpChi through `FraudDataset`, and the integrated models either use DGL directly or can be adapted to DGL. Using DGL also lets the benchmark save graph variants as binary graph files and load exactly the same graph later.

How we used it: `benchmark/data.py` uses DGL to load the dataset and convert heterographs to homogeneous graphs through `dgl.to_homogeneous`. `benchmark/cache.py` uses DGL graph save/load utilities. `benchmark/baselines.py` uses DGL's `SAGEConv` for GraphSAGE. The PMP and SEC-GFD adapters also build on DGL graphs and dataloaders.

## Graph Representation

What this is: graph representation means the form of the graph used by the benchmark. YelpChi can be relation-aware as a heterograph, while the benchmark's canonical evaluation view is usually homogeneous. A homogeneous graph merges relations into one shared graph structure.

Why we used it: the repository compares several models, and not all of them have the same relation-handling assumptions. A single canonical evaluation view makes cross-model comparison simpler. At the same time, v3 keeps relation-aware generation for scenarios where relation semantics matter, such as relation camouflage and relation-filtered edge noise.

How we used it: configs define `graph_representation.canonical_view` as `homogeneous`. The graph stage loads the source graph, optionally applies some scenarios to the source heterograph, and then converts the result back to the homogeneous canonical view. The implementation path is `_make_source_graph()`, `_to_canonical_graph()`, and `_graphs_only()` in `benchmark/run.py`.

## Frozen Configs

What this is: frozen configs are JSON files under `configs/` that define experiments. They specify datasets, splits, models, stress scenarios, severity grids, graph seeds, training seeds, metrics, and disclosure notes.

Why we used them: an experiment should be reproducible and reviewable. If important choices are hidden in code or command-line memory, it becomes hard to know what was actually run. Frozen configs make the experiment definition explicit. They also prevent accidental mutation of historical experiments because old configs can be kept and new configs can be added when the design changes.

How we used them: `benchmark/config.py` loads and validates config files. The runner copies the active config into `runs/<experiment_name>/config.json` during the graph stage. The v3 main config is `configs/exp_yelpchi_v3.json`, while `exp_yelpchi_v3_fast.json` is used for quick smoke testing and `exp_yelpchi_v3_full.json` is used for denser runs.

## Train, Validation, And Test Masks

What this is: masks are boolean arrays on graph nodes that decide which nodes are used for training, validation, and testing. The training mask tells the model which labels it can learn from. The validation mask is used for early stopping and threshold selection. The test mask is reserved for final evaluation.

Why we used them: without fixed masks, a performance difference could come from different data splits rather than from the stress scenario. The benchmark wants clean and stressed variants to share the same task definition. This is especially important in fraud detection because the positive class is rare and split imbalance can distort results.

How we used them: `ensure_masks()` in `benchmark/data.py` overwrites DGL's masks with deterministic stratified masks based on `split_seed`, `train_size`, and `val_size`. The perturbation code checks that these masks do not change after graph stress is applied. Training stages then read `train_mask`, `val_mask`, and `test_mask` directly from the cached graph.

## Graph Caching

What this is: graph caching means saving the clean graph and all stressed graph variants to disk before model training. The graphs are stored as DGL `graph.bin` files with companion metadata.

Why we used it: every model must be evaluated on the exact same graph variants. If each model regenerated perturbations independently, random differences could make comparisons unfair. Caching also makes the workflow resumable because graph generation does not need to be repeated after a model-stage interruption.

How we used it: `benchmark/cache.py` saves graph binaries and `meta.json` files. `benchmark/paths.py` defines stable paths for clean graphs and variant graphs. The graph stage writes a row to `graph_variants.csv` for every graph state, and model stages load graph files from that CSV.

## Stress Scenarios

What this is: stress scenarios are deterministic transformations applied to the clean graph or its features. The current v3 benchmark includes oracle heterophily rewiring, non-oracle heterophily rewiring, feature camouflage, relation camouflage, and uniform random edge noise.

Why we used them: each scenario targets a different reason graph fraud detection can fail. Heterophily makes neighbors less label-consistent. Feature camouflage makes fraud nodes look normal in feature space. Relation camouflage makes fraud neighborhoods look more benign. Edge noise makes neighborhoods denser and less informative.

How we used them: scenarios are declared in the config and applied by `apply_scenario()` in `benchmark/scenarios.py`. The graph stage creates a `ScenarioSpec` for each `(scenario, severity, graph_seed)` combination. The scenario returns a perturbed graph, an applied/no-op flag, and an info dictionary. That info is exported into `variant_audit.csv`.

## Oracle And Non-Oracle Perturbations

What this is: an oracle perturbation uses true labels during construction. A non-oracle perturbation avoids true labels when creating the stressed graph, though labels may still be used later for auditing.

Why we used it: oracle scenarios are useful because they create strong, clear stress tests. For example, oracle heterophily can directly increase opposite-label edges. But they are not realistic attacker models, so the benchmark also includes non-oracle references such as feature-based pseudo-partition rewiring and random edge noise.

How we used it: configs use `oracle_mode` and `oracle_labels` to mark scenarios. `scenario_oracle_labels()` in `benchmark/config.py` normalizes that metadata. The graph ledger, audit file, summary CSVs, and plots preserve oracle status so it remains visible in reporting.

## Graph Seeds And Training Seeds

What this is: graph seeds control perturbation randomness, while training seeds control model training randomness. They are separate because graph generation and model training are different sources of variation.

Why we used it: if results vary, we want to know whether they vary because the stressed graph realization changed or because the model training process changed. Decoupling the seeds makes that distinction possible. It also improves reproducibility because the same graph seed always maps to the same scenario realization.

How we used it: `get_graph_seeds()` and `get_training_seeds()` in `benchmark/config.py` read seed lists from the config. `benchmark/scenarios.py` derives a deterministic scenario RNG from graph seed plus scenario metadata. Model stages call `set_seeds()` from `benchmark/variants.py` before training.

## Evaluation Protocols

What this is: an evaluation protocol defines how training and evaluation graphs are paired. `train_on_variant` trains a model separately on each graph variant and evaluates on that same variant. `train_clean_eval_all` trains once on the clean graph and evaluates the trained model on every variant.

Why we used it: the two protocols answer different robustness questions. `train_on_variant` asks whether a model can adapt when stressed data is available during training. `train_clean_eval_all` asks how a model trained under normal conditions behaves under test-time graph shift.

How we used it: standard stages such as `baselines`, `pmp`, and `secgfd` write rows with protocol `train_on_variant`. The `shift` stage writes rows with protocol `train_clean_eval_all`. The `matrix` stage dispatches to the shift stage when called with `--protocol train_clean_eval_all`. `results.csv` includes `protocol` and `train_graph_ref` so the protocol is visible in every run row.

## MLP Baseline

What this is: the MLP is a multilayer perceptron that uses node features only and ignores graph edges.

Why we used it: the MLP is a control model. It tells us how much signal exists in node features without graph structure. It also helps validate the benchmark: graph-only perturbations should not change MLP predictions under clean-train shift, while feature camouflage can change them.

How we used it: the model is defined in `benchmark/baselines.py` and trained in `benchmark/baselines_stage.py`. It uses class-weighted cross entropy, validation ROC-AUC for early stopping, and validation-selected thresholding for F1.

## GraphSAGE Baseline

What this is: GraphSAGE is a standard message passing graph neural network. It updates node representations by aggregating neighbor information.

Why we used it: GraphSAGE is a simple graph-based baseline. It gives the project a reference for how a normal message passing model behaves under heterophily, camouflage, and noise. This is useful when comparing specialized methods against a familiar GNN.

How we used it: `benchmark/baselines.py` defines the GraphSAGE model using DGL's `SAGEConv`. The same baseline training stage handles both MLP and GraphSAGE, with model-specific forward logic.

## PMP

What this is: PMP is a specialized graph fraud detection model from the vendored `Repos/PMP-master` repository. It is designed around partitioned message passing for fraud graphs with label imbalance and mixed homophily/heterophily.

Why we used it: PMP is relevant to this benchmark because the stress scenarios directly test graph fraud weaknesses that PMP is intended to address. It gives the project a stronger specialized method beyond simple baselines.

How we used it: `benchmark/pmp_stage.py` imports LA-SAGE-S from the PMP repo, loads the original PMP YAML config, merges benchmark-side hparams, creates PMP-specific `label_unk`, builds DGL dataloaders, trains with validation monitoring, and writes results into the same `results.csv` schema as every other model.

## SEC-GFD

What this is: SEC-GFD is a specialized graph fraud detection model from `Repos/SEC-GFD-main`. It is designed around heterophily and spectral filtering.

Why we used it: SEC-GFD is relevant because the benchmark includes strong heterophily stress. A heterophily-aware model is a natural comparison point against MLP, GraphSAGE, and PMP.

How we used it: `benchmark/secgfd_stage.py` imports the research model by file path, resolves hparams from the config and CLI overrides, trains the model with benchmark-side validation monitoring and loss handling, and writes unified result rows. The adapter also caches spectral coefficient calculations to avoid repeated recomputation.

## Metrics

What this is: the benchmark reports ROC-AUC, average precision, and macro-F1. ROC-AUC and average precision are ranking metrics. Macro-F1 is a thresholded classification metric that balances the normal and fraud classes.

Why we used them: fraud detection is imbalanced, so accuracy alone would be misleading. ROC-AUC shows ranking quality, average precision is useful when positives are rare, and macro-F1 checks hard classification behavior without letting the majority class dominate.

How we used them: `benchmark/metrics.py` implements these metrics without requiring scikit-learn. Training functions choose an F1 threshold on validation data using `best_f1_macro_threshold()`, then apply that threshold to test scores. This avoids test leakage.

## Unified Results CSV

What this is: `results.csv` is the single row-level performance table for all models, scenarios, seeds, and protocols.

Why we used it: separate result formats for different models would make reporting fragile. A unified schema makes plots, summaries, resumability, and completeness checks much simpler.

How we used it: `benchmark/results.py` defines the result columns and helper functions. All model stages call `write_result_row()`. The row key includes dataset, split, scenario, severity, graph seed, training seed, model ID, and protocol.

## Variant Audit

What this is: `variant_audit.csv` is the table that records what each perturbation actually changed. It includes requested and realized changes plus scenario-specific before/after metrics.

Why we used it: severity values are not enough. A severity of `0.3` means different things across stress families, and requested changes may not be fully realized if edge sampling rejects candidates. The audit file lets the report justify claims using realized evidence.

How we used it: scenario implementations return an `info` dictionary. The graph stage passes that dictionary into `build_variant_audit_row()` in `benchmark/scenario_audit.py`, which normalizes the information into a wide CSV row.

## Plots And Report Artifacts

What this is: the plots stage creates summary CSVs and PNG figures from `results.csv`, `graph_variants.csv`, and `variant_audit.csv`.

Why we used it: raw result rows are too detailed for a report. The report needs curves, clean-to-stress drops, robustness scores, audit curves, and missing-run diagnostics.

How we used it: `benchmark/plots_stage.py` computes `summary_curves.csv`, `performance_drop_max_stress.csv`, `robustness_scores.csv`, `audit_curves.csv`, cross-split summaries, `performance_audit_join.csv`, and `missing_or_error_runs.csv`. With `--ci`, it also computes bootstrap confidence intervals.

## Tests

What this is: the `tests/` folder contains regression tests for configs, scenarios, result schemas, stage behavior, shift protocol, plot outputs, and model adapters.

Why we used it: a benchmark can produce misleading conclusions if scenario semantics, run keys, protocol handling, or result schemas break. Tests protect those contracts.

How we used it: the tests use a mixture of real config files, mocked stages, and fake graph objects. This keeps tests fast while still checking important behavior such as deterministic perturbations, preserved masks and labels, scenario audit export, resumability, and matrix dispatch.

