# Fraud Detection Robustness Benchmark - OOP, Design Patterns, and Architecture

## Scope

This document explains the software architecture, object-oriented design, design patterns, and engineering structure used in the `fraud-detection-robustness-benchmark` repository. It is based on the repository source code, configuration files, tests, and local project documentation.

The project is a reproducible benchmark harness for graph-based fraud detection robustness. It builds graph perturbation variants, evaluates several fraud detection models, records metrics in a unified result format, audits whether perturbations were actually realized, and generates plot-ready summaries.

## System Purpose

The benchmark evaluates how graph fraud detection models behave under controlled graph and feature shifts. The current frozen experiment configuration targets YelpChi from `dgl.data.FraudDataset` and supports several model families:

- `mlp`: a node-feature-only baseline.
- `sage`: a GraphSAGE baseline using graph structure and node features.
- `pmp`: LA-SAGE-S integrated from `Repos/PMP-master`.
- `secgfd`: SEC-GFD integrated from `Repos/SEC-GFD-main`.

The system is designed around reproducibility:

- Split masks are deterministic per `split_id`.
- Graph perturbations are deterministic per `graph_seed`.
- Training is deterministic per `training_seed` where the underlying libraries allow it.
- Thresholds are selected on validation data only.
- `results.csv` acts as the main source of truth for model performance.
- `variant_audit.csv` records requested versus realized perturbation strength.

## High-Level Architecture

The repository is organized as an offline benchmark pipeline rather than a web service or long-running application. The core workflow is:

```text
experiment config
  -> dataset loading and deterministic split creation
  -> graph cache generation
  -> scenario perturbation generation
  -> variant ledger and audit files
  -> model training and evaluation stages
  -> unified results.csv
  -> summaries, audit joins, and plots
```

The pipeline is split into explicit CLI stages:

- `graphs`: builds the base graph, graph variants, variant ledger, and audit file.
- `baselines`: trains and evaluates `mlp` and `sage`.
- `pmp`: trains and evaluates the PMP integration.
- `secgfd`: trains and evaluates the SEC-GFD integration.
- `shift`: trains selected models once on the clean graph and evaluates all variants.
- `matrix`: launches the integrated model matrix and dispatches protocols.
- `plots`: reads results and audit files, checks completeness, writes summaries and figures.

This staged design keeps expensive operations separate. Graph generation can be run once, model evaluations can resume from existing results, and plotting can be repeated without retraining.

## Repository Layout

Important source areas include:

- `benchmark/run.py`: CLI entry point and stage dispatcher.
- `benchmark/config.py`: experiment config validation and output path setup.
- `benchmark/data.py`: DGL FraudDataset loading, split creation, dtype normalization, and graph conversion.
- `benchmark/cache.py`: graph save/load helpers and graph reference files.
- `benchmark/paths.py`: deterministic graph cache path generation.
- `benchmark/scenarios.py`: perturbation scenario implementations.
- `benchmark/relation_utils.py`: heterograph and relation-aware edge utilities.
- `benchmark/scenario_audit.py`: variant audit schema and audit row generation.
- `benchmark/results.py`: result schema, run-key logic, CSV migration, and thread-safe CSV appends.
- `benchmark/variants.py`: variant ledger parsing, graph loading, seed setting, and class-weight helpers.
- `benchmark/baselines.py`: PyTorch baseline model classes and baseline factory.
- `benchmark/baselines_stage.py`: baseline training, evaluation, artifacts, and stage loop.
- `benchmark/pmp_stage.py`: adapter around the vendored PMP repository.
- `benchmark/secgfd_stage.py`: adapter around the vendored SEC-GFD repository.
- `benchmark/shift_stage.py`: clean-train/evaluate-all protocol.
- `benchmark/matrix_stage.py`: orchestrator for multi-model matrix execution.
- `benchmark/metrics.py`: dependency-light binary classification metrics and threshold search.
- `benchmark/plots_stage.py`: result aggregation, audit joins, completeness checks, and plotting.
- `benchmark/summarize.py`: grouped result summary generation.
- `benchmark/preflight.py`: expected-run construction and progress estimation.
- `tests/`: tests for config validation, operational behavior, result handling, scenarios, audits, plots, model stages, and shift protocol.

## Configuration-Driven Architecture

The benchmark is driven by experiment JSON files, especially `configs/exp_yelpchi_v3.json`. This config defines the dataset, split, graph representation, models, perturbation scenarios, seeds, metrics, threshold policy, and disclosure notes.

The config is treated as an executable contract. `benchmark/config.py` validates:

- Required top-level keys.
- Dataset and split schema.
- Model list and model-specific settings.
- Graph representation settings.
- Scenario IDs, methods, severities, oracle labels, graph view mode, and parameters.
- Training seed and graph seed lists.
- SEC-GFD hyperparameter schema.
- Evaluation metric and threshold-selection schema.

This is a configuration-as-contract pattern. Most benchmark behavior is selected by config data, while Python modules provide the validated operations. That separation makes the benchmark easier to reproduce because experiment structure is visible without reading the training code.

The V3 config captures several important engineering decisions:

- Canonical model evaluation uses a homogeneous graph view.
- Some perturbations can be generated on a source heterograph and converted back with `dgl.to_homogeneous`.
- Oracle and non-oracle scenarios are explicitly labeled.
- Feature camouflage, relation camouflage, heterophily rewiring, and random edge noise are separate scenario families.
- Graph seeds and training seeds are separated, allowing perturbation variance and training variance to be analyzed independently.

## Core Runtime Pipeline

### CLI Dispatch

`benchmark/run.py` is the main process entry point. It parses CLI arguments and dispatches to the selected stage. The CLI supports stage selection, model filtering, clean-only execution, retry behavior, force behavior, protocol selection, and SEC-GFD overrides.

The dispatcher acts as a facade over the benchmark subsystems. Users run one command, but the command delegates to smaller modules for config validation, graph caching, model training, result writing, or plotting.

### Graph Stage

The `graphs` stage performs the most important setup work:

1. Load the source DGL FraudDataset.
2. Normalize feature and label dtypes.
3. Create deterministic train, validation, and test masks.
4. Convert the graph to the canonical evaluation view.
5. Save the clean base graph.
6. Build configured scenario variants.
7. Save graph binaries or graph reference files.
8. Write `graph_variants.csv`.
9. Write `variant_audit.csv`.
10. Ensure `results.csv` has the expected header.

The stage writes variant and audit files through temporary files and then replaces the final files. This atomic-write style reduces the chance of leaving partially written ledgers if graph generation is interrupted.

No-op variants are represented by graph references instead of duplicated graph binaries. This is a storage-conscious cache design: severity `0.0` can point back to the base graph while still appearing as a separate experimental variant in the ledger.

### Model Stages

Model stages share the same conceptual structure:

```text
read variant ledger
  -> filter variants and models
  -> skip completed run keys unless forced
  -> load graph
  -> train model artifact
  -> evaluate model artifact
  -> write one result row
  -> write error row on failure
```

The stages avoid mixing training logic with result schema logic. Training functions return artifacts, evaluation functions consume artifacts, and stage loops handle resumability and CSV writes.

### Matrix Stage

`benchmark/matrix_stage.py` is an orchestration layer. It validates that graph variants exist, computes a preflight estimate, and then launches baseline, PMP, and SEC-GFD stages. When the requested protocol is `train_clean_eval_all`, it delegates to the shift stage.

The matrix stage is intentionally thin. It does not implement model-specific behavior itself; it coordinates existing stage functions.

### Shift Stage

`benchmark/shift_stage.py` implements the `train_clean_eval_all` protocol:

1. Group variants by dataset and split.
2. Find the clean variant.
3. Train each selected model once on the clean graph per training seed.
4. Evaluate the trained artifact on every graph variant.
5. Write result rows with `train_graph_ref` pointing to the clean graph.

This protocol separates training distribution from evaluation distribution, which is useful for robustness analysis. It answers a different question from `train_on_variant`: instead of asking how well a model performs when trained and tested on each perturbed graph, it asks how well a clean-trained model tolerates perturbations at evaluation time.

### Plot and Reporting Stage

`benchmark/plots_stage.py` reads `results.csv`, `graph_variants.csv`, and `variant_audit.csv`. It validates required audit columns, joins performance and audit evidence, checks for missing or errored runs, and writes CSV summaries plus PNG figures.

Important outputs include:

- `performance_audit_join.csv`
- `summary_curves*.csv`
- `performance_drop_max_stress*.csv`
- `robustness_scores*.csv`
- `audit_curves*.csv`
- `missing_or_error_runs.csv`
- plot PNG files

This stage acts like a read-model over the benchmark outputs. It does not mutate trained models or graph caches; it projects recorded evidence into summaries and figures.

## Object-Oriented Design

The repository uses a pragmatic hybrid style:

- Object-oriented design is used for value objects, model classes, artifacts, policies, and progress tracking.
- Functional procedural code is used for pipeline stages and data transformations.

This is appropriate for a scientific benchmark. The core domain objects need clear schemas, but the workflow itself is a sequence of reproducible transformations.

## Value Objects and Dataclasses

The project uses dataclasses to define typed records that move between modules.

### `ExperimentPaths`

Defined in `benchmark/config.py`, `ExperimentPaths` is a frozen dataclass containing output paths:

- `out_dir`
- `config_copy_path`
- `results_csv_path`
- `variants_csv_path`
- `variant_audit_csv_path`
- `graphs_dir`

It centralizes filesystem path construction after config validation. Making it frozen communicates that output locations should not be mutated casually after initialization.

### `LoadedDataset`

Defined in `benchmark/data.py`, `LoadedDataset` groups:

- `dataset_id`
- `source_name`
- `graph`

This gives the dataset loader a stable return type rather than returning a loose tuple.

### `GraphPath`

Defined in `benchmark/paths.py`, `GraphPath` is a frozen dataclass that groups the path for one graph cache entry:

- `dir_path`
- `graph_bin_path`
- `meta_json_path`
- `ref_json_path`

This object prevents repeated manual construction of graph cache paths across modules.

### `ScenarioSpec`

Defined in `benchmark/scenarios.py`, `ScenarioSpec` captures the scenario identity and behavior:

- `scenario_id`
- `family`
- `method`
- `oracle_labels`
- `graph_view_mode`
- `params`

It is created from config dictionaries and passed into scenario application code. This is a clear domain object: it represents "what perturbation should be applied" independently of the graph being perturbed.

### `EdgeSamplingPolicy`

Defined in `benchmark/relation_utils.py`, `EdgeSamplingPolicy` describes edge sampling constraints:

- Reject self-loops.
- Reject existing edges.
- Reject duplicate sampled edges.
- Bound sampling attempts.

It is a policy object. Scenario code can use the policy without hard-coding every rejection rule inline.

### `VariantRow`

Defined in `benchmark/variants.py`, `VariantRow` is a parsed representation of a row from `graph_variants.csv`. It converts string CSV values into typed fields such as severity, graph seed, and boolean flags.

This avoids spreading CSV parsing details across model stages.

### Model Artifact Dataclasses

Each model family has an artifact dataclass:

- `BaselineModelArtifact` in `benchmark/baselines_stage.py`
- `PMPModelArtifact` in `benchmark/pmp_stage.py`
- `SECGFDModelArtifact` in `benchmark/secgfd_stage.py`

These objects hold the trained state and evaluation context needed after training:

- Model ID or repository root.
- Hyperparameters.
- State dict.
- Decision threshold.
- Device.
- Feature and label keys where relevant.

The artifact pattern separates training from evaluation. Training returns an artifact; evaluation reconstructs a model from the artifact and computes metrics. This keeps stage loops model-agnostic at a higher level.

### Metric and Plot Dataclasses

The reporting layer uses small dataclasses for structured aggregation:

- `ThresholdSearchResult` in `benchmark/metrics.py`
- `SummaryRow` in `benchmark/summarize.py`
- `ResultRow`, `MetricStats`, `AuditMetricSpec`, and `CompletenessReport` in `benchmark/plots_stage.py`
- `TrainingPreflightSummary` in `benchmark/preflight.py`

These objects make the reporting logic more explicit than passing nested dictionaries everywhere.

## PyTorch Model Classes

The baseline models are implemented as subclasses of `torch.nn.Module` in `benchmark/baselines.py`.

### `MLP`

`MLP` is a feed-forward neural network using linear layers, ReLU activations, and dropout. It is designed as a node-feature-only baseline and does not consume graph edges.

This class follows the standard PyTorch object model:

- Network layers are initialized in `__init__`.
- Forward computation is defined in `forward`.
- Parameters are managed by PyTorch through module registration.

### `GraphSAGE`

`GraphSAGE` uses DGL `SAGEConv` layers with a mean aggregator by default. It consumes both graph structure and node features.

The class encapsulates graph neural network architecture details while keeping training code separate. Stage code does not need to know how a SAGE layer is implemented; it only needs a module with a forward method.

### Baseline Factory

`build_baseline` acts as a simple factory. Given a model ID and hyperparameters, it constructs either an `MLP` or `GraphSAGE` instance.

This reduces conditional construction logic inside training code and makes supported baseline IDs explicit.

## Error and Exception Design

The project defines focused exception types:

- `ConfigError` in `benchmark/config.py`
- `CacheError` in `benchmark/cache.py`

These exceptions improve failure clarity. A config schema issue is different from a graph cache issue, and the exception classes preserve that distinction.

Model stages also use an error-row pattern. When a graph load, training run, or evaluation run fails, the system writes a row to `results.csv` with:

- `status=error`
- a truncated error message
- run identity columns

This is important for benchmark execution because one failed run should not necessarily erase all progress. Failures become part of the experiment evidence and can be retried later.

## Important Design Patterns

### Pipeline Architecture

The entire benchmark is a staged pipeline. Each stage has explicit inputs and outputs:

- Config and source dataset enter the graph stage.
- Graph cache and variant ledger enter model stages.
- Result and audit CSVs enter the plot stage.

This architecture makes long-running experiments resumable and inspectable.

### Facade Pattern

The CLI in `benchmark/run.py` provides a facade over many lower-level modules. Users interact with stage names and CLI flags, while the command hides details such as DGL loading, graph cache paths, model adapter setup, result-key generation, and plotting.

### Strategy Pattern

The project uses strategy-like dispatch in several places:

- Scenario `method` selects a perturbation implementation.
- Model ID selects MLP, GraphSAGE, PMP, or SEC-GFD behavior.
- Protocol selects `train_on_variant` or `train_clean_eval_all`.
- Graph view mode selects canonical homogeneous perturbation or source heterograph perturbation.

These strategies are mostly implemented with explicit conditionals rather than class hierarchies. That keeps the implementation direct while preserving the design idea: behavior is selected by a validated config value.

### Factory Pattern

The clearest factory is `build_baseline`, which constructs baseline model objects from a model ID and hyperparameters.

Other factory-like logic appears in:

- Scenario spec construction from config dictionaries.
- Graph cache path construction through `base_graph_path` and `variant_graph_path`.
- SEC-GFD hyperparameter resolution from config and CLI overrides.

### Adapter Pattern

The PMP and SEC-GFD integrations are adapters around external research repositories.

`benchmark/pmp_stage.py` adapts `Repos/PMP-master` by:

- Loading the PMP YAML config.
- Merging benchmark-side hyperparameters.
- Inserting the repository root into `sys.path`.
- Importing `model.LASAGE_S`.
- Creating PMP-specific graph fields such as `label_unk`.
- Restoring graph data after training and evaluation.

`benchmark/secgfd_stage.py` adapts `Repos/SEC-GFD-main` by:

- Loading `model/SECGFD.py` with `importlib.util.spec_from_file_location`.
- Giving the loaded module a unique name to avoid `model` package collisions.
- Installing a cache around `calculate_theta2`.
- Replacing or fixing loss behavior through benchmark-side functions.
- Resolving config and CLI hyperparameters into one artifact.

These adapters isolate external repository assumptions from the rest of the benchmark. The rest of the pipeline can treat PMP and SEC-GFD as train/evaluate stages even though their native code structures differ.

### Artifact Pattern

The trained model artifact dataclasses preserve everything needed for evaluation after training. This avoids relying on global variables or mutable stage state.

The pattern is visible in all model families:

```text
train_*_model(...)
  -> artifact

eval_*_model(artifact, graph, ...)
  -> metrics
```

This is especially useful for the shift protocol, where a clean-trained artifact is reused across multiple perturbed variants.

### Repository and Cache Pattern

`benchmark/cache.py` and `benchmark/paths.py` implement a lightweight graph repository:

- Graph binaries are stored in deterministic paths.
- Metadata JSON can live next to graph binaries.
- Reference JSON files allow one variant to point to another graph.
- Loading code resolves graph references before reading binaries.

The graph cache is a domain-specific repository for expensive graph objects.

### Ledger Pattern

`graph_variants.csv`, `variant_audit.csv`, and `results.csv` are appendable or generated ledgers:

- `graph_variants.csv` records which graph variant exists and where it is stored.
- `variant_audit.csv` records what was requested and what was actually realized.
- `results.csv` records every model run by run key.

This pattern fits benchmark work well. The system needs durable, inspectable records more than it needs an online database.

### Run-Key Idempotency Pattern

`benchmark/results.py` defines a run key:

```text
(dataset_id, split_id, scenario_id, severity, graph_seed, training_seed, model_id, protocol)
```

Stages load completed keys and skip work that already exists unless forced or explicitly retried. This gives the benchmark idempotent behavior across repeated runs.

### Validation and Guard Clause Pattern

Config validation, graph view checks, scenario schema checks, seed checks, model filtering, and audit column validation all use guard clauses. Invalid states fail early with targeted messages rather than failing later inside DGL, PyTorch, or plotting code.

### Deterministic Randomness Pattern

Scenario generation derives random state from a stable SHA-256 hash of:

- Scenario ID.
- Scenario method.
- Graph seed.
- Severity.
- Sorted parameters.

This avoids accidental dependence on Python process state or dictionary ordering. The same config and graph seed should produce the same perturbation behavior.

### Atomic Output Pattern

The graph stage writes variant and audit ledgers to temporary files before replacing final files. This reduces corruption risk when graph generation is interrupted.

### Lazy Import Pattern

Heavy dependencies such as DGL are imported lazily in places like dataset loading and graph cache operations. This allows lightweight commands, such as CLI help or some tests, to run without importing the full ML stack immediately.

### Progress Tracking Pattern

`ProgressTracker` in `benchmark/preflight.py` tracks progress, recent durations, and estimated time remaining. This is a small stateful object used for operational feedback during long experiment matrices.

## Scenario Subsystem

`benchmark/scenarios.py` is the core perturbation module. It applies graph or feature shifts while preserving core graph invariants.

Key invariants include:

- Node count is preserved.
- Label tensor is preserved.
- Train, validation, and test masks are preserved.
- Feature tensor shape is preserved.

The major scenario families are:

### Heterophily Rewiring

`heterophily_rewire_oracle` rewires selected edge destinations to opposite true-label nodes. It is an oracle scenario because it uses real labels to choose opposite-label targets.

`heterophily_rewire_nonoracle` uses feature-derived pseudo labels instead of true labels to choose pseudo-opposite targets. The implementation includes a two-means style partitioning helper and then audits outcomes against real labels.

These scenarios test how model behavior changes when graph neighborhoods become less label-homophilic.

### Feature Camouflage

`camouflage_feature_oracle` changes selected fraud-node features by replacing or interpolating them toward normal-node features. It records audit metrics such as feature-distance and cosine-similarity changes.

This tests whether models are robust when fraud nodes become more similar to normal nodes in feature space.

### Relation Camouflage

`camouflage_relation_oracle` adds benign-looking relation edges from fraud nodes to normal nodes and can remove suspicious fraud-to-fraud edges. It records neighbor-ratio and edge-count audit metrics.

Because YelpChi has relation types, this scenario can be generated on a heterograph source view and then converted back to the canonical homogeneous view.

### Uniform Edge Noise

`noise_edges_uniform` adds random edges under relation-aware constraints. It can reject self-loops, existing edges, and duplicates through `EdgeSamplingPolicy`.

This provides a less targeted structural perturbation baseline.

## Relation-Aware Architecture

`benchmark/relation_utils.py` isolates the complexity of DGL homogeneous and heterograph edge handling.

The module handles:

- Relation-key detection.
- Relation filtering by canonical tuple, full relation name, or short edge type.
- Edge extraction by relation.
- Heterograph rebuilding.
- Node data copying.
- Edge data copying when compatible.
- Relation count computation.
- Flattened edge views.
- Total edge counts.

This separation is important. Scenario implementations can focus on "what perturbation should happen" while relation utilities handle "how to perform that perturbation in DGL graph structures."

The design also protects the rest of the benchmark from DGL representation details. Models evaluate the canonical graph view, while scenario generation can still operate on richer source heterographs where appropriate.

## Results and CSV Architecture

`benchmark/results.py` defines the durable result schema.

Important features include:

- Protocol constants for `train_on_variant` and `train_clean_eval_all`.
- `RESULTS_COLUMNS` and `VARIANTS_COLUMNS`.
- Run-key construction and row-key parsing.
- Result CSV header creation and migration.
- Completed-key loading for resumability.
- Truncated error-message storage.
- Thread-safe CSV appends using `threading.RLock`.

The result writer separates successful rows from error rows:

- Successful rows include metrics such as ROC-AUC, average precision, and macro-F1.
- Error rows include status and error text, allowing failed runs to be visible in downstream completeness reports.

The schema includes `protocol` and `train_graph_ref`, which are essential for distinguishing train-on-variant results from clean-train/evaluate-all results.

## Metrics Architecture

`benchmark/metrics.py` implements core binary classification metrics without requiring scikit-learn:

- `roc_auc_binary`
- `average_precision_binary`
- `f1_macro_from_predictions`
- `best_f1_macro_threshold`
- `f1_macro_at_threshold`

The threshold search returns a `ThresholdSearchResult` dataclass. The best threshold is selected on validation scores only, and the selected threshold is then used for test-set macro-F1.

This design supports the benchmark's no-test-leakage rule. Test labels are used for final evaluation, not for threshold selection.

## Training Architecture

### Baseline Training

`benchmark/baselines_stage.py` handles MLP and GraphSAGE training. It:

- Sets seeds.
- Resolves the target device.
- Builds tensors and masks.
- Builds model instances through `build_baseline`.
- Applies class-weighted cross entropy when train labels are imbalanced.
- Uses Adam optimization.
- Monitors validation ROC-AUC.
- Applies early stopping by patience.
- Stores the best state dict on CPU.
- Selects the final threshold by validation macro-F1.

Evaluation rebuilds the model from the artifact and computes test metrics.

### PMP Training

`benchmark/pmp_stage.py` adapts the PMP repository into the benchmark protocol. It handles PMP-specific concerns:

- YAML config loading.
- Benchmark hyperparameter overrides.
- Windows-safe `num_workers=0`.
- DGL dataloader construction.
- Importing `LASAGE_S`.
- Creating `label_unk` for PMP training.
- Row-normalizing features.
- Restoring graph data in `finally` blocks after training or evaluation.

The restore behavior is an important reliability detail. PMP mutates graph fields, so the adapter protects later stages from accidental graph-state contamination.

### SEC-GFD Training

`benchmark/secgfd_stage.py` adapts SEC-GFD into the benchmark protocol. It handles:

- Config and CLI hyperparameter resolution.
- Dynamic import with a unique module name.
- Theta cache installation for repeated operations.
- NCE loss correction through benchmark-side code.
- Class imbalance handling.
- Early stopping on validation ROC-AUC or fallback loss.
- Validation-based threshold selection.

The adapter isolates SEC-GFD repository assumptions while producing a standard `SECGFDModelArtifact`.

## Audit Architecture

`benchmark/scenario_audit.py` converts scenario-specific perturbation information into a common audit schema.

The audit layer records:

- Requested severity.
- Realized perturbation strength.
- Edge counts before and after perturbation.
- Rewired edge counts.
- Added edge counts.
- Removed edge counts.
- Feature camouflage metrics.
- Relation camouflage metrics.
- Extra scenario-specific info as JSON.

This is an important engineering feature because requested perturbation severity is not always equal to realized perturbation severity. Sampling constraints, graph topology, relation filters, and available node classes can limit what is possible.

The audit file makes that gap visible.

## Plotting and Analysis Architecture

`benchmark/plots_stage.py` is a reporting subsystem rather than just plotting code. It:

- Reads result rows into structured objects.
- Reads audit rows.
- Validates audit schema.
- Checks duplicate variant and audit keys.
- Joins results, variants, and audits.
- Calculates metric means and standard deviations.
- Optionally bootstraps confidence intervals.
- Computes robustness summaries across severity curves.
- Writes completeness reports for missing or errored runs.
- Produces plot-ready CSVs and PNG figures.

The plotting stage uses the ledger outputs as immutable inputs. This means plots can be regenerated after code changes without rerunning model training.

## Testing Architecture

The `tests/` directory covers the most important nontrivial behavior:

- Config validation and frozen config files.
- Operational CLI behavior, model filtering, and preflight logic.
- Result CSV schema and row handling.
- Scenario audit generation.
- Scenario perturbation behavior.
- PMP adapter behavior.
- SEC-GFD adapter behavior and CLI overrides.
- Shift protocol and matrix dispatch.
- Plot-stage completeness and aggregation behavior.
- Variant parsing and loading behavior.

This test suite is focused on reproducibility, schema stability, orchestration, and adapter correctness. Those are the highest-risk areas in a benchmark harness.

## Engineering Strengths

### Reproducible Experiment Design

The benchmark explicitly controls split seeds, graph seeds, and training seeds. It also stores enough metadata to reconstruct which graph and protocol produced each result.

### Clear Separation of Concerns

The code separates:

- Config validation.
- Dataset loading.
- Graph caching.
- Scenario generation.
- Model training.
- Result writing.
- Audit writing.
- Plot generation.

This makes the project easier to debug because each subsystem has a defined responsibility.

### Resumable Execution

Run keys and completed-key loading allow long experiments to resume without repeating successful runs. Error rows preserve failure information instead of silently dropping failed jobs.

### Honest Perturbation Auditing

The benchmark does not assume that requested severity equals realized severity. It records audit metrics so downstream analysis can compare performance against actual perturbation effects.

### Practical External Model Integration

The PMP and SEC-GFD adapters show strong engineering ownership. They integrate research repositories into a unified benchmark interface while handling import conflicts, graph mutations, config translation, and hyperparameter overrides.

### Validation-First Workflow

The config and plot stages validate schema expectations early. This is important in CSV-based benchmark systems where small schema drift can otherwise produce misleading summaries.

## Tradeoffs and Risks

### CSV-Based State Is Simple but Fragile

CSV files are easy to inspect and version, but schema changes require careful migration. `ensure_csv_header` helps with this by migrating and reordering existing rows, but CSV workflows still need discipline around column names and key uniqueness.

### Config Uses Dictionaries Rather Than a Fully Typed Model

`benchmark/config.py` validates dictionaries directly. This keeps dependencies light, but a typed config model could provide stronger editor support and clearer nested schemas.

### External Repository Imports Are Brittle by Nature

PMP and SEC-GFD are integrated through repository paths, `sys.path`, and dynamic imports. The adapters reduce risk, but changes inside those external repositories could still break integration.

### Thread Lock Is Process-Local

`results.py` uses a `threading.RLock` for CSV writes. This protects writes inside one Python process. It does not provide a cross-process file lock if multiple separate processes append to the same `results.csv`.

### Shift-Stage Import Risk

In the inspected code, `benchmark/shift_stage.py` calls `truncate_error_message(e)` in error handling. If the function is not imported from `benchmark.results` in that file, the error path can raise a `NameError` instead of writing a clean error row. This is a code-level risk worth fixing before relying heavily on shift-protocol error handling.

### Static Stress Tests Are Not a Realistic Adversarial Simulator

The config disclosures correctly state that oracle stress tests must be disclosed and that static robustness evaluation is not the same as a live adversarial simulation.

## Design Patterns by File

| File | Main Design Role | Patterns Used |
| --- | --- | --- |
| `benchmark/run.py` | CLI and stage dispatcher | Facade, pipeline orchestration |
| `benchmark/config.py` | Config validation and paths | Configuration-as-contract, guard clauses, immutable value object |
| `benchmark/data.py` | Dataset loading and graph conversion | Adapter over DGL dataset, data normalization |
| `benchmark/cache.py` | Graph persistence | Repository/cache pattern, lazy imports |
| `benchmark/paths.py` | Deterministic cache paths | Immutable value object, path factory |
| `benchmark/scenarios.py` | Perturbation logic | Strategy dispatch, deterministic randomness, invariant checks |
| `benchmark/relation_utils.py` | Relation-aware graph operations | Policy object, utility module, representation adapter |
| `benchmark/scenario_audit.py` | Audit row generation | Ledger pattern, normalization adapter |
| `benchmark/results.py` | Results schema and appends | Ledger pattern, run-key idempotency, error-row pattern |
| `benchmark/variants.py` | Variant parsing and runtime helpers | Typed row object, parser helpers |
| `benchmark/baselines.py` | Baseline neural networks | PyTorch OOP, factory pattern |
| `benchmark/baselines_stage.py` | Baseline train/eval stage | Artifact pattern, template-like stage loop |
| `benchmark/pmp_stage.py` | PMP integration | Adapter pattern, artifact pattern, cleanup guards |
| `benchmark/secgfd_stage.py` | SEC-GFD integration | Adapter pattern, dynamic import, monkey-patch wrapper |
| `benchmark/shift_stage.py` | Clean-train/evaluate-all protocol | Artifact reuse, protocol strategy |
| `benchmark/matrix_stage.py` | Multi-model orchestration | Facade/orchestrator |
| `benchmark/metrics.py` | Metric calculations | Pure functions, value object result |
| `benchmark/plots_stage.py` | Reporting and plotting | Read-model projection, aggregation, completeness validation |
| `benchmark/preflight.py` | Expected run and progress analysis | Progress tracker object, planning model |

## OOP Interview Talking Points

This repository can be explained as a configuration-driven benchmark pipeline with selective OOP:

- Dataclasses model stable domain records such as paths, scenarios, variants, artifacts, and reports.
- PyTorch modules implement neural model classes using standard inheritance from `torch.nn.Module`.
- External research models are wrapped with adapter modules so they fit the benchmark's train/evaluate artifact interface.
- Scenario methods act like strategies selected by config.
- Result and audit CSVs form durable ledgers that make experiments resumable and inspectable.
- Graph cache helpers act like a lightweight repository for expensive DGL graph objects.
- The shift protocol reuses trained artifacts, showing why separating training artifacts from evaluation logic matters.

The strongest software engineering story is not only that the project trains models. It is that the project turns research code into a reproducible system with validation, caching, resumability, auditing, error reporting, and downstream analysis.

## Summary

`fraud-detection-robustness-benchmark` is a well-structured offline benchmark system. Its architecture is built around deterministic experiment configuration, staged execution, graph caching, scenario strategies, model adapters, result ledgers, and audit-driven reporting.

The project uses OOP where it adds clarity: dataclasses for domain records, PyTorch classes for models, artifact objects for trained state, and policy/progress objects for constrained behavior. It avoids unnecessary class hierarchies in the pipeline itself, using direct functions for reproducible transformations.

For a software engineering internship CV or interview, this repository demonstrates backend-style engineering skills in a research context: CLI design, data pipelines, configuration validation, caching, adapter integration, testing, reproducibility, error handling, and structured reporting.
