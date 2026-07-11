# Lessons Learned

## Purpose

This repository developed from a working experiment pipeline into a reproducible and auditable robustness benchmark for graph-based fraud detection. The lessons below come from the v1-to-v3 development history, the repository architecture and tests, and the two Kaggle executions recorded in `KAGGLE_DUAL_T4_RESEARCH_RUN_with_outputs.ipynb` and `KAGGLE_DUAL_T4_RESEARCH_RUN_with_outputs_latest.ipynb`.

The Kaggle run is especially instructive. It showed that scientific design, Python correctness, package installation, native-library compatibility, successful execution, and report completeness are separate quality layers. Each layer needs its own evidence.

## 1. Research design

### Keep the scientific claim narrow and explicit

The benchmark is a controlled static stress test. It is not a temporal fraud simulator, an adaptive attacker, or evidence that a model is production-ready. The defensible claim is comparative robustness under the same dataset, split, scenario, severity, protocol, and cached graph.

This boundary makes the project stronger. Oracle perturbations can be useful controlled probes, but they must remain visibly labelled and must not be described as realistic attacks.

### Treat evaluation protocol as an experimental variable

The two protocols answer different questions:

- `train_on_variant` measures adaptation when a model is trained and evaluated on the same stressed graph.
- `train_clean_eval_all` measures test-time shift after training only on the clean graph.

Protocol therefore belongs in the run key, result schema, summaries, figures, and written conclusions. Results from the two protocols should never be silently averaged together.

### Measure realized perturbations

The severity value has a different meaning for rewiring, feature camouflage, relation camouflage, and random-edge noise. Sampling rules can also make the realized change smaller than the requested change.

The v3 artifact separation is consequently essential:

- `graph_variants.csv` records which graph state exists.
- `variant_audit.csv` records what actually changed.
- `results.csv` records model performance.
- `plots/performance_audit_join.csv` connects performance to perturbation evidence.

A performance change should be interpreted only after the corresponding audit confirms that the stress changed the intended property in the intended direction.

### Separate sources of randomness

The project correctly distinguishes the split seed, graph seed, and training seed. This is more informative than one global seed because it separates variation caused by the perturbation realization from variation caused by model training.

Seeds improve repeatability but do not guarantee bitwise-identical CUDA results. Hardware, drivers, native libraries, and package versions are also part of the experimental record.

### Use baselines as controls

The MLP is not only a simple competitor. Because it does not consume edges, it is a structural control. Under clean-train evaluation, graph-only perturbations should not change its predictions when features are unchanged. Such metamorphic checks can reveal implementation or data-joining errors without requiring a known target score.

### State the fairness-versus-fidelity tradeoff

Generating some stresses on the source heterograph improves relation semantics. Converting every variant to one homogeneous evaluation view improves comparability across models. This is a deliberate tradeoff: relation-aware generation does not imply heterograph-native downstream evaluation.

## 2. Reproducibility and traceability

### Frozen configs are research records

The v1, v2, and v3 configs preserve historical experiment definitions. A changed experimental design should receive a new config instead of silently changing an old one.

A complete run should be traceable to the exact config and its hash, dataset and split, graph and training seeds, model and protocol, upstream repository commits, compatibility patches, dependency versions, hardware, and final artifacts.

### The scientific run key is the operational identity

The unified key is:

`(dataset_id, split_id, scenario_id, severity, graph_seed, training_seed, model_id, protocol)`

This key supports deduplication, resumability, retries, and completeness checks. Progress and final counts should be computed from canonical unique keys, not from raw CSV line counts.

Expected matrix sizes should be derived from the active config and asserted at runtime. Hand-maintained counts can drift when scenarios, severities, models, or seeds change.

### Self-contained execution still needs provenance

Embedding the benchmark implementation makes the notebook independent of the current repository at execution time. It does not remove the need to record what ran. Useful evidence includes embedded-module hashes, the config hash, pinned upstream commits, compatibility diffs, a dependency freeze, dataset hashes, GPU information, and output hashes.

Self-containment answers whether the notebook can run without an uploaded project. Provenance answers exactly which implementation produced the results.

### Review compatibility patches scientifically

An upstream patch should be classified as import-only, runtime-safety, numerical-behavior changing, or training-protocol changing. A small patch can still change the evaluated method. Benchmark-side feasibility changes, such as sampling and epoch limits, should be disclosed rather than presented as exact paper reproduction.

The SEC-GFD integration illustrates this directly. Its training-index correction and mapping of cosine similarity from `[-1, 1]` into a positive logarithm domain are necessary for split-correct, finite training, but they change numerical behavior and therefore must be visible in the notebook and report.

## 3. Runtime and dependency engineering

### Installation is not compatibility

The failed Kaggle run installed the requested Python, PyTorch, and DGL packages, yet DGL could not load `libcusparse.so.11`. This was a native dynamic-linker problem, not evidence that the Python package was absent.

A GPU runtime is valid only after it can import PyTorch and DGL, resolve required native libraries, create a DGL graph on CUDA, execute a small operation, and repeat that probe on both physical GPUs.

### A compatible wheel tag is not native-dependency closure

The second Kaggle execution showed a stricter version of the same lesson. PyTorch and DGL were both CUDA 11.8 wheels, and `pip check` passed, yet DGL still could not load because the PyTorch wheel did not ship `libcusparse.so.11`. Python package metadata, matching `+cu118` labels, and a successful import of PyTorch do not prove that an external extension's ELF dependencies are complete.

For GPU stacks, validation should inspect the actual files, run `ldd` on the external native library, and execute a representative operation. If a required runtime component is absent, install its pinned official package explicitly; a loader-path change cannot discover a binary that was never installed.

### Use one subprocess environment definition

CUDA visibility, DGL backend, Python paths, native-library paths, thread limits, and unbuffered logging should be produced by one shared environment builder. Dependency probes, tests, graph generation, smoke runs, and full training should all use it. A probe run under different environment variables does not validate the real execution path.

### Budget against observed free resources

The uploaded execution reported substantially less free disk than the nominal Kaggle disk size. Resource planning should therefore use current free space plus estimated growth for the runtime, dataset, graph cache, lane outputs, and final archive.

Practical controls include no package cache, shallow Git fetches, references for no-op graphs, compact report archives, and a safety reserve. A hard threshold should be justified by measured workload size and should include a clear recovery action.

### Make setup recoverable

Directory existence is not proof of a complete setup. An interrupted run can leave a partial virtual environment, incomplete Git checkout, header-only CSV, or stale output from another config.

Each setup stage should validate its content and write a success marker only after verification. Reruns should repair or recreate invalid state while safely reusing verified state.

A single run fingerprint is stronger than several unrelated version checks. Config, embedded modules, upstream commits and exact compatibility edits, dependency freeze, GPU runtime, driver, and tool version should be hashed together. Cached root, smoke, and lane rows should be reusable only when that fingerprint matches.

## 4. Error propagation and notebook control flow

### Preserve the first causal error

The native DGL loader error caused later adapter, test, graph, training, plotting, and packaging failures. These downstream errors were consequences, not independent scientific failures.

Errors should retain the original exception type and message, chained traceback, stage, command, environment summary, and full log path. Friendly wrapper messages are useful only when they do not hide the actionable cause.

### A failed cell is not a pipeline-wide state transition

The recorded Kaggle execution continued into later cells after earlier failures. Therefore, reaching a later cell is not proof that its prerequisites succeeded.

Every major stage should publish and verify explicit state, such as runtime validated, adapters validated, tests passed, graphs complete, smoke complete, each protocol complete, and reporting complete. Dependent cells should check these states before doing any work.

### `SKIPPED` is not the same as unexecuted

In the second run, every code cell received an execution count, but 32 cells printed `SKIPPED` after prerequisite checks failed. This was useful error containment, not silent Kaggle scheduling behavior. Reports should distinguish cell execution, phase execution, and scientific completion so a safe no-op is neither mistaken for success nor reported as a platform skip. The clean revision now uses `BLOCKED` for this state.

### Use informative operational checks

Checks should first verify that input tables are non-empty, have the required columns, and cover the expected keys. Only then should scientific invariants be evaluated. Errors should report expected and observed values rather than raising blank assertions.

### Reject success-shaped empty artifacts

The failed execution could still create header-only result summaries. File existence alone is therefore weak evidence. Summary and reporting code should require successful input rows, valid schemas, expected key coverage, and finite metrics.

Final success should come from a verified manifest with a `complete` status. Static markdown should not infer success merely because execution reached the final section.

## 5. Progress, resumability, and dual-GPU orchestration

### Show exact, key-aware progress

Long experiments are easier to trust when preflight output states the expected graph states and scientific run keys. Progress should distinguish successful, missing, and error keys. Raw file-line counts can be misleading when retries or duplicates exist.

### Validate before resuming

Before reusing output, verify the config hash, ledger and audit completeness, upstream fingerprints, result schema, and relevant environment fingerprint. This prevents rows from different experiments from being merged accidentally.

Temporary files followed by atomic rename are a strong pattern for ledgers, results, summaries, figures, manifests, and archives.

### Retry only recoverable failures

Automatic retry is appropriate for transient process or I/O failures. It is wasteful for deterministic dependency failures. Failure classes should indicate whether retry is sensible, and logs from every attempt should be retained.

Retry control flow itself needs testing. A loop labelled “two attempts” is not a retry if the first subprocess exception escapes the loop. The parent also needs a `finally` cleanup that terminates and waits for every active child, closes log handles, and closes progress bars on interruption or monitoring failure.

### Isolate GPU lanes

One independent process per T4 fits this workload better than forcing data parallelism. Each process should see one logical CUDA device and write to its own lane directory.

The in-process CSV lock in the benchmark cannot coordinate separate GPU processes. Lane results should therefore be merged by the parent process using the scientific run key, preferring successful retries over earlier error rows.

Both T4s still share CPU, RAM, and disk bandwidth. Heartbeats should report these shared resources, and concurrent durations should not be treated as fair model-speed comparisons.

## 6. Testing and validation

### Keep local verification and target validation distinct

Local verification can prove notebook structure, Python syntax, config validity, deterministic helper behavior, result-schema rules, stage dispatch, and other unit-level contracts. It cannot prove that Kaggle's CUDA driver, native libraries, DGL wheel, memory limits, or dual-GPU execution are compatible.

Target-environment validation must occur inside Kaggle using the same isolated interpreter, environment variables, upstream patches, graph representation, adapters, and devices as the final run. Neither type of evidence replaces the other.

### Validate from cheapest to most expensive

A useful validation ladder is:

1. Notebook schema and source compilation.
2. Config validation and matrix-size derivation.
3. Native-library and CUDA probes on both GPUs.
4. Upstream adapter imports.
5. Fast semantic tests.
6. Clean graph creation.
7. A real short smoke run for all four models.
8. Full graph-variant generation.
9. Both full protocols.
10. Completeness and reporting audits.

This order finds environment and integration defects before expensive graph generation or training.

The integration smoke should include both the clean graph and the largest expected graph, not only the easiest case. This gives an early, inexpensive check of adapter correctness and likely graph-memory pressure.

### Test scientific invariants as well as code paths

Important invariants include unchanged labels and split masks, deterministic same-seed perturbations, explicit severity-zero no-ops, matching requested and realized counts, MLP stability under graph-only shift, validation-only threshold selection, failure on missing audit evidence, and successful resume without duplicate keys.

Metric helpers also need edge-case tests. Average Precision must group tied scores so it is independent of row order, and a row may receive `status=ok` only when every reported metric and duration is finite and within its valid domain.

## 7. Reporting and interpretation

### Preserve the evidence chain

Every report value should trace through:

`config -> graph ledger -> perturbation audit -> result key -> summary -> figure`

Figures are views of the evidence, not replacements for the underlying tables.

### Completeness comes before plotting

Before producing report figures, verify the expected unique run keys, successful statuses, required ledger and audit rows, finite metrics, per-model and per-protocol counts, and an empty missing/error report.

Figures should be created only after their input data pass these gates and should be published atomically. This prevents an empty or partial figure from looking like a valid result.

### Derive conclusions from fresh results

The notebook should not hard-code a winning model. Findings should be calculated from the completed result table and should always retain protocol, scenario, oracle status, metric, uncertainty, and realized perturbation evidence.

Uncertainty from the small crossed seed design is descriptive. Stronger inferential claims would require repeated splits or datasets and a statistical method that respects the dependence structure.

## 8. Architecture and project process

### Keep the staged pipeline

Separating graph generation, model evaluation, and reporting ensures that every model uses the same cached variants, interrupted training can resume, and figures can be regenerated without retraining.

Thin adapters around PMP and SEC-GFD keep shared benchmark logic independent of upstream research code while enforcing common splits, metrics, protocols, and result schemas.

### Maintain one current contract

README files, working agreements, configs, tests, notebook explanations, and report text should agree on the current benchmark version, scenario catalog, expected outputs, protocol meanings, and environment. Historical documents should be visibly labelled as historical.

Authoritative documentation should be version-controlled and distributed with the repository. A document that is ignored, stale, or available only in one local workspace cannot reliably guide collaborators or reviewers.

### Treat completion as an evidence claim

Local verification should be reported as local verification. Final Kaggle completion requires a clean top-to-bottom target run with validated runtime probes, successful smoke models, complete protocol matrices, no unresolved error keys, valid reports and figures, and a completed manifest.

## Final takeaway

The repository has a strong scientific core: frozen experiment definitions, deterministic graph generation, separated randomness sources, protocol-aware evaluation, semantic perturbation tests, a unified result schema, and direct perturbation audits.

The Kaggle incident clarified the remaining engineering standard: validate the native runtime before using it, model notebook execution as a dependency-aware pipeline, preserve the first causal error, reject empty success-shaped artifacts, and declare completion only from verified evidence.

A trustworthy ML benchmark must show not only model scores, but also what ran, what changed, what succeeded, what failed, and which conclusions the evidence can support.
