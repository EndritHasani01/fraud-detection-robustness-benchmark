# CV Bullet Points: Fraud Detection Robustness Benchmark

Use these as candidate CV bullets. Each bullet is written as action + technical scope + measurable impact, so you can choose the strongest 4-6 for a resume entry.

## Strongest Resume Bullets

- Built a reproducible graph-fraud robustness benchmark in Python, DGL, PyTorch, pandas, and matplotlib for YelpChi, evaluating 45,954 nodes and 8,051,348 edges across 4 fraud-detection models.
- Architected a staged experiment runner for graph generation, model evaluation, clean-train shift testing, and plotting, producing 252 successful result rows with 0 error rows in the latest benchmark run.
- Integrated 4 model families, MLP, GraphSAGE, PMP/LA-SAGE-S, and SEC-GFD, into a unified evaluation matrix with shared metrics, split masks, cached graph variants, and a single `results.csv` schema.
- Implemented 5 deterministic graph stress scenarios, including oracle heterophily rewiring, non-oracle heterophily rewiring, feature camouflage, relation camouflage, and uniform edge noise, generating 31 audited graph variants.
- Developed 2 evaluation protocols, `train_on_variant` and `train_clean_eval_all`, to separate stressed retraining from test-time distribution shift and preserve protocol-level interpretability in every output artifact.
- Designed seed-decoupled reproducibility controls with 2 graph seeds and 3 training seeds, isolating perturbation variance from model-training variance across the benchmark matrix.
- Added validation-only threshold selection for ROC-AUC, average precision, and macro-F1 evaluation, preventing test leakage while reporting 3 complementary fraud-detection metrics.
- Created report-ready robustness outputs with 10 plot-stage CSV artifacts and 59 total plot/report files, including robustness scores, max-stress performance drops, audit curves, and missing-run diagnostics.
- Implemented `variant_audit.csv` and `performance_audit_join.csv` to connect requested perturbation severity with realized graph changes across 31 variants, improving traceability from graph edits to model outcomes.
- Built resumable training infrastructure using run keys over dataset, split, scenario, severity, graph seed, training seed, model, and protocol, enabling safe restart behavior for a 252-run experiment matrix.

## Benchmark Architecture

- Engineered a modular benchmark package with 22 Python modules and 7,169 lines of benchmark code, separating configuration, graph caching, model adapters, result schemas, preflight checks, scenario generation, and plotting.
- Defined 7 frozen JSON experiment configs for v1, v2, v3, fast smoke tests, and full runs, preserving backward compatibility while allowing benchmark methodology to evolve without mutating prior experiments.
- Implemented a CLI stage system with `graphs`, `baselines`, `pmp`, `secgfd`, `shift`, `matrix`, and `plots` stages, reducing benchmark execution to repeatable `py -m benchmark.run` commands on Windows PowerShell.
- Centralized result identity through an 8-field run key, eliminating duplicate or ambiguous rows across 4 models, 5 scenarios, 3 training seeds, 2 graph seeds, and 2 supported protocols.
- Added preflight counting and progress reporting for training stages, giving expected-run totals, skip counts, retry behavior, and ETA visibility before expensive model jobs begin.
- Implemented skip-existing, retry-errors, force, model-subset, clean-only, max-variant, and max-training-seed controls, supporting both fast developer smoke tests and full benchmark reruns from the same runner.
- Standardized output contracts across stages with `config.json`, `graph_variants.csv`, `variant_audit.csv`, `results.csv`, stage summaries, and plot-ready CSVs under `runs/<experiment_name>/`.

## Graph Stress Testing

- Implemented heterograph-aware perturbation generation for YelpChi relations `net_rsr`, `net_rtr`, and `net_rur`, then converted variants to a homogeneous DGL evaluation view for cross-model comparability.
- Built oracle heterophily rewiring with opposite-label target selection, bounded resampling, duplicate rejection, self-loop rejection, and existing-edge rejection, measuring realized heterophily shifts per variant.
- Built non-oracle heterophily rewiring using feature-derived pseudo partitions, enabling a label-free structural stress reference alongside oracle rewiring in the same benchmark surface.
- Built feature camouflage with fraud-node feature replacement toward sampled normal-node features, auditing changed-node counts, cosine-similarity shifts, and L2 deltas for severity validation.
- Built relation camouflage with 2 benign-looking edge additions per selected fraud node and 50% suspicious-edge removal, separating structural camouflage from feature-only camouflage in configs and outputs.
- Built uniform relation-aware edge-noise injection with explicit duplicate, self-loop, existing-edge, and attempt-budget policies, quantifying requested versus realized edge additions in audit outputs.
- Audited 31 graph variants across clean, heterophily, camouflage, and noise scenarios, making each perturbation traceable by dataset, split, scenario, severity, and graph seed.
- Preserved deterministic split masks with a fixed split seed of 717 and a 40% train, 20% validation, 40% test policy, keeping all scenario comparisons aligned to the same data partition.

## Model Integration

- Integrated MLP as a node-feature-only sanity baseline, establishing a non-message-passing control against graph-dependent models across all stress scenarios.
- Integrated GraphSAGE as a message-passing baseline in PyTorch/DGL, enabling direct comparison between feature-only and graph-neighborhood learning under structural perturbations.
- Integrated PMP/LA-SAGE-S from a vendored research repository through benchmark-side adapters, overriding sampled neighbors, batch size 512, 100 epochs, patience 10, and Windows-safe `num_workers=0`.
- Integrated SEC-GFD from a vendored research repository with reduced-cost benchmark hyperparameters, including hidden dimension 32, order 2, high-order flag 1, lambda 0.2, and learning rate 0.01.
- Normalized metrics, graph references, status handling, runtime measurement, and output rows across 4 heterogeneous model implementations, enabling direct model comparison in one CSV.
- Added train-graph references for shift-protocol rows, proving whether each result came from clean-graph training or per-variant training without relying on external logs.

## Evaluation And Reporting

- Evaluated each model with ROC-AUC, average precision, and macro-F1, covering ranking quality, fraud-class precision-recall behavior, and thresholded classification quality in the same benchmark.
- Implemented validation-based best-F1 thresholding, selecting decision thresholds without accessing the test split and recording the chosen threshold for each completed run.
- Generated robustness-score tables, max-stress performance-drop tables, summary curves, cross-split summary outputs, and missing-or-error diagnostics from `results.csv`, keeping reports reproducible from one source of truth.
- Added bootstrap 95% confidence interval support to the plots stage through `--ci`, improving report readiness for mean performance curves and error bars.
- Propagated oracle-status disclosure into graph ledgers, summary CSVs, and plot outputs, making controlled oracle perturbations visible in downstream analysis.
- Joined performance metrics with audit metrics in `performance_audit_join.csv`, enabling analysis against realized perturbation strength rather than requested severity alone.
- Produced 10 top-level plot CSV artifacts and 49 generated PNG figures in the latest run, giving both table-ready and figure-ready evidence for the final report.
- Wrote completeness diagnostics to `missing_or_error_runs.csv`; the latest v3 output contains only the header row, confirming 0 missing or failed expected runs for the completed protocol.

## Testing And Quality

- Built 11 regression-test modules covering config parsing, scenario generation, variant utilities, result schemas, operational behavior, PMP integration, SEC-GFD integration, shift stages, plotting, and scenario audits.
- Added tests for result-key behavior and resumability helpers, reducing risk of duplicated rows or accidental overwrites in long-running benchmark jobs.
- Added tests for scenario semantics and audit outputs, validating that perturbation implementations preserve expected graph-edit policies and report requested-versus-realized changes.
- Added tests for plot-stage behavior, protecting report artifacts such as summary curves, robustness scores, performance drops, and missing-run diagnostics from schema regressions.
- Preserved generated `runs/` outputs as ignored artifacts while keeping configs, benchmark code, tests, and documentation reviewable in source control.

## Documentation And Stakeholder Communication

- Documented the benchmark workflow in `README.md`, covering environment setup, frozen configs, stage commands, CLI flags, output schemas, reporting boundaries, and reproducibility rules.
- Authored a v3 benchmark reference that explains 5 scenario families, 2 protocols, audit artifacts, graph view modes, and interpretation limits for report-facing use.
- Created stakeholder-facing documentation that distinguishes controlled static robustness testing from realistic adversarial fraud simulation, reducing the risk of over-claiming benchmark results.
- Wrote methodology and report guidance for the final project, connecting experiment design, reproducibility constraints, metrics, plots, and disclosure requirements into a coherent deliverable.

