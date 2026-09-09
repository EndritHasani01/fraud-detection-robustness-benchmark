# Graph Fraud Detection Robustness Benchmark

Course final project: a controlled study of how graph-based fraud detectors respond to changing features and graph structure on YelpChi.

## Final submission

1. Read the **[academic report (PDF)](final-report/Graph_Fraud_Detection_Robustness_Academic_Report.pdf)**.
2. View the **[presentation](final-report/Fraud_Detection_Robustness_Presentation.pptx)**.
3. Inspect the [executed notebook with outputs](notebooks/KAGGLE_DUAL_T4_RESEARCH_RUN_with_outputs_latest_v4_r1_multi_seeds.ipynb) and [final evidence bundle](gfd-robustness-v4-factorial-report-r1/README.md).

The retained v4 results contain **5,040 successful evaluation rows**: 2,520 per training protocol, covering MLP, GraphSAGE, PMP, and SEC-GFD across six split/allocation identifiers. The evidence bundle retains the original results, executed configuration, audits, plots, and provenance.

## Research scope

The benchmark separates retraining on each perturbed graph (`train_on_variant`) from training on the clean graph and evaluating graph shift (`train_clean_eval_all`). It evaluates feature camouflage, relation camouflage, oracle and non-oracle heterophily rewiring, and noisy neighborhoods. Severity must be interpreted within each scenario family.

Oracle perturbations use label information and are diagnostic experiments, not evidence of operational attack robustness. PMP and SEC-GFD are integrated adapters under shared benchmark conditions; these results do not claim full reproduction of their papers. The PDF discusses findings, uncertainty, and limitations.

## Run or inspect the implementation

The [clean Kaggle notebook](KAGGLE_DUAL_T4_RESEARCH_RUN.ipynb) contains the benchmark implementation, experiment configuration, setup, and runtime checks. It writes its own source files and downloads pinned upstream dependencies, so the separate local development folders are not required for this notebook workflow. Follow the [reproduction guide](docs/REPRODUCIBILITY.md).

| Location | Purpose |
|---|---|
| [final-report/](final-report/README.md) | Final PDF and presentation |
| [KAGGLE_DUAL_T4_RESEARCH_RUN.ipynb](KAGGLE_DUAL_T4_RESEARCH_RUN.ipynb) | Runnable implementation and embedded experiment definition |
| [notebooks/](notebooks/README.md) | Historical execution with outputs |
| [gfd-robustness-v4-factorial-report-r1/](gfd-robustness-v4-factorial-report-r1/README.md) | Results and supporting evidence |
| [third_party/](third_party/README.md) | Vendored PMP and SEC-GFD research code and upstream licenses |

Personal explanations, duplicate report sources, historical configurations, and development tooling are kept locally and excluded from Git. The submission retains the full notebook implementation and the executed configuration in the evidence bundle.
