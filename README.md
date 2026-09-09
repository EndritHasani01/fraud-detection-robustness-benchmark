# Graph Fraud Detection Robustness Benchmark

Course final project: a controlled study of how graph-based fraud detectors respond to changing features and graph structure on YelpChi.

## Start here

1. **Read the [final academic report (PDF)](final-report/Graph_Fraud_Detection_Robustness_Academic_Report.pdf).**
2. View the [presentation](final-report/Fraud_Detection_Robustness_Presentation.pptx).
3. Read the [detailed technical report and oral explanation](GFD_ROBUSTNESS_FINAL_REPORT.md).
4. Inspect the [executed evidence](gfd-robustness-v4-factorial-report-r1/README.md) and [notebook with outputs](notebooks/KAGGLE_DUAL_T4_RESEARCH_RUN_with_outputs_latest_v4_r1_multi_seeds.ipynb).

The retained v4 results contain **5,040 successful evaluation rows**: 2,520 for each training protocol, covering MLP, GraphSAGE, PMP, and SEC-GFD across six split/allocation identifiers. These counts were checked directly against the retained `results.csv` during repository organization on 9 September 2026.

## Research scope

The benchmark separates retraining on each perturbed graph (`train_on_variant`) from training on the clean graph and evaluating graph shift (`train_clean_eval_all`). It evaluates feature camouflage, relation camouflage, oracle and non-oracle heterophily rewiring, and noisy neighborhoods. Interpret severity within each scenario family.

Oracle scenarios use label information during perturbation construction and are diagnostic experiments. They do not establish operational attack robustness. PMP and SEC-GFD are integrated adapters under shared benchmark conditions, not claims of full paper reproduction. See the report for conclusions, uncertainty, and limitations.

## Repository guide

| Location | Contents |
|---|---|
| [final-report/](final-report/README.md) | Academic PDF, editable Word report, and presentation |
| [GFD_ROBUSTNESS_FINAL_REPORT.md](GFD_ROBUSTNESS_FINAL_REPORT.md) | Detailed report with evidence links |
| [gfd-robustness-v4-factorial-report-r1/](gfd-robustness-v4-factorial-report-r1/README.md) | Final results, audits, plots, logs, and execution provenance |
| [notebooks/](notebooks/README.md) | Executed notebook with outputs |
| [KAGGLE_DUAL_T4_RESEARCH_RUN.ipynb](KAGGLE_DUAL_T4_RESEARCH_RUN.ipynb) | Clean, self-contained Kaggle execution notebook |
| [benchmark/](benchmark/) | Runner, graph stresses, model adapters, and reporting |
| [configs/](configs/README.md) | Immutable experiment definitions, including historical versions |
| [tests/](tests/) and [tools/](tools/) | Regression checks and notebook synchronization |
| [docs/](docs/README.md) | Methodology, architecture, reproduction, and submission notes |
| [project-explanation/](project-explanation/README.md) | Beginner-oriented walkthrough and discussion preparation |
| [Repos/](Repos/) | Vendored research code; retained paths and upstream files |

Start with [reproduction instructions](docs/REPRODUCIBILITY.md) to run the code, or the [submission guide](docs/SUBMISSION_GUIDE.md) to review the project. Historical drafts, old run bundles, duplicate exports, and design assets are preserved locally in the Git-ignored `_local_archive/`. The [organization record](docs/REPOSITORY_ORGANIZATION.md) explains this separation.
