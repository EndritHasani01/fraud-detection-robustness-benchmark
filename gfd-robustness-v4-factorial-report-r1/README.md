# Final executed evidence: v4 factorial R1

This bundle supports the [academic report](../final-report/Graph_Fraud_Detection_Robustness_Academic_Report.pdf). Original evidence files are retained unchanged.

| Artifact | Purpose |
|---|---|
| [results.csv](results.csv) | Authoritative measurements: 5,040 rows, all status `ok` |
| [config.json](config.json) | Configuration captured for this execution |
| [graph_variants.csv](graph_variants.csv) | Graph variant ledger |
| [variant_audit.csv](variant_audit.csv) | Requested versus realized perturbations |
| [kaggle_run_manifest.json](kaggle_run_manifest.json) | Execution provenance |
| [plots/](plots/) | Figures, summary tables, protocol contrasts, and completeness diagnostics |
| [split_evidence/](split_evidence/) | Retained split evidence |
| [logs/](logs/) | Execution logs |
| [REPORT_BUNDLE_README.txt](REPORT_BUNDLE_README.txt) | Original bundle description and upstream revisions |

Each protocol contributes 2,520 rows. The four models are `mlp`, `sage`, `pmp`, and `secgfd`. Six split identifiers represent the factorial split/allocation design. Evaluation counts must not be treated as independent statistical replicates; oracle perturbations remain diagnostic.

Graph/data caches and the isolated runtime are intentionally excluded. See the [executed notebook](../notebooks/KAGGLE_DUAL_T4_RESEARCH_RUN_with_outputs_latest_v4_r1_multi_seeds.ipynb) and [reproduction guide](../docs/REPRODUCIBILITY.md). Paths in historical manifests reflect the original execution environment.
