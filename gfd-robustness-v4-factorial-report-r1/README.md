# Final executed evidence: v4 factorial R1

This is the evidence bundle referenced by the [technical report](../GFD_ROBUSTNESS_FINAL_REPORT.md). Original evidence files were retained unchanged during organization.

| Artifact | Purpose |
|---|---|
| [results.csv](results.csv) | Authoritative per-evaluation measurements: 5,040 rows, all status `ok` |
| [config.json](config.json) | Configuration captured for this execution |
| [graph_variants.csv](graph_variants.csv) | Graph variant ledger |
| [variant_audit.csv](variant_audit.csv) | Requested versus realized perturbations |
| [kaggle_run_manifest.json](kaggle_run_manifest.json) | Execution provenance and implementation details |
| [plots/](plots/) | Figures, summary tables, protocol contrasts, and completeness diagnostics |
| [split_evidence/](split_evidence/) | Retained split evidence |
| [logs/](logs/) | Execution logs |
| [REPORT_BUNDLE_README.txt](REPORT_BUNDLE_README.txt) | Original bundle description and upstream revision identifiers |

Each protocol contributes 2,520 rows. Model identifiers are `mlp`, `sage`, `pmp`, and `secgfd`. Six split identifiers represent the factorial split/allocation design. These are evaluation counts, not 5,040 independent statistical replicates.

The bundle intentionally excludes graph/data caches and the isolated execution runtime. See the [executed notebook](../notebooks/KAGGLE_DUAL_T4_RESEARCH_RUN_with_outputs_latest_v4_r1_multi_seeds.ipynb) and [reproducibility guide](../docs/REPRODUCIBILITY.md). Oracle perturbations remain diagnostic and must stay distinct from non-oracle claims.
