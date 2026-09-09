# Reproducing the submitted experiment

## Inspect the completed run

The [executed notebook](../notebooks/KAGGLE_DUAL_T4_RESEARCH_RUN_with_outputs_latest_v4_r1_multi_seeds.ipynb) and [final evidence bundle](../gfd-robustness-v4-factorial-report-r1/README.md) document the completed experiment. The bundle's [config.json](../gfd-robustness-v4-factorial-report-r1/config.json) captures its configuration, and [kaggle_run_manifest.json](../gfd-robustness-v4-factorial-report-r1/kaggle_run_manifest.json) records execution provenance. These historical files are preserved unchanged.

## Execute the clean notebook

1. Upload [KAGGLE_DUAL_T4_RESEARCH_RUN.ipynb](../KAGGLE_DUAL_T4_RESEARCH_RUN.ipynb) to Kaggle.
2. Select the dual-T4 accelerator environment and enable internet access for dependency setup and dataset downloads.
3. Run the cells in order, including setup and runtime checks, before starting the full experiment.
4. Preserve the new results bundle, configuration, logs, and manifest when execution finishes.

The notebook embeds 22 benchmark Python modules, the factorial experiment configuration, and its notebook contract test. It writes its own runtime source tree and downloads PMP and SEC-GFD at pinned upstream revisions. It does not require this repository's separate `benchmark/`, `configs/`, `tests/`, `tools/`, or vendored code directories. Those development folders remain local and ignored.

The runtime uses its own `Repos/` paths inside Kaggle. The repository's vendored folder was renamed to `third_party/`; this does not change the notebook's independent runtime paths or the historical configuration. On the original development computer, an ignored `Repos` junction points to `third_party` so older local configurations still resolve.

The full factorial workflow manages graph caches split by split. Dataset downloads, graph caches, and the isolated Python runtime are not included in the submitted evidence bundle. A new execution is a new result set; do not overwrite the historical notebook or evidence.

## Interpretation and verification limits

The historical results table contains 5,040 successful rows, evenly divided between `train_on_variant` and `train_clean_eval_all`. These are evaluation rows, not independent statistical replicates. Oracle scenarios must remain diagnostic and separate from non-oracle claims.

Repository organization verified notebook/source synchronization and preservation of the notebook and evidence. It did not execute a new GPU experiment or verify current Kaggle hardware availability. The prior local regression suite passed 121 tests; its separate development files are intentionally excluded from this submission.
