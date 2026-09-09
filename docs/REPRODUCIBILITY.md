# Reproducing and checking the benchmark

## Evidence versus a new run

The [final evidence bundle](../gfd-robustness-v4-factorial-report-r1/README.md) and [executed notebook](../notebooks/KAGGLE_DUAL_T4_RESEARCH_RUN_with_outputs_latest_v4_r1_multi_seeds.ipynb) document the completed experiment. Consult the bundle's `config.json` and `kaggle_run_manifest.json` for the captured settings and revisions. The frozen repository [v4 factorial config](../configs/exp_yelpchi_v4_factorial.json) defines the factorial experiment; preserve all existing config JSON files unchanged.

## Kaggle execution

Open the root [clean notebook](../KAGGLE_DUAL_T4_RESEARCH_RUN.ipynb) in Kaggle, select the dual-T4 accelerator environment, enable the network access needed for setup, and follow its cells in order. Its setup cells specify the execution dependencies; its factorial workflow manages graph caches split by split. Preserve the newly generated evidence bundle when the run completes. Hardware availability, downloads, and a complete new training execution were not verified as part of repository organization.

Check that embedded source and configuration match the repository before uploading:

```powershell
py tools/sync_kaggle_notebook.py --check
```

To deliberately refresh the clean notebook after a source change, run the same command without `--check`; do not overwrite the archived executed snapshot.

## Local development

Use Python 3.10/3.11 for the DGL/PyTorch research stack. The [CLI and environment reference](CLI_REFERENCE.md) preserves the existing platform-specific installation recipe and full command reference; the clean notebook contains the execution setup. Install the appropriate DGL/PyTorch builds for the chosen platform before training. Windows PMP requires `num_workers=0`.

After configuring dependencies, run a small developer experiment from the repository root:

```powershell
py -m benchmark.run --config configs/exp_yelpchi_v3_fast.json --stage graphs
py -m benchmark.run --config configs/exp_yelpchi_v3_fast.json --stage baselines
py -m benchmark.run --config configs/exp_yelpchi_v3_fast.json --stage plots
```

This smoke experiment is not the complete v4 study. Runtime output goes into ignored `runs/`; final submission evidence is retained separately. Training is resumable using the unified run key, and model thresholds must come from validation data.

## Regression checks

```powershell
py -m unittest discover -s tests
py tools/sync_kaggle_notebook.py --check
git diff --check
```

Some tests need scientific dependencies; skipped tests do not validate model execution. Organization changes preserve source code, notebook contents, frozen JSON configurations, and research adapters. Local checks do not establish a new GPU run or production performance.
