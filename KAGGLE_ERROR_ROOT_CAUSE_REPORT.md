# Kaggle notebook error root-cause report

## Superseding update from the second Kaggle execution

A second saved execution is analyzed in [`KAGGLE_LATEST_RUN_ROOT_CAUSE_REPORT.md`](KAGGLE_LATEST_RUN_ROOT_CAUSE_REPORT.md). In that rerun, all 67 code cells executed; 32 deliberately printed `SKIPPED`, and only two stored errors remained. The fail-closed gates therefore fixed the earlier cascade behavior.

The rerun also refines this report's native-linker diagnosis. Prepending native-library directories is necessary but was insufficient: the installed PyTorch 2.1.0+cu118 wheel did not contain `libcusparse.so.11`, and no separate CUDA 11 cuSPARSE runtime package was installed. The repaired clean notebook now explicitly installs and inventories `nvidia-cusparse-cu11==11.7.5.86` before running `ldd` and the two-GPU DGL probes. Treat the new report as authoritative for the current repair.

## Scope and evidence

This report analyzes the stored outputs and execution metadata in `KAGGLE_DUAL_T4_RESEARCH_RUN_with_outputs.ipynb`. It distinguishes failures that were actually observed from risks in code paths that the failed run never reached. It does not treat a file's existence, a successful setup cell, or a header-only CSV as evidence that the experiment ran.

The uploaded notebook contains 86 cells: 63 code cells and 23 Markdown cells. Of the code cells, 23 contain a stored error, 38 executed without a stored error, and 2 have no execution count. The meaningful unexecuted cell is the Figure 2 cell (cell 77); the other is an empty trailing cell (cell 85).

## Executive conclusion

The benchmark did not download YelpChi, build any graph variant, train any model, or generate the final report bundle. Two independent failures occurred:

1. The disk preflight required 35 GiB free even though the Kaggle session exposed only 19.5 GiB free.
2. DGL's CUDA native library could not resolve `libcusparse.so.11` in the isolated Python 3.11 runtime.

The second failure is the experiment-blocking root cause. It prevented DGL import, then graph generation, and therefore every downstream smoke, training, and reporting phase. Kaggle continued executing queued cells after exceptions, which converted the one native-library failure into a long series of missing-file, undefined-variable, and empty-result errors.

## High-confidence runtime facts

| Evidence | Observed value |
|---|---:|
| GPUs | 2 x Tesla T4 |
| Memory per GPU | 15,360 MiB |
| NVIDIA driver | 580.159.04 |
| RAM | 31.3 GiB |
| Free disk before setup | 19.5 GiB |
| Free disk after runtime installation | 14.4 GiB |
| Internet probe | HTTP 200 |
| Host notebook Python | 3.12.13 |
| Isolated Python | 3.11.15 |
| Installed PyTorch | 2.1.0+cu118 |
| Installed DGL wheel | 1.1.3+cu118 for CPython 3.11 |
| First executed cell timestamp | 2026-07-10 17:48:27 UTC |
| Last executed cell timestamp | 2026-07-10 17:53:46 UTC |
| Total observed run span | About 5 minutes 19 seconds |
| GPU heartbeat after setup | 0% utilization, 0 MiB used on both GPUs |
| Successful benchmark result rows | 0 |

The short run time, zero GPU usage, empty model/status table, and missing graph ledger jointly prove that no model training occurred. They are stronger evidence than merely noting that the final assertions failed.

## Root causes and cascade map

| Level | Cell / execution | Exact evidence | Classification | Correction in the clean notebook |
|---|---|---|---|---|
| Root A | 4 / 2 | `Free disk: 19.5 GiB`, followed by `AssertionError: At least 35 GiB free disk is required before setup.` | Independent preflight-design failure | Report total and free capacity separately; use transparent state-dependent reserves plus post-install, post-graph, and live checks instead of an arbitrary 35 GiB gate. |
| Root B | 38 / 31 | DGL fails in `ctypes.CDLL(...)` with `OSError: libcusparse.so.11: cannot open shared object file` | Experiment-blocking native dependency failure | Locate the CUDA 11 shared libraries installed with the runtime, prepend all relevant library directories to every child process's `LD_LIBRARY_PATH`, check DGL with `ldd`, then probe both GPUs independently. |
| Direct cascade | 39 / 32 | PMP adapter import reaches DGL and raises the same `libcusparse.so.11` error | Same as Root B | Use the corrected common runtime environment; probe PMP and SEC-GFD in separate subprocesses so one cannot mask the other. |
| Direct cascade | 42 / 34 | Pytest stops during collection on the same DGL `OSError`; zero tests execute | Same as Root B | Apply the native-library environment to the test subprocess and require an actual passing test count. |
| Direct cascade | 52 / 40 | Graph stage reports `DGL FraudDataset is required` and exits in 1.37 seconds | Same as Root B, obscured by exception wrapping | Preserve the underlying exception and traceback; do not replace all DGL import errors with a generic installation message. |
| Data cascade | 53, 54, 56, 62, 65, 69 | Repeated `FileNotFoundError` for `graph_variants.csv` or plot-stage message that it is absent | Graph generation never completed | Do not execute dependent phases unless the graph ledger and audit pass their content gates. |
| State cascade | 57 / 44 | `NameError: smoke_results_path is not defined` | Cell 56 exited before assigning the variable | Make each phase consume validated artifacts rather than relying on partially created notebook globals. |
| Result cascade | 63 / 48 and 66 / 50 | Bare assertions fail; cell 63 displays an empty model/status table | No training task launched and result table has zero rows | Validate expected versus observed run keys with explicit diagnostics before summaries or plots. |
| Reporting cascade | 70, 72, 73, 74, 76, 78, 80, 81, 83 | Missing report CSVs, undefined `audit`/`drops`/figure variables, and an empty-data assertion | No graph or result evidence exists | Gate reporting on 504 unique successful rows and all graph-audit invariants. Validate data before creating figures. |

## Detailed observed failures

### Disk preflight is inconsistent with the actual target session

Cell 4 correctly detects both T4s, sufficient GPU memory, 31.3 GiB RAM, and working Internet. It then fails only the 35 GiB free-space assertion. Despite being described as a hard preflight, execution continues into environment setup.

The isolated environment reduces free space from 19.5 GiB to 14.4 GiB. Therefore, changing the assertion to a warning without further work would also be unsafe: graph-cache space must be estimated and monitored. The correction should budget separately for the Python runtime, downloaded dataset, graph variants, transient lane outputs, and a safety margin.

### Installation succeeds, native linking does not

Cell 7 successfully installs a 2.3259 GB PyTorch CUDA 11.8 wheel and the DGL 1.1.3 CUDA 11.8 CPython 3.11 wheel. This rules out a simple `pip install` failure. The official [PyTorch previous-version instructions](https://docs.pytorch.org/get-started/previous-versions/) document CUDA-specific wheel indexes, and DGL's official [CUDA 11.8 wheel index](https://data.dgl.ai/wheels/cu118/repo.html) lists `dgl-1.1.3+cu118-cp311-cp311-manylinux1_x86_64.whl`, which is the wheel installed by the notebook.

The direct observed cause is that the Linux dynamic loader cannot resolve the `libcusparse.so.11` soname when DGL loads its C library. The failed revision never checks where that library was installed and never adds PyTorch/NVIDIA native-library directories to the child environment. Cell 48's `benchmark_env` sets DGL, buffering, thread, and GPU-selection variables, but not `LD_LIBRARY_PATH`. Cells 38, 39, and 42 construct separate environments with the same omission.

Import order is not a sufficient correction. The CUDA probe imports DGL before PyTorch, but the test module imports PyTorch before DGL and still fails identically. The robust repair is native-library discovery plus a loader check. PyTorch 2.1's own [native dependency loading code](https://github.com/pytorch/pytorch/blob/v2.1.0/torch/__init__.py) also demonstrates that CUDA dependencies are native shared objects that must be found and loaded; the notebook must validate this explicitly for the external DGL library.

### Error wrapping hides the useful cause

The failed revision's embedded `benchmark/data.py` catches every exception around `from dgl.data.fraud import FraudDataset` and raises a generic message saying that DGL and PyTorch must be installed. That message is false in this run: both packages are present. Its embedded `benchmark/run.py` then catches the replacement exception and prints only its string, discarding the chained traceback from the streamed graph log.

The adapter probe exposes the real cause only because its uncaught subprocess traceback retains the exception chain. The graph command should provide the same level of evidence. Narrow exception handling, explicit native-dependency diagnostics, and `traceback.print_exc()` are required.

### Regression tests never run

Cell 42 reports one collection error and four deprecation warnings. No notebook-owned test body executes. Therefore, the later successful config-validation cell is not evidence that scenario determinism, no-op behavior, metric correctness, or protocol run keys passed in Kaggle.

### Empty summaries create false signals of progress

Cell 67 succeeds and writes four summary CSV files even though the unified result table is header-only. Cell 70 shows the result file at about 0.000309 MiB and each summary at about 0.000182 MiB. These are schemas, not results.

Only 6 of the 18 artifacts listed in cell 70 exist: the config, header-only results, and four header-only summaries. Twelve required graph, audit, plot, or completeness artifacts are absent. The notebook must gate summary generation on exactly 504 canonical successful run keys and reject empty or protocol-mixed outputs.

### The MLP invariant did not scientifically fail

Cell 80 displays an empty `mlp_edge_check` table. Its `.max()` value is `NaN`, so `NaN < 1e-10` evaluates false and produces a bare assertion. This is an empty-input validation defect, not evidence that a feature-only MLP changed under a graph-only shift.

### The manifest can misrepresent partial execution

Cell 38 initializes `RUNTIME_INFO = []` before its first probe fails. Cell 46 later writes the initial manifest using that empty list, so the manifest cell appears successful without two validated GPU runtime records.

Cell 83 mutates the in-memory manifest with a completion timestamp, `final_result_rows = 0`, and `missing_or_error_rows = 0` before failing because `figure_1` does not exist. In this run the mutated manifest is not written, but stale variables from a prior partial execution could make this unsafe. The corrected notebook resolves this with fresh-attempt setup receipts, phase state, atomic manifest writes, exact-key validation, and an atomically replaced ZIP whose internal manifest is already complete.

## Execution-order and notebook-state evidence

Execution counts are monotonic from 1 through 61, so there is no evidence that a corrected cell was rerun. Instead, the queued run continues after every exception:

- Cell 4 fails, but environment installation starts afterward.
- Cell 38 fails, but adapter probing, pytest, graph generation, training, plotting, and presentation cells continue.
- Most downstream training cells complete in milliseconds, consistent with immediate missing-file failures rather than training.

Cell 77, which should create Figure 2, is anomalous. It has execution metadata at `2026-07-10T17:53:36.864Z`, but it has no execution count, `execute_input` record, or output. Cell 78 then runs as execution 58. This cell was silently skipped or cancelled. The empty cell 85 also has no execution count but is not functionally relevant.

Because queued Kaggle execution did not stop on assertion failures, the corrected notebook uses fresh setup receipts, explicit state gates, and concise `SKIPPED` messages so unavailable downstream phases do not raise secondary missing-variable or missing-file errors.

## Non-fatal warnings observed

- Cell 6 inherits `UV_SYSTEM_PYTHON`/`--system`, which has no effect for `uv venv`.
- Cell 6 cannot hardlink across filesystems and falls back to copying.
- Cell 9 prints Git default-branch hints for both initialized upstream repositories.
- Cell 42 reports four mpmath deprecation warnings in addition to the fatal collection error.

The official [uv environment-variable reference](https://docs.astral.sh/uv/reference/environment/) documents both `UV_SYSTEM_PYTHON` and `UV_LINK_MODE`. The corrected setup removes the inherited system-Python setting for virtual-environment creation and sets `UV_LINK_MODE=copy` intentionally. The [uv environments guide](https://docs.astral.sh/uv/pip/environments/) confirms that `uv venv --python 3.11` is the appropriate isolated-environment pattern.

The notebook also does not truly pin uv: it requests `uv==0.8.15` only if no executable exists, but the output shows preinstalled uv 0.11.13 was used. This did not cause the DGL failure, but it weakens reproducibility.

## Predicted later risks and their corrected-notebook disposition

The failed run never reached these paths. They must not be reported as errors that already occurred.

1. **Additional missing CUDA sonames.** The corrected notebook discovers bundled CUDA directories, runs `ldd`, and rejects every `not found` line. This is structurally mitigated but still needs a fresh Kaggle execution.
2. **Disk exhaustion during graph caching.** The corrected notebook reports total/free disk, checks before and after setup, checks after graph generation, shows actual unique graph bytes, and enforces a live 1 GiB emergency reserve while subprocesses run. End-to-end storage fit remains target-unverified.
3. **SEC-GFD import remains untested in the saved evidence.** The corrected notebook probes PMP and SEC-GFD independently, but those new probes have no retained Kaggle output yet.
4. **Stale partial artifacts can be reused.** Fresh setup receipts, a consolidated runtime/upstream fingerprint, exact config/module/patch hashes, per-lane fingerprint files, graph binary hashes, and exact scientific-key comparisons now prevent existence-only reuse.
5. **Changed notebook revisions can collide with old output.** The corrected notebook uses a new implementation identifier, experiment name, output directory, and report ZIP name.
6. **Blank figures can be displayed before failure.** Corrected figure cells check the report phase and required input variables before constructing figures.
7. **Bare assertions lack useful evidence.** Operational count/key/metric checks now include observed values and missing/unexpected key samples; a few simple internal invariants remain assertions.
8. **Long-run model failures would be under-diagnosed.** Outer and per-run tracebacks are now retained in streamed/lane logs, and interrupted parents reclaim child GPU processes in `finally` blocks.
9. **Runtime duration remains unverified.** No evidence from this run establishes that all 504 evaluations finish within Kaggle's session limit.
10. **Scientific findings remain unavailable.** No robustness curve, model comparison, MLP invariant, or perturbation-effect conclusion can be drawn until the complete graph audit and 504 successful rows exist.

## Corrections implemented in the clean notebook

The clean `KAGGLE_DUAL_T4_RESEARCH_RUN.ipynb` now implements the following. These are source-level corrections, not a claim that Kaggle has executed them successfully:

1. Replace the 35 GiB constant with a transparent storage budget, show total and free space, and recheck free space after environment installation and graph generation.
2. Discover `torch/lib` and any `site-packages/nvidia/*/lib` directories immediately after package installation.
3. Build one common runtime environment that prepends those directories to `LD_LIBRARY_PATH` before every isolated-runtime process starts.
4. Locate DGL's native library, run `ldd`, and fail with the complete unresolved-library list.
5. Run independent DGL CUDA operations on physical GPUs 0 and 1 and require two successful records before constructing the manifest.
6. Probe PMP and SEC-GFD separately so both results are visible.
7. Preserve chained exception tracebacks in dataset loading and the benchmark CLI.
8. Require pytest execution to report exactly 12 passing notebook-owned tests.
9. Add fresh setup receipts and explicit phase gates. Queued downstream cells print a concise `SKIPPED` line when an earlier phase is unavailable instead of accessing missing files or stale variables.
10. Gate notebook summary and plot stages on the complete 504-key matrix, and reject empty or protocol-mixed summary outputs.
11. Validate exactly 31 ledger rows, 31 audit rows, 21 evaluated graph states, 252 successful rows per protocol, and 504 unique successful final run keys.
12. Validate report inputs before creating figures and write completion metadata atomically only after every required artifact passes.
13. Explicitly record the uv version in the run fingerprint and remove avoidable warnings with `UV_LINK_MODE=copy` and a clean virtual-environment invocation.
14. Validate graph binaries by size and SHA-256, publish graph/manifest/ZIP artifacts atomically, and preserve the last completed ledger if a rebuild fails.
15. Compare exact config-derived scientific keys—not only row counts—at lane, protocol, final-matrix, and packaging gates.
16. Repair tied-score Average Precision, reject non-finite successful metrics, and disclose the SEC-GFD cosine-domain numerical correction.

The intended dataset path itself is correct: DGL 1.1.3 documents [`FraudDataset('yelp')`](https://www.dgl.ai/dgl_docs/en/1.1.x/generated/dgl.data.FraudDataset.html), including its downloaded-data directory and graph features, labels, and masks. The observed failure happens before that constructor runs, so it is a runtime-linking problem rather than a YelpChi API problem.

## Local verification of the corrected artifact

The regenerated clean notebook has 91 cells, including 67 code cells and 22 embedded benchmark modules. Every code cell and every `%%writefile` module parses, cell IDs are unique, execution counts are null, stored outputs are empty, and the embedded-source hashes match the hash contract inside the notebook. The final clean artifact has SHA-256 `6EBD2C658A7EB59BC044473509B65F7F3781E906EA846FB182264D98D4BE472E`.

The local repository suite passes 90 tests, the focused undefined/unused-import lint checks pass, and `git diff --check` reports no whitespace errors. These checks validate source structure and CPU-testable semantics; they do not execute Kaggle's DGL/CUDA runtime or the 504-run matrix.

## Target-environment validation limitations

This forensic report can prove what failed in the uploaded Kaggle execution, and static/local tests can verify notebook structure and ordinary Python behavior. They cannot prove that the corrected CUDA runtime works on Kaggle's current image or that the complete benchmark fits its time, VRAM, RAM, and disk limits.

A fresh Kaggle run with Internet enabled and `GPU T4 x2` remains mandatory. Completion requires retained output evidence for all of these gates:

- The disk budget passes both before and after environment setup.
- `ldd` reports no unresolved DGL native dependencies.
- DGL imports and executes a CUDA graph operation independently on both physical T4s.
- PMP and SEC-GFD import independently.
- Notebook-owned tests execute and pass; collection success alone is insufficient.
- YelpChi downloads through DGL and the graph/audit ledgers each contain 31 valid rows.
- All four one-epoch CUDA paths produce eight successful smoke rows: one clean and one largest-edge graph per model.
- The standard protocol contains 252 unique successful rows.
- The clean-train shift protocol contains 252 unique successful rows with clean training references.
- The unified result table contains 504 unique successful run keys and no error rows.
- Missing/error diagnostics are empty and every required report CSV and PNG exists.
- The final manifest is written only after these conditions and records two validated GPU runtime entries.

Until a fresh target run supplies this evidence, the corrected notebook should be described as structurally and diagnostically validated, not as runtime-complete.
