# Latest Kaggle rerun: root-cause and recovery report

## Scope

This report analyzes `KAGGLE_DUAL_T4_RESEARCH_RUN_with_outputs_latest.ipynb`. It supersedes the current-runtime diagnosis from the earlier saved execution while preserving `KAGGLE_ERROR_ROOT_CAUSE_REPORT.md` as historical evidence. The clean `KAGGLE_DUAL_T4_RESEARCH_RUN.ipynb` remains the single runtime source of truth; the notebook with outputs is evidence only and was not edited.

## Executive conclusion

Kaggle did not silently omit most cells. All 67 code cells executed with execution counts 1 through 67. Thirty-two executed cells deliberately printed `SKIPPED` because their fail-closed prerequisite gates detected an incomplete setup. This containment prevented the long cascade of secondary missing-file and undefined-variable errors seen in the first saved run.

The first and causal failure occurred in notebook cell 9, execution 5:

```text
AssertionError: PyTorch installed, but libcusparse.so.11 was not found under torch/lib or nvidia/*/lib.
```

The installed `torch==2.1.0+cu118` wheel did not contain `libcusparse.so.11`, while DGL 1.1.3+cu118's `libdgl.so` requires that soname. The installation log contains no separate CUDA 11 cuSPARSE runtime. Therefore, changing `LD_LIBRARY_PATH` could not repair this execution: no compatible cuSPARSE binary existed in the isolated environment to discover.

The second stored error, in cell 50 / execution 39, was the intentional consolidated setup gate. It correctly rejected the receipts made unavailable by the first failure. It was a consequence, not a second root cause.

## Observed execution facts

| Evidence | Latest saved execution |
|---|---:|
| Notebook cells | 91 |
| Code cells | 67 |
| Code cells with execution counts | 67 |
| Stored errors | 2 |
| Executed cells printing `SKIPPED` | 32 |
| Run span | 2026-07-10 21:04:37.737–21:07:44.071 UTC |
| Approximate duration | 3 minutes 6 seconds |
| GPUs | 2 × Tesla T4, 15,360 MiB each |
| GPU activity | 0% and 0 MiB on both devices |
| RAM | 31.3 GiB |
| Free disk before / after setup | 19.5 / about 14.4 GiB |
| Host / isolated Python | 3.12.13 / 3.11.15 |
| Installed PyTorch / DGL | 2.1.0+cu118 / 1.1.3+cu118 |
| Dataset, graph, training, or report completion | None |

The zero GPU use, three-minute duration, and absence of graph and result artifacts jointly prove that no scientific workload ran.

## Root-cause proof

1. Package installation and `pip check` completed, but `pip check` validates declared Python distribution requirements; it does not prove that every ELF `DT_NEEDED` library exists.
2. Direct inspection of the official PyTorch 2.1.0+cu118 CPython 3.11 wheel shows CUDA runtime, cuBLAS, cuDNN, and related libraries, but no `libcusparse.so.11`.
3. Direct inspection of DGL 1.1.3+cu118's `libdgl.so` shows native requirements for `libcudart.so.11.0`, `libcublas.so.11`, and `libcusparse.so.11`.
4. The saved run found no matching cuSPARSE file under either `torch/lib` or `site-packages/nvidia/*/lib`.
5. The official PyTorch CUDA 11.8 package index provides `nvidia-cusparse-cu11==11.7.5.86`, the missing native runtime component.

Primary package indexes: [PyTorch CUDA 11.8 wheels](https://download.pytorch.org/whl/cu118/torch/), [PyTorch CUDA 11.8 cuSPARSE](https://download.pytorch.org/whl/cu118/nvidia-cusparse-cu11/), and [DGL CUDA 11.8 wheels](https://data.dgl.ai/wheels/cu118/repo.html).

## Why so many cells printed `SKIPPED`

The exact dependency cascade was:

```text
missing libcusparse.so.11
  -> no native_linker receipt
  -> no verified upstream checkout
  -> no upstream patch or inline-config receipt
  -> no embedded-module, CUDA, adapter, test, or scale receipt
  -> setup validation refuses to advance
  -> graph, smoke, training, plotting, and packaging bodies remain blocked
```

The 32 guarded cells did execute. Their bodies did not run because their prerequisites were false. Four additional code cells only define helper functions and correctly produce no output. The corrected clean notebook now says `BLOCKED` instead of `SKIPPED` so the message cannot be confused with a Kaggle scheduler decision.

## Repairs in clean revision r3

### Native runtime repair

- Pins `torch==2.1.0+cu118` explicitly.
- Installs the genuine official `nvidia-cusparse-cu11==11.7.5.86` wheel by direct URL with SHA-256 `4ae709fe78d3f23f60acaba8c54b8ad556cf16ca486e0cc1aa92dca7555d2d2b`.
- Verifies the installed distribution versions, finds a real `libcusparse.so.11*`, hashes it, and records that evidence before issuing the package receipt.
- Prints all candidate native-library directories and resolved cuSPARSE files before assertions.
- Retains one shared `LD_LIBRARY_PATH`, runs `ldd` on DGL's actual `libdgl.so`, and rejects every unresolved soname.
- Retains independent DGL CUDA operations on physical GPUs 0 and 1.
- Does not create an unsafe CUDA 12-to-11 symlink.

### Pipeline and scientific hardening

- Uses implementation ID `single-source-v3-r3-2026-07-11`, experiment `gfd_kaggle_notebook_v3_final_r3`, output directory `gfd-robustness-v3-final-r3`, and archive `gfd-robustness-v3-report-r3.zip`; failed r2 evidence cannot be reused.
- Makes phase transitions monotonic and removes the smoke cell's temporary backward transition.
- Uses the complete eight-field smoke key, including `training_seed`, from the start.
- Clears stale in-memory graph-profile state before rebuilding current-attempt evidence.
- Adds current-attempt receipts and hashes for summaries, plot outputs, completeness artifacts, report tables, all three presentation figures, the MLP invariant, and derived findings.
- Requires every final-stage receipt before `FINAL_STATUS=complete`; a failed interpretation or stale report file can no longer be ignored.
- Makes requested CUDA execution fail closed in every adapter and records the effective device instead of silently falling back to CPU.
- Adds a graph-build source fingerprint and rejects same-shape caches with different immutable provenance.
- Corrects macro-F1 threshold search to include the all-negative candidate and permits a finite threshold just above the maximum validation score.
- Expands notebook-local regression coverage from 12 to 14 tests for the new threshold endpoint and cache-provenance contract.

## Acceptance checklist for the next Kaggle run

A target rerun is accepted only when all of the following are visible in retained outputs:

- [ ] The package receipt contains PyTorch 2.1.0+cu118, DGL 1.1.3+cu118, and CUDA 11 cuSPARSE 11.7.5.86.
- [ ] The notebook prints and hashes a resolved `libcusparse.so.11*` file.
- [ ] `ldd` on `libdgl.so` contains no `not found` entry.
- [ ] Independent DGL CUDA probes pass on physical GPUs 0 and 1.
- [ ] Every mandatory setup receipt is fresh and the consolidated setup gate passes.
- [ ] All 14 notebook-owned tests pass with no collection error.
- [ ] The graph ledger and audit each contain exactly 31 validated rows, representing 21 evaluated states.
- [ ] The real CUDA smoke matrix succeeds for MLP, GraphSAGE, PMP, and SEC-GFD.
- [ ] Each protocol has exactly 252 unique successful keys; the merged result has exactly 504.
- [ ] Results contain no missing, unexpected, duplicate, error, or non-finite successful row.
- [ ] All report and interpretation receipts belong to the current run attempt.
- [ ] `plots/missing_or_error_runs.csv` is empty.
- [ ] The final manifest and atomic ZIP both record completion, and the last code cell prints `FINAL_STATUS=complete`.
- [ ] No required setup, graph, training, or reporting phase prints `BLOCKED`.

## Validation boundary

Local review can prove JSON validity, Python syntax, embedded-source hash consistency, package and gate contracts, and repository regression behavior. It cannot prove the live Kaggle driver, network, DGL wheel, upstream repositories, full YelpChi graph cache, or several-hour dual-T4 workload. Revision r3 is therefore ready for a fresh **Run All**, but scientific success must still be established by the checklist above and a new output-bearing notebook.

Local validation of the delivered clean artifact passed the full 101-test suite (including six notebook-artifact contract tests), Ruff F/E9 checks, `git diff --check`, `nbformat` schema validation, all 67 code cells parsed, 91 unique cell IDs, and zero stored outputs. The delivered notebook has SHA-256 `884e48e05427e1b0fe1d46b8d604c5f309bf74e7f57963fe387f1a9d72a29a8d`.
