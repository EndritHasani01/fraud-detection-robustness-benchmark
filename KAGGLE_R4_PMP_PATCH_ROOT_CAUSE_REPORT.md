# Kaggle r4 PMP generated-source failure report

## Scope

This report analyzes the PMP adapter traceback supplied from the r4 Kaggle execution. The observed failure is the `IndentationError` in `Repos/PMP-master/model/LASAGE_S.py`; the clean notebook has been advanced to revision r5. The output-bearing notebook remains execution evidence and is not the source distributed for the next run.

## Executive conclusion

The pinned PMP repository was valid when downloaded. Revision r4 then replaced an optional PyTorch-Geometric import with a malformed multi-line string. The generated text had these indentation levels:

```text
0 spaces: try:
8 spaces:         from torch_geometric.nn.norm import GraphNorm, GraphSizeNorm
4 spaces:     except Exception:
```

Python therefore raised `IndentationError: unindent does not match any outer indentation level`. The same invalid payload was written into both `model/LASAGE_S.py` and `model/base.py`; the traceback happened to encounter `LASAGE_S.py` first.

This was a notebook source-generation defect, not a Kaggle scheduler issue, CUDA failure, missing PMP dependency, or defect in the pinned Git commit. SEC-GFD printed `SECGFD`, which proves its import probe had already passed in the supplied run.

Pinned primary sources: [PMP `LASAGE_S.py`](https://github.com/Xtra-Computing/PMP/blob/3f7629f6c180891a0bc1bba3c66d94d288a1ddae/model/LASAGE_S.py), [PMP `base.py`](https://github.com/Xtra-Computing/PMP/blob/3f7629f6c180891a0bc1bba3c66d94d288a1ddae/model/base.py), and [SEC-GFD `SECGFD.py`](https://github.com/Sunxkissed/SEC-GFD/blob/97faa51145ed1fbcbdc67cc5d399490da9a9797a/model/SECGFD.py).

## Why r4's checks accepted invalid Python

The defect crossed four gaps:

1. The notebook cell itself was valid Python. The invalid program existed only as text inside `pmp_fallback`, so parsing the cell did not parse the generated source.
2. The patch check reconstructed its expected file with the same malformed string. Consequently, `actual == expected`, diff hashes, and file hashes all agreed on invalid content.
3. The compile cell compiled only the notebook-owned `benchmark/` directory, not the three modified upstream files.
4. The adapter gate was the first operation that actually imported `model.LASAGE_S`, so it became the first honest syntax test.

Retrying r4 in the same session could not self-heal. Its conditional saw the malformed payload already present and skipped replacement. A second integrity cell could also enter using the earlier upstream receipt while relying on globals that a failed patch cell had not created.

## Corrections in revision r5

### Deterministic generated-source contract

- Uses an explicit five-line payload with four-space indentation and `except ImportError`.
- Aliases the unused optional normalization classes to `torch.nn.Identity`; it does not install PyG or unrelated upstream packages.
- Reads pristine text on every attempt with `git show HEAD:<path>` from the pinned clone.
- Asserts exactly one declared replacement in each source.
- Compiles every complete expected source before writing anything.
- Always writes the exact expected text, so a retry repairs a partial or malformed earlier write.
- Compiles all three written files again with the isolated Python 3.11 runtime.
- Runs isolated PMP import/normalization and SEC-GFD zero-in-degree GCN probes.
- Verifies exact file contents, changed paths, `git diff HEAD --check`, untracked status, diff hashes, and file hashes.
- Issues one `upstream_patches` receipt only after all checks pass; the unsafe second integrity cell was removed.

### Adapter forecasting gate

The old gate imported class names only. R5 now runs each exact adapter in a fresh subprocess on a T4 and requires:

- DGL sampling through the PMP adapter;
- PMP construction, CUDA forward, finite loss, backward, and finite gradients;
- a one-node PMP evaluation batch;
- SEC-GFD construction on a graph containing a zero-in-degree node;
- SEC-GFD CUDA forward, embedding path, finite loss, backward, and finite gradients;
- one visible logical GPU in each isolated subprocess;
- structured JSON evidence in the setup receipt.

This happens before YelpChi download, graph generation, or the 504-row experiment.

### Forecasted PMP shape defect

The pinned `LASAGE_S.forward()` ends with a bare `squeeze()`. A one-node batch becomes shape `[2]`, so the former adapter would fail at `softmax(dim=1)`. The fixed YelpChi split has non-singleton remainders, but r5 now restores and validates `[batch, 2]`, reshapes labels to `[batch]`, and tests the singleton path in both local and Kaggle gates.

The pinned upstream `relation_agg="mean"` and `"add"` branches reduce a Python list directly. R5 fails early with an actionable message if configuration drift selects either branch; the frozen experiment continues to use the verified `"cat"` branch.

## Dependency forecast

The selected PMP path requires PyTorch, DGL, tqdm, and its own `model.base`. Those packages are already present. PyG is optional for this configuration and is handled by the identity fallback. Packages referenced only by unused PMP modules—such as OGB, torch-scatter, torch-sparse, and imbalanced-learn—must not be added to the compact notebook.

Independent audit evidence also exercised one-epoch PMP train/evaluate on the complete 45,954-node, 8,051,348-edge YelpChi canonical graph and obtained finite metrics with the same DGL 1.1.x Python sampling path. Target CUDA correctness still belongs to the r5 in-notebook probes and real Kaggle smoke matrix.

## Remaining scientific boundary

PMP's upstream custom reducer has undefined behavior if an entire sampled block contains zero edges. The fixed YelpChi split has only 13 isolated base nodes—far fewer than its 512-node batches—and the selected scenarios do not make this condition an expected run state. R5 does not silently add self-loops or invent neighbor messages because either would change the evaluated method. The real clean-plus-largest-graph smoke remains mandatory; a future experiment that changes splits, batch sizes, or edge-removal semantics must add a methodologically reviewed zero-edge policy.

## Acceptance checklist for the next fresh Kaggle run

- [ ] The first cell reports `single-source-v3-r5-2026-07-12` and the r5 output path.
- [ ] Both pinned repositories checkout at their declared commits.
- [ ] Patch progress reaches `[patch 5/5]`.
- [ ] Python 3.11 compilation succeeds for both PMP files and SEC-GFD's model file.
- [ ] The exact patch/status contract records one `upstream_patches` receipt.
- [ ] Generic DGL CUDA probes pass independently on physical GPUs 0 and 1.
- [ ] PMP's CUDA probe reports `LASAGE_S`, shape `[8, 2]`, finite gradients, and one singleton evaluation row.
- [ ] SEC-GFD's CUDA probe reports `SECGFD`, shape `[8, 2]`, a zero-in-degree node, and finite gradients.
- [ ] All 15 notebook-owned tests pass.
- [ ] The later real four-model smoke passes on the clean and maximum-edge graphs.
- [ ] Both protocols finish with 252 successful keys each and 504 total.
- [ ] The final cell prints `FINAL_STATUS=complete` with no required stage blocked.

Use a new Kaggle session and **Run All** from the clean r5 notebook. Do not continue the failed r4 kernel or reuse its r4 output directory.

## Local r5 validation record

Local validation passed all 106 repository tests, including nine notebook-artifact contract tests and the executable singleton-logit regressions. Ruff F/E9 checks, `git diff --check`, `nbformat` schema validation, parsing of all 66 code cells, embedded-module hash verification, and idempotence of the notebook transformation also passed. The clean artifact has 90 unique cells, no stored outputs, and SHA-256 `8b6f1af1d145bfd6171b92cd945ecdb15f22a087f7283f233c6830daaa11d3e5`.

Local validation cannot substitute for Kaggle's CUDA driver, network, exact Linux wheels, or the full dual-T4 workload. Target completion still requires the acceptance checklist and `FINAL_STATUS=complete` from a fresh output-bearing run.
