# Kaggle r3 canonical-cuDART failure report

## Scope

This report analyzes the runtime output supplied from revision r3 of `KAGGLE_DUAL_T4_RESEARCH_RUN.ipynb`. Only the pasted native-linker and CUDA-probe failures are treated as observed evidence; no complete output-bearing r3 notebook was supplied. The clean notebook has been advanced to revision r4.

## Executive conclusion

Revision r3 successfully repaired the earlier missing-cuSPARSE failure. Its linker output resolves both `libcusparse.so.11` and `libcublas.so.11`. It then exposes the remaining direct DGL dependency:

```text
libcudart.so.11.0 => not found
```

PyTorch 2.1.0+cu118 contains `torch/lib/libcudart-d0da41ae.so.11.0`, a hash-renamed internal file. DGL's `libdgl.so` requests the canonical filename and soname `libcudart.so.11.0`, so adding `torch/lib` to `LD_LIBRARY_PATH` cannot make `ldd` resolve it by that name.

The official `nvidia-cuda-runtime-cu11==11.8.89` wheel supplies the exact canonical file under `site-packages/nvidia/cuda_runtime/lib/`. Revision r4 installs that wheel alongside the already-correct cuSPARSE provider.

## Observed linker evidence

The supplied output establishes the following directly:

| Native dependency | r3 result |
|---|---|
| `libcudart.so.11.0` | Not found |
| `libcublas.so.11` | Resolved from `torch/lib` |
| `libcusparse.so.11` | Resolved from `nvidia/cusparse/lib` |
| `libcublasLt.so.11` | Resolved transitively from `torch/lib` |
| Linux system libraries | Resolved |

This is useful progress: it proves the r3 cuSPARSE package and library discovery work. The `ldd` assertion is also behaving correctly by refusing to issue the `native_linker` receipt while one soname remains unresolved.

## Exact official runtime provider

- Distribution: `nvidia-cuda-runtime-cu11==11.8.89`
- [Official PyTorch CUDA 11.8 package index](https://download.pytorch.org/whl/cu118/nvidia-cuda-runtime-cu11/)
- Wheel: `nvidia_cuda_runtime_cu11-11.8.89-py3-none-manylinux1_x86_64.whl`
- Wheel SHA-256: `f587bd726eb2f7612cf77ce38a2c1e65cf23251ff49437f6161ce0d647f64f7c`
- Canonical file: `nvidia/cuda_runtime/lib/libcudart.so.11.0`
- Canonical file SHA-256: `d0da41ae1323cf4eeb610123d69d7714124cfe5ebfcc4e45f02b910e51c57ee6`

The wheel is about 0.84 MiB, supports Python 3, and declares no Python dependencies. It is the minimal genuine provider; revision r4 does not create a symlink to PyTorch's hash-renamed file.

## Native dependency closure

Direct inspection of DGL 1.1.3+cu118's `libdgl.so` shows three CUDA dependencies: `libcudart.so.11.0`, `libcublas.so.11`, and `libcusparse.so.11`. The supplied r3 `ldd` output already resolves the latter two. The cuSPARSE and CUDA-runtime libraries add only ordinary Linux system dependencies, all present in the target output. No cuRAND, cuSOLVER, nvJitLink, cuDNN, or other CUDA wheel is predicted for DGL's native load.

The NVIDIA driver library is loaded at runtime rather than appearing as this unresolved `DT_NEEDED` entry. The existing independent DGL operation on each physical T4 remains the final driver and GPU validation.

## Why the later `NameError` appeared

The later error was:

```text
NameError: name 'NATIVE_LINKER_VALIDATED' is not defined
```

That error is secondary. In a pristine top-to-bottom r3 run, the failed `ldd` cell cannot record `native_linker`, so the receipt-gated CUDA cell should print `BLOCKED`. Entering the CUDA body with an undefined success flag indicates divergent notebook state from partial or out-of-order re-execution.

The underlying weakness was real: r3 initialized success flags only when a stage passed, and retrying a setup cell did not invalidate every downstream receipt. A prior same-attempt receipt could therefore survive a failed retry while its matching variable was absent or stale.

## Corrections in revision r4

- Installs both exact NVIDIA providers by direct official wheel URLs with SHA-256 fragments:
  - `nvidia-cuda-runtime-cu11==11.8.89`
  - `nvidia-cusparse-cu11==11.7.5.86`
- Verifies both distribution versions and canonical files before issuing `packages`.
- Hashes `libcudart.so.11.0` and asserts its known official file digest.
- Prints both resolved canonical files before `ldd`.
- Records the complete successful `ldd` output in the native-linker receipt.
- Initializes native-linker, CUDA, adapter, and test flags to `False`.
- Adds an ordered setup receipt graph. Every retried setup cell invalidates itself and all descendants before doing work.
- Removes the bare `assert NATIVE_LINKER_VALIDATED`; the CUDA cell now requires both fresh receipts and `globals().get(...) is True`.
- Uses implementation/output/report/experiment revision r4 so no partial r3 state is reused.

## Acceptance checklist for the next fresh run

- [ ] Package output reports CUDA runtime 11.8.89 and cuSPARSE 11.7.5.86.
- [ ] The inventory prints and hashes `nvidia/cuda_runtime/lib/libcudart.so.11.0`.
- [ ] The inventory prints and hashes `nvidia/cusparse/lib/libcusparse.so.11*`.
- [ ] `ldd libdgl.so` contains zero `not found` lines.
- [ ] The `native_linker` receipt is recorded only after that complete `ldd` result.
- [ ] Independent DGL CUDA operations pass on physical GPUs 0 and 1.
- [ ] A failed or manually retried setup stage causes later cells to print `BLOCKED`, not `NameError`.
- [ ] The remaining graph, smoke, 504-row matrix, report, and final-manifest checks from the prior acceptance checklist pass.

Revision r4 is locally validated but still requires a fresh Kaggle **Run All** to establish target-runtime and scientific completion.

Local validation passed the full 102-test suite, including seven notebook-artifact contract tests and an executable setup-invalidation test. Ruff F/E9, `git diff --check`, `nbformat` validation, and parsing of all 67 code cells passed. The clean r4 notebook has 91 unique cells, zero stored outputs, and SHA-256 `0b121afca544f9a145e3fdbc583accf5eea04cd56ceea304688101fa4db3f6e2`.
