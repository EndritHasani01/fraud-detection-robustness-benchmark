# Shared Code Consolidation

## Context

The three training stages (`baselines_stage.py`, `pmp_stage.py`, `secgfd_stage.py`) each contain their own copies of the `VariantRow` dataclass, the `_read_variants_csv` function, the `_set_seeds` function, `_parse_bool`, `_safe_int`, `_safe_float`, `_load_graph_bin`, and `_class_weights_from_train_labels`. These are functionally identical across files. This duplication makes it harder to implement cross-cutting changes (like adding the `protocol` column or the skip-existing logic) because every change must be applied three or four times.

## Consolidation Plan

Create a new module `benchmark/variants.py` that contains the shared code:

- `VariantRow` dataclass (single definition)
- `read_variants_csv(path: Path) -> list[VariantRow]`
- `filter_variants(variants, *, include_noop: bool, only_clean: bool, max_variants: int | None) -> list[VariantRow]` - extracts the filtering logic that is also duplicated across all stages
- `load_graph_bin(path: Path) -> DGLGraph`
- `set_seeds(seed: int) -> None` (the version that seeds torch, numpy, random, and optionally DGL)
- `parse_bool(x) -> bool`
- `safe_int(x, default=0) -> int`
- `safe_float(x, default=None) -> float | None`
- `class_weights_from_train_labels(y_train) -> Tensor | None`

Then update each stage file to import from `benchmark.variants` instead of defining its own copies. The private underscore prefix on these functions should be removed since they are now part of a shared module.

## Row-Writing Helper

The pattern for writing a result row (building `row_common`, then adding metric fields or error fields, then calling `append_csv_row`) is also duplicated. Consider adding a helper function:

```python
def write_result_row(
    results_csv: Path,
    variant: VariantRow,
    model_id: str,
    training_seed: int,
    protocol: str,
    metrics: dict | None,  # None means error
    error: str = "",
    duration_sec: float = 0.0,
) -> None:
```

This would centralize the row construction logic and ensure that new columns (like `protocol` or `train_graph_ref`) are always populated consistently.

## Scope

This is a refactoring task with no behavioral changes. All existing tests and CLI behaviors should remain identical after the consolidation. It should be done early in the v2 implementation (ideally right after the resumability work) because it simplifies all subsequent changes. Without it, adding skip-existing logic means editing the same pattern in four files.
