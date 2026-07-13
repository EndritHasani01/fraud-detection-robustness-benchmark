from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = ROOT / "KAGGLE_DUAL_T4_RESEARCH_RUN.ipynb"
DEFAULT_CONFIG = ROOT / "configs" / "exp_yelpchi_v4_factorial.json"
EMBED_PREFIX = "%%writefile /kaggle/working/fraud-notebook-source/benchmark/"
HASH_PATTERN = re.compile(r"json\.loads\(r'''(\{.*?\})'''\)", re.DOTALL)
CONFIG_PATTERN = re.compile(
    r"(NOTEBOOK_CONFIG\s*=\s*json\.loads\(r''')(\{.*?\})('''\))",
    re.DOTALL,
)


def _source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def sync_notebook(
    notebook_path: Path,
    *,
    check: bool,
    config_path: Path = DEFAULT_CONFIG,
) -> list[str]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    embedded_paths: list[str] = []
    changed: list[str] = []

    expected_config_text = json.dumps(
        json.loads(config_path.read_text(encoding="utf-8")),
        indent=2,
        ensure_ascii=False,
    )
    config_cells = [cell for cell in notebook["cells"] if "NOTEBOOK_CONFIG" in _source(cell)]
    if len(config_cells) != 1:
        raise RuntimeError(f"Expected exactly one inline notebook config cell, found {len(config_cells)}")
    config_source = _source(config_cells[0])
    config_match = CONFIG_PATTERN.search(config_source)
    if config_match is None:
        raise RuntimeError("Inline NOTEBOOK_CONFIG JSON was not found")
    if config_match.group(2) != expected_config_text:
        changed.append(config_path.relative_to(ROOT).as_posix())
        config_cells[0]["source"] = (
            config_source[: config_match.start(2)]
            + expected_config_text
            + config_source[config_match.end(2) :]
        )

    for cell in notebook["cells"]:
        source = _source(cell)
        if not source.startswith(EMBED_PREFIX):
            continue
        first_line, _old_body = source.split("\n", 1)
        relative = "benchmark/" + first_line.rsplit("/", 1)[-1]
        local_path = ROOT / relative
        if not local_path.is_file():
            raise FileNotFoundError(f"Embedded module has no local source: {relative}")
        body = local_path.read_text(encoding="utf-8")
        replacement = f"{first_line}\n{body}"
        embedded_paths.append(relative)
        if replacement != source:
            changed.append(relative)
            cell["source"] = replacement

    local_modules = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "benchmark").glob("*.py")
    )
    if sorted(embedded_paths) != local_modules:
        missing = sorted(set(local_modules) - set(embedded_paths))
        stale = sorted(set(embedded_paths) - set(local_modules))
        raise RuntimeError(f"Notebook/local module set differs; missing={missing}, stale={stale}")

    expected_hashes = {
        relative: hashlib.sha256((ROOT / relative).read_text(encoding="utf-8").encode("utf-8")).hexdigest()
        for relative in local_modules
    }
    hash_json = json.dumps(expected_hashes, sort_keys=True)
    compile_cells = [
        cell
        for cell in notebook["cells"]
        if "# Compile and verify every notebook-embedded" in _source(cell)
    ]
    if len(compile_cells) != 1:
        raise RuntimeError(f"Expected exactly one embedded-module verification cell, found {len(compile_cells)}")
    compile_source = _source(compile_cells[0])
    match = HASH_PATTERN.search(compile_source)
    if match is None:
        raise RuntimeError("Embedded-module hash dictionary was not found")
    if match.group(1) != hash_json:
        changed.append("embedded-module-hash-contract")
        compile_cells[0]["source"] = (
            compile_source[: match.start(1)] + hash_json + compile_source[match.end(1) :]
        )

    if check:
        if changed:
            raise RuntimeError("Kaggle notebook is out of sync: " + ", ".join(changed))
        return []

    if changed:
        notebook_path.write_text(
            json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Synchronize benchmark/*.py into the clean Kaggle notebook.")
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check", action="store_true", help="Fail instead of writing when the notebook is stale.")
    args = parser.parse_args()
    notebook_path = args.notebook.resolve()
    changed = sync_notebook(
        notebook_path,
        check=bool(args.check),
        config_path=args.config.resolve(),
    )
    if changed:
        print("Updated:", ", ".join(changed))
    else:
        print("Notebook modules are synchronized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
