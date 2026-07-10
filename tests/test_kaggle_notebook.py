from __future__ import annotations

import ast
import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "KAGGLE_DUAL_T4_RESEARCH_RUN.ipynb"


class KaggleNotebookArtifactTests(unittest.TestCase):
    @staticmethod
    def _source(cell: dict) -> str:
        value = cell.get("source", "")
        return "".join(value) if isinstance(value, list) else str(value)

    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.cells = cls.notebook["cells"]
        cls.all_source = "\n".join(cls._source(cell) for cell in cls.cells)

    def test_notebook_is_clean_and_has_unique_cell_ids(self) -> None:
        self.assertEqual(self.notebook["nbformat"], 4)
        identifiers = [cell["id"] for cell in self.cells]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        for cell in self.cells:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])

    def test_every_code_cell_and_embedded_module_parses(self) -> None:
        for index, cell in enumerate(self.cells):
            if cell["cell_type"] != "code":
                continue
            source = self._source(cell)
            if source.startswith("%%writefile "):
                source = source.split("\n", 1)[1]
            with self.subTest(cell=index):
                ast.parse(source, filename=f"notebook-cell-{index}")

    def test_embedded_module_hash_contract_matches_sources(self) -> None:
        embedded: dict[str, str] = {}
        prefix = "%%writefile /kaggle/working/fraud-notebook-source/benchmark/"
        for cell in self.cells:
            source = self._source(cell)
            if source.startswith(prefix):
                first_line, body = source.split("\n", 1)
                name = "benchmark/" + first_line.rsplit("/", 1)[-1]
                embedded[name] = hashlib.sha256(body.encode("utf-8")).hexdigest()

        compile_source = next(
            self._source(cell)
            for cell in self.cells
            if "# Compile and verify every notebook-embedded" in self._source(cell)
        )
        match = re.search(r"json\.loads\(r'''(\{.*?\})'''\)", compile_source, re.DOTALL)
        self.assertIsNotNone(match)
        expected = json.loads(match.group(1))
        self.assertEqual(embedded, expected)
        self.assertEqual(len(embedded), 22)

    def test_single_source_and_completion_guards_are_present(self) -> None:
        required_tokens = (
            "current_project_repo_downloaded': False",
            "RUN_FINGERPRINT_SHA256",
            "EXPECTED_PATCHED_FILE_HASHES",
            "'diff', 'HEAD'",
            "graph_cache_integrity_errors",
            "EXPECTED_EVALUATED_VARIANT_KEYS_FROM_CONFIG",
            "expected_scientific_keys",
            "SMOKE_TRAINING_SEED",
            "SMOKE_ATTEMPT_ERRORS",
            "require_disk_reserve",
            "report_zip_temp.replace(REPORT_ZIP)",
            "mark_phase('complete')",
            "SKIPPED:",
        )
        for token in required_tokens:
            with self.subTest(token=token):
                self.assertIn(token, self.all_source)
        self.assertNotIn(
            "github.com/EndritHasani/fraud-detection-robustness-benchmark",
            self.all_source,
        )
        self.assertNotIn(
            "MANIFEST_PATH.write_text(json.dumps(MANIFEST",
            self.all_source,
        )


if __name__ == "__main__":
    unittest.main()
