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

    def test_embedded_modules_match_repository_sources_exactly(self) -> None:
        embedded: dict[str, str] = {}
        prefix = "%%writefile /kaggle/working/fraud-notebook-source/benchmark/"
        for cell in self.cells:
            source = self._source(cell)
            if source.startswith(prefix):
                first_line, body = source.split("\n", 1)
                name = "benchmark/" + first_line.rsplit("/", 1)[-1]
                embedded[name] = body

        local = {
            path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
            for path in sorted((ROOT / "benchmark").glob("*.py"))
        }
        self.assertEqual(set(embedded), set(local))
        for name in sorted(local):
            with self.subTest(module=name):
                self.assertEqual(embedded[name], local[name])

    def test_single_source_and_completion_guards_are_present(self) -> None:
        required_tokens = (
            "single-source-v3-r6-2026-07-12",
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
            "BLOCKED:",
            "FINAL_REQUIRED_STAGE_RECEIPTS",
            "stage_receipts_ready",
            "Non-monotonic phase transition refused",
            "ADAPTER_PROBE_INFO",
            "_reshape_binary_logits",
            "MPLBACKEND",
            "epochs_trained",
            "protocol_contrasts.csv",
            "worst_case_performance.csv",
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
        self.assertNotIn("SKIPPED:", self.all_source)

    def test_generated_upstream_sources_are_compiled_and_self_healing(self) -> None:
        patch_cell = next(
            self._source(cell)
            for cell in self.cells
            if "pmp_fallback = (" in self._source(cell)
        )
        parsed = ast.parse(patch_cell)
        fallback_assignment = next(
            node
            for node in ast.walk(parsed)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "pmp_fallback"
                for target in node.targets
            )
        )
        fallback = ast.literal_eval(fallback_assignment.value)
        self.assertEqual(
            fallback.splitlines(),
            [
                "try:",
                "    from torch_geometric.nn.norm import GraphNorm, GraphSizeNorm",
                "except ImportError:",
                "    GraphNorm = nn.Identity",
                "    GraphSizeNorm = nn.Identity",
            ],
        )
        compile("import torch.nn as nn\n" + fallback, "generated-pmp", "exec")
        self.assertNotIn("except Exception", fallback)

        required_tokens = (
            "'show', f'HEAD:{relative}'",
            "compile(expected, f'PMP/{relative}', 'exec')",
            "compile(sec_expected, f'SEC-GFD/{sec_relative}', 'exec')",
            "'-m', 'py_compile'",
            "GraphNorm(4)(features)",
            "module.GCN(4, 3, 2, graph)",
            "'diff', 'HEAD', '--check'",
            "allow_zero_in_degree=True",
        )
        for token in required_tokens:
            with self.subTest(token=token):
                self.assertIn(token, patch_cell)

        self.assertLess(
            patch_cell.index("compile(expected, f'PMP/{relative}', 'exec')"),
            patch_cell.index(".write_text("),
        )
        self.assertLess(
            patch_cell.index("module.GCN(4, 3, 2, graph)"),
            patch_cell.index("record_setup_receipt('upstream_patches'"),
        )
        self.assertEqual(
            patch_cell.count("record_setup_receipt('upstream_patches'"),
            1,
        )
        self.assertEqual(
            self.all_source.count("invalidate_setup_from('upstream_patches')"),
            1,
        )
        self.assertNotIn("if pmp_fallback not in source", patch_cell)

    def test_adapter_gate_executes_cuda_forward_backward_and_singleton_eval(self) -> None:
        adapter_cell = next(
            self._source(cell)
            for cell in self.cells
            if "Adapter CUDA forward/backward validation failed" in self._source(cell)
        )
        required_tokens = (
            "_make_dataloaders",
            "_build_pmp_model",
            "_predict_probs",
            "singleton_loader",
            "singleton_eval_rows",
            "_build_secgfd_model",
            "_resolve_secgfd_graph_device",
            "loss.backward()",
            "torch.cuda.synchronize()",
            "runtime_env(0)",
            "'upstream_patches', 'config', 'embedded_modules', 'cuda_runtime'",
        )
        for token in required_tokens:
            with self.subTest(token=token):
                self.assertIn(token, adapter_cell)
        self.assertEqual(adapter_cell.count("loss.backward()"), 2)
        self.assertEqual(adapter_cell.count("torch.cuda.synchronize()"), 2)
        self.assertLess(
            adapter_cell.index("loss.backward()"),
            adapter_cell.index("record_setup_receipt('adapters'"),
        )

    def test_native_cuda_dependency_is_pinned_and_proved_before_dgl(self) -> None:
        required_tokens = (
            "torch==2.1.0+cu118",
            "nvidia_cuda_runtime_cu11-11.8.89-py3-none-manylinux1_x86_64.whl",
            "f587bd726eb2f7612cf77ce38a2c1e65cf23251ff49437f6161ce0d647f64f7c",
            "nvidia-cuda-runtime-cu11",
            "libcudart.so.11.0",
            "d0da41ae1323cf4eeb610123d69d7714124cfe5ebfcc4e45f02b910e51c57ee6",
            "nvidia_cusparse_cu11-11.7.5.86-py3-none-manylinux1_x86_64.whl",
            "4ae709fe78d3f23f60acaba8c54b8ad556cf16ca486e0cc1aa92dca7555d2d2b",
            "nvidia-cusparse-cu11",
            "libcusparse.so.11*",
            "ldd",
            "Unresolved DGL native libraries",
        )
        for token in required_tokens:
            with self.subTest(token=token):
                self.assertIn(token, self.all_source)

        runtime_position = self.all_source.index("nvidia_cuda_runtime_cu11-11.8.89")
        cusparse_position = self.all_source.index("nvidia_cusparse_cu11-11.7.5.86")
        linker_position = self.all_source.index("['ldd', str(DGL_NATIVE_LIBRARY)]")
        self.assertLess(runtime_position, linker_position)
        self.assertLess(cusparse_position, linker_position)

    def test_setup_retries_invalidate_descendants_and_flags_are_safe(self) -> None:
        required_tokens = (
            "SETUP_RECEIPT_ORDER",
            "def invalidate_setup_from(stage: str)",
            "NATIVE_LINKER_VALIDATED = False",
            "CUDA_RUNTIME_VALIDATED = False",
            "ADAPTERS_VALIDATED = False",
            "NOTEBOOK_TESTS_PASSED = False",
            "globals().get('NATIVE_LINKER_VALIDATED') is True",
        )
        for token in required_tokens:
            with self.subTest(token=token):
                self.assertIn(token, self.all_source)

        setup_stages = (
            "preflight",
            "venv",
            "packages",
            "native_linker",
            "post_setup_disk",
            "upstreams",
            "upstream_patches",
            "config",
            "embedded_modules",
            "cuda_runtime",
            "adapters",
            "tests",
            "experiment_scale",
        )
        for stage in setup_stages:
            with self.subTest(stage=stage):
                self.assertIn(f"invalidate_setup_from('{stage}')", self.all_source)
        self.assertNotIn("assert NATIVE_LINKER_VALIDATED", self.all_source)

        setup_cell = next(
            self._source(cell)
            for cell in self.cells
            if "def invalidate_setup_from(stage: str)" in self._source(cell)
        )
        parsed = ast.parse(setup_cell)
        required_assignments = {
            "SETUP_RECEIPTS",
            "NATIVE_LINKER_VALIDATED",
            "CUDA_RUNTIME_VALIDATED",
            "ADAPTERS_VALIDATED",
            "NOTEBOOK_TESTS_PASSED",
            "RUNTIME_INFO",
            "ADAPTER_CLASSES",
            "ADAPTER_PROBE_INFO",
            "SETUP_RECEIPT_ORDER",
        }
        selected_nodes = []
        for node in parsed.body:
            if isinstance(node, ast.Assign):
                names = {
                    target.id
                    for target in node.targets
                    if isinstance(target, ast.Name)
                }
                if names & required_assignments:
                    selected_nodes.append(node)
            elif isinstance(node, ast.FunctionDef) and node.name == "invalidate_setup_from":
                selected_nodes.append(node)

        namespace: dict[str, object] = {}
        exec(compile(ast.Module(selected_nodes, type_ignores=[]), "setup-contract", "exec"), namespace)
        order = namespace["SETUP_RECEIPT_ORDER"]
        self.assertIsInstance(order, tuple)
        namespace["SETUP_RECEIPTS"] = {name: {"attempt_id": "x"} for name in order}
        for flag in (
            "NATIVE_LINKER_VALIDATED",
            "CUDA_RUNTIME_VALIDATED",
            "ADAPTERS_VALIDATED",
            "NOTEBOOK_TESTS_PASSED",
        ):
            namespace[flag] = True
        namespace["runtime_env"] = object()

        namespace["invalidate_setup_from"]("packages")

        self.assertEqual(set(namespace["SETUP_RECEIPTS"]), {"preflight", "venv"})
        self.assertFalse(namespace["NATIVE_LINKER_VALIDATED"])
        self.assertFalse(namespace["CUDA_RUNTIME_VALIDATED"])
        self.assertFalse(namespace["ADAPTERS_VALIDATED"])
        self.assertFalse(namespace["NOTEBOOK_TESTS_PASSED"])
        self.assertEqual(namespace["ADAPTER_PROBE_INFO"], {})
        self.assertNotIn("runtime_env", namespace)

    def test_completion_requires_fresh_report_and_interpretation_receipts(self) -> None:
        required_receipts = {
            "summaries_validated",
            "plots_validated",
            "report_artifacts_validated",
            "report_tables_loaded",
            "audit_table_validated",
            "ap_tables_validated",
            "figure_1",
            "figure_2",
            "figure_3",
            "mlp_invariant",
            "findings",
        }
        for receipt in required_receipts:
            with self.subTest(receipt=receipt):
                self.assertIn(f"'{receipt}'", self.all_source)

        final_cell = next(
            self._source(cell)
            for cell in self.cells
            if "FINAL_STATUS=complete" in self._source(cell)
        )
        self.assertIn("stage_receipts_ready", final_cell)
        self.assertIn("FINAL_REQUIRED_STAGE_RECEIPTS", final_cell)


if __name__ == "__main__":
    unittest.main()
