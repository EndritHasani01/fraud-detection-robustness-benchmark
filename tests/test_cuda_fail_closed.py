from __future__ import annotations

from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest import mock

import torch

from benchmark.baselines_stage import _resolve_baseline_device
from benchmark.pmp_stage import _build_pmp_model, _resolve_requested_device
from benchmark.secgfd_stage import _resolve_secgfd_graph_device


class _TransferFailGraph:
    def to(self, device: str):
        raise RuntimeError(f"simulated graph transfer failure: {device}")


class _TransferFailPmpModel:
    def __init__(self, **_kwargs) -> None:
        pass

    def to(self, device: str):
        raise RuntimeError(f"simulated model transfer failure: {device}")


class CudaFailClosedTests(unittest.TestCase):
    def test_cuda_unavailable_is_never_downgraded_to_cpu(self) -> None:
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.subTest(model="baseline"):
                with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                    _resolve_baseline_device("mlp", object(), "cuda")
            with self.subTest(model="pmp"):
                with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                    _resolve_requested_device("cuda")
            with self.subTest(model="secgfd"):
                with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                    _resolve_secgfd_graph_device(_TransferFailGraph(), device="cuda")

    def test_graph_transfer_failures_are_not_hidden_by_cpu_fallback(self) -> None:
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            with self.subTest(model="graphsage"):
                with self.assertRaisesRegex(RuntimeError, "could not be moved to CUDA"):
                    _resolve_baseline_device("sage", _TransferFailGraph(), "cuda")
            with self.subTest(model="secgfd"):
                with self.assertRaisesRegex(RuntimeError, "could not be moved to CUDA"):
                    _resolve_secgfd_graph_device(_TransferFailGraph(), device="cuda")

    def test_pmp_model_transfer_failure_is_not_hidden_by_cpu_fallback(self) -> None:
        graph = SimpleNamespace(
            ndata={
                "feature": torch.zeros((2, 3), dtype=torch.float32),
                "label": torch.tensor([0, 1], dtype=torch.int64),
            },
            etypes=["_E"],
        )
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            with mock.patch(
                "benchmark.pmp_stage._import_pmp_model",
                return_value=_TransferFailPmpModel,
            ):
                with self.assertRaisesRegex(RuntimeError, "could not be moved to CUDA"):
                    _build_pmp_model(
                        graph,
                        repo_root=Path("unused"),
                        cfg_pmp={},
                        device="cuda",
                    )


if __name__ == "__main__":
    unittest.main()
