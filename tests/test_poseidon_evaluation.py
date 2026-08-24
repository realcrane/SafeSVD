from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None


EVALUATION_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "poseidon"


@unittest.skipIf(torch is None, "PyTorch is required for Poseidon evaluation tests")
class PoseidonEvaluationTest(unittest.TestCase):
    def test_increases_are_percentages_rounded_to_six_places(self) -> None:
        sys.path.insert(0, str(EVALUATION_DIR))
        from physics import get_metric_increases

        increases = get_metric_increases(
            {"base_loss": 1.123456789, "model_size": 1},
            {"base_loss": 1.0, "model_size": 2},
        )
        self.assertEqual(increases, {"base_loss": 12.345679})

    def test_compressed_metrics_are_written_at_run_root(self) -> None:
        sys.path.insert(0, str(EVALUATION_DIR))
        from evaluate import default_output_path

        model = Path("/tmp/artifacts/poseidon/paper/0.2/dataset/traj64_ftime20/compressed/compressed_model_origin.pt")
        self.assertEqual(
            default_output_path(Path("/tmp/checkpoint"), model),
            Path("/tmp/artifacts/poseidon/paper/0.2/dataset/traj64_ftime20/metrics.json"),
        )
