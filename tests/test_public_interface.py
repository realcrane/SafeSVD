from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]


class PublicInterfaceTest(unittest.TestCase):
    def test_all_models_share_one_compress_interface(self) -> None:
        for model in ("poseidon", "vicon", "mpp"):
            result = subprocess.run(
                [
                    sys.executable, "-m", "safesvd.cli", "compress",
                    "--model", model,
                    "--checkpoint", f"/tmp/{model}-weights",
                    "--data-root", f"/tmp/{model}-data",
                    "--output-dir", f"/tmp/{model}-output",
                    "--rank-ratio", "0.2",
                    "--calibration-size", "16",
                    "--dry-run",
                ],
                cwd=PROJECT_DIR,
                env={"PYTHONPATH": str(PROJECT_DIR), "PYTHONDONTWRITEBYTECODE": "1"},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"SAFE-SVD {model}/paper", result.stdout)
            expected_route = "safesvd.adapters.poseidon_runner" if model == "poseidon" else "/backends/"
            self.assertIn(expected_route, result.stdout)

    def test_poseidon_default_output_includes_calibration_hyperparameters(self) -> None:
        result = subprocess.run(
            [
                sys.executable, "-m", "safesvd.cli", "compress",
                "--model", "poseidon",
                "--checkpoint", "/tmp/poseidon-weights",
                "--data-root", "/tmp/poseidon-data",
                "--rank-ratio", "0.2",
                "--calibration-size", "64",
                "--calibration-seed", "7",
                "--dataset", "fluids.incompressible.PiecewiseConstants",
                "--final-time", "20",
                "--dry-run",
            ],
            cwd=PROJECT_DIR,
            env={"PYTHONPATH": str(PROJECT_DIR), "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "artifacts/poseidon/paper/0.2/fluids.incompressible.PiecewiseConstants/"
            "traj64_seed7_ftime20",
            result.stdout,
        )
        self.assertIn("--calibration_seed 7", result.stdout)
