from __future__ import annotations

import os
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
            self.assertIn("--rank-selector " + ("safe" if model == "poseidon" else "legacy"), result.stdout)
            expected_route = f"safesvd.adapters.{model}_runner"
            self.assertIn(expected_route, result.stdout)

    def test_all_models_share_one_evaluate_interface(self) -> None:
        for model in ("poseidon", "vicon", "mpp"):
            result = subprocess.run(
                [
                    sys.executable, "-m", "safesvd.cli", "evaluate",
                    "--model", model,
                    "--checkpoint", f"/tmp/{model}-weights",
                    "--data-root", f"/tmp/{model}-data",
                    "--compressed-model", f"/tmp/{model}-compressed.pt",
                    "--output", f"/tmp/{model}-metrics.json",
                    "--dry-run",
                ],
                cwd=PROJECT_DIR,
                env={"PYTHONPATH": str(PROJECT_DIR), "PYTHONDONTWRITEBYTECODE": "1"},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"SAFE-SVD evaluation {model}", result.stdout)
            if model == "poseidon":
                self.assertIn("evaluation/poseidon/evaluate.py", result.stdout)
            else:
                self.assertIn(f"safesvd.adapters.{model}_runner evaluate", result.stdout)

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
            "artifacts/poseidon/paper/safe/0.2/fluids.incompressible.PiecewiseConstants/"
            "traj64_seed7_ftime20",
            result.stdout,
        )
        self.assertIn("--calibration_seed 7", result.stdout)

    def test_poseidon_runtime_python_can_be_overridden(self) -> None:
        environment = os.environ.copy()
        environment.update({
            "PYTHONPATH": str(PROJECT_DIR),
            "PYTHONDONTWRITEBYTECODE": "1",
            "SAFE_SVD_RUNTIME_PYTHON": "/tmp/poseidon-env/bin/python",
        })
        result = subprocess.run(
            [
                sys.executable, "-m", "safesvd.cli", "compress",
                "--model", "poseidon",
                "--checkpoint", "/tmp/poseidon-weights",
                "--data-root", "/tmp/poseidon-data",
                "--rank-ratio", "0.2",
                "--calibration-size", "64",
                "--dry-run",
            ],
            cwd=PROJECT_DIR,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "/tmp/poseidon-env/bin/python -X faulthandler -u -m safesvd.adapters.poseidon_runner",
            result.stdout,
        )

    def test_legacy_selector_has_distinct_output_and_rank_paths(self) -> None:
        result = subprocess.run(
            [
                sys.executable, "-m", "safesvd.cli", "compress",
                "--model", "vicon",
                "--checkpoint", "/tmp/vicon-weights",
                "--data-root", "/tmp/vicon-data",
                "--rank-ratio", "0.2",
                "--rank-selector", "legacy",
                "--calibration-size", "128",
                "--dry-run",
            ],
            cwd=PROJECT_DIR,
            env={"PYTHONPATH": str(PROJECT_DIR), "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("artifacts/vicon/paper/legacy/", result.stdout)
        self.assertIn("ranks/vicon/legacy/", result.stdout)
        self.assertIn("--rank-selector legacy", result.stdout)

    def test_poseidon_unified_evaluation_preserves_run_root_metrics_path(self) -> None:
        model = "/tmp/artifacts/poseidon/paper/safe/0.2/dataset/traj64/compressed/compressed_model.pt"
        result = subprocess.run(
            [
                sys.executable, "-m", "safesvd.cli", "evaluate",
                "--model", "poseidon",
                "--checkpoint", "/tmp/poseidon-weights",
                "--data-root", "/tmp/poseidon-data",
                "--compressed-model", model,
                "--dry-run",
            ],
            cwd=PROJECT_DIR,
            env={"PYTHONPATH": str(PROJECT_DIR), "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = Path("/tmp/artifacts/poseidon/paper/safe/0.2/dataset/traj64/metrics.json").resolve()
        self.assertIn(f"Output: {expected}", result.stdout)

    def test_backend_extra_cannot_override_public_paths(self) -> None:
        result = subprocess.run(
            [
                sys.executable, "-m", "safesvd.cli", "compress",
                "--model", "mpp",
                "--checkpoint", "/tmp/mpp-weights",
                "--data-root", "/tmp/mpp-data",
                "--output-dir", "/tmp/mpp-output",
                "--rank-ratio", "0.2",
                "--calibration-size", "16",
                "--dry-run",
                "--", "--model-path", "/tmp/other-weights",
            ],
            cwd=PROJECT_DIR,
            env={"PYTHONPATH": str(PROJECT_DIR), "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("public CLI options", result.stderr)
