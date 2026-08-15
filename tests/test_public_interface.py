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
            self.assertIn("/backends/", result.stdout)
