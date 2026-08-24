from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from safesvd.adapters.base import RunRequest
from safesvd.assets import AssetLayout
from safesvd.config import SafeSVDConfig
from safesvd.pipeline import _initialise_run


class PipelineTest(unittest.TestCase):
    def test_existing_run_directory_is_reused_without_removing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory) / "run"
            compressed_dir = run_dir / "compressed"
            compressed_dir.mkdir(parents=True)
            partial_checkpoint = compressed_dir / "compressing_model.pt"
            partial_checkpoint.write_text("partial")
            config = SafeSVDConfig(
                model="poseidon",
                checkpoint=Path("/tmp/checkpoint"),
                data_root=Path("/tmp/data"),
                output_dir=run_dir,
                rank_ratio=0.2,
                calibration_size=64,
                calibration_seed=-1,
            )
            request = RunRequest(
                repo_root=Path(temporary_directory),
                assets=AssetLayout(Path(temporary_directory), "poseidon"),
                run_dir=run_dir,
                device="cpu",
                config=config,
            )

            _initialise_run(request, ("python", "backend.py"))

            self.assertEqual(partial_checkpoint.read_text(), "partial")
            self.assertTrue((run_dir / "manifest.json").is_file())
            for child in ("cache", "ranks", "compressed", "evaluation"):
                self.assertTrue((run_dir / child).is_dir())
