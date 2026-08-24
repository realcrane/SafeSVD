"""Public SAFE-SVD pipeline API.

The three mathematical backends are still being regression-migrated.  This
module is the sole supported application interface; it maps one common config
to the matching backend adapter and records a portable run manifest.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path

from safesvd.adapters import get_adapter
from safesvd.adapters.base import RunRequest
from safesvd.assets import AssetLayout
from safesvd.config import SafeSVDConfig


@dataclass(frozen=True)
class CompressionResult:
    output_dir: Path
    backend_command: tuple[str, ...]
    returncode: int | None


def compress(config: SafeSVDConfig, *, dry_run: bool = False) -> CompressionResult:
    """Run SAFE-SVD rank selection then compression for one foundation model."""
    config.validate()
    output_dir = config.output_dir.expanduser().resolve()
    assets = AssetLayout(
        assets_dir=output_dir.parent,
        model=config.model,
        model_path=config.checkpoint.expanduser().resolve(),
        data_root=config.data_root.expanduser().resolve(),
    )
    request = RunRequest(
        repo_root=Path(__file__).resolve().parents[1],
        assets=assets,
        run_dir=output_dir,
        device=config.device,
        config=config,
    )
    adapter = get_adapter(config.model)
    command = tuple(adapter.command(request))
    if dry_run:
        return CompressionResult(output_dir, command, None)

    adapter.validate(request)
    _initialise_run(request, command)
    return CompressionResult(output_dir, command, adapter.execute(request))


def _initialise_run(request: RunRequest, command: tuple[str, ...]) -> None:
    """Prepare a reproducible run directory without blocking resumable runs."""
    request.run_dir.mkdir(parents=True, exist_ok=True)
    for child in ("cache", "ranks", "compressed", "evaluation"):
        (request.run_dir / child).mkdir(exist_ok=True)
    config = asdict(request.config)
    config.update({key: str(value) for key, value in config.items() if isinstance(value, Path)})
    manifest = {
        "schema_version": 1,
        "safe_svd_config": config,
        "backend_command": list(command),
        "backend_command_shell": shlex.join(command),
    }
    (request.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
