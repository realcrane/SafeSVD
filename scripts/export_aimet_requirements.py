#!/usr/bin/env python3
"""Export the active Python environment as portable pip requirements."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import re
import subprocess
import sys


def export(output: Path) -> None:
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], check=True,
                            capture_output=True, text=True).stdout
    requirements = []
    for line in freeze.splitlines():
        if not line or line.startswith('#'):
            continue
        if ' @ ' in line:
            name = line.split(' @ ', 1)[0]
            # Conda wheels often retain build-machine file:/// paths.
            line = f'{name}=={metadata.version(name)}'
        if not re.fullmatch(r'[A-Za-z0-9_.-]+==[^\s/]+', line):
            raise ValueError(f'Cannot export a portable pinned requirement: {line!r}')
        requirements.append(line)
    torch_version = metadata.version('torch')
    cuda = re.search(r'\+cu(\d+)', torch_version)
    header = [
        '# aimet_env pip snapshot; Linux x86_64.',
        f'# Python {sys.version.split()[0]}; exported {datetime.now(timezone.utc).date()}.',
        '# Install in a new Python 3.11 environment: python -m pip install -r requirement.tx',
        '# Local Conda build paths are replaced with installed distribution versions.',
        '# GPU driver, datasets, checkpoints and upstream model source are separate.',
    ]
    if cuda:
        header += [f'--extra-index-url https://download.pytorch.org/whl/cu{cuda.group(1)}']
    output.write_text('\n'.join(header) + '\n\n' + '\n'.join(requirements) + '\n')
    print(f'Wrote {len(requirements)} pinned distributions to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('requirement.tx'))
    export(parser.parse_args().output)
