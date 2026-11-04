import argparse
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ALL_DATASETS = [
    "NS-PwC", "NS-BB", "CE-RPUI", "CE-RM", "Wave-Layer", "Wave-Gauss"
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=ALL_DATASETS,
        default=ALL_DATASETS,
        help="Datasets to process. Defaults to all datasets.",
    )
    args = parser.parse_args()
    datasets = list(dict.fromkeys(args.dataset))

    for name in datasets:
        snapshot_download(
            repo_id=f"camlab-ethz/{name}",
            repo_type="dataset",
            local_dir=f"poseidon/camlab-ethz/down_streams/{name}",
            local_dir_use_symlinks=False,
        )

    for dataset in datasets:
        input_dir = Path("poseidon/camlab-ethz/down_streams") / dataset
        assemble_script = input_dir / "assemble_data.py"
        output_file = Path(f"{input_dir}.nc")

        if not input_dir.exists():
            print(f"[Skipped] Input directory does not exist: {input_dir}")
            continue

        cmd = [
            sys.executable,
            str(assemble_script),
            "--input_dir", str(input_dir),
            "--output_file", str(output_file),
        ]

        print(f"[Processing] {dataset}")
        try:
            subprocess.run(cmd, check=True)
            print(f"[Completed] {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"[Failed] {dataset}: {e}")


if __name__ == "__main__":
    main()
