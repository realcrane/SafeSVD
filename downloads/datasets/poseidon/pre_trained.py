from huggingface_hub import snapshot_download
import subprocess
from pathlib import Path

dataset_names = [
    "CE-CRP", "CE-KH", "CE-RP", "CE-Gauss",
    "NS-Gauss", "NS-Sines"
]

for name in dataset_names:
    repo_id = f"camlab-ethz/{name}"
    local_dir = f"poseidon_main/camlab-ethz/pre_trained/{name}"
    print(f"Downloading {repo_id} to {local_dir}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )

for dataset in dataset_names:
    assemble_script = f"poseidon_main/camlab-ethz/pre_trained/{dataset}/assemble_data.py" 
    input_dir = f"poseidon_main/camlab-ethz/pre_trained/{dataset}"
    output_file = f"{input_dir}.nc"

    if not Path(input_dir).exists():
        print(f"[Skipped] Input directory does not exist: {input_dir}")
        continue

    cmd = [
        "python",
        assemble_script,
        "--input_dir", input_dir,
        "--output_file", output_file
    ]

    print(f"[Processing] {dataset}")
    try:
        subprocess.run(cmd, check=True)
        print(f"[Completed] {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"[Failed] {dataset}: {e}")
