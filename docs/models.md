# Foundation-model weights

Use **Hugging Face Hub** as the public distribution point for all model
weights. It is the best fit here because it provides model repositories,
revisions, model cards, download metrics, and reliable large-file transfer;
it also supports custom PyTorch models, not only `transformers` models.

Create one Hugging Face organisation (recommended name: `SAFE-SVD`) and keep
the source code in this Git repository. Create one Hub model repository per
foundation-model family:

| Model | Hub repository | What it contains |
| --- | --- | --- |
| Poseidon | `camlab-ethz/Poseidon-<MODEL_SIZE>` | Upstream official weights only; do not mirror or redistribute them. |
| VICON | `SAFE-SVD/SAFE-SVD-VICON` | Your released VICON configuration, model weights, checksum manifest, and model card. |
| MPP | `SAFE-SVD/SAFE-SVD-MPP` | Your released MPP configuration, model weights, checksum manifest, and model card. |

Do not put VICON and MPP into one large shared repository: separate model
repositories give independent version histories, download counts, licences,
and releases. Use repository tags such as `v0.1-paper` after the reproduction
test passes. SAFE-SVD code should pin an immutable commit SHA, not `main`.

## Poseidon

Poseidon already distributes the pretrained foundation models through its
official Hugging Face collection. Follow its upstream instructions and load
the selected model size through `ScOT.from_pretrained`. The official README is
`poseidon/poseidon_main/README.md` in the separate source checkout.

The SAFE-SVD release must record the chosen upstream repository and immutable
revision in `configs/model_registry.yaml`. Do not upload a copy unless the
upstream licence explicitly permits redistribution and the original authors
have agreed.

## VICON and MPP: release now, in a safe order

1. Choose the exact checkpoint that produced the reported baseline results.
2. Export a loadable checkpoint and the exact architecture/configuration file.
   Prefer `safetensors` where the loading code can support it; otherwise keep
   the original `.pth` format and state that loading it requires trusted local
   files.
3. Compute SHA-256 for every uploaded checkpoint and place it in
   `checksums.txt` beside the weights.
4. Fill the corresponding model-card template in
   [`model_cards/`](model_cards/), including licence, data provenance,
   intended use, and the exact code commit.
5. Upload to a **private** Hugging Face repository first and download it into
   a fresh empty directory using the documented command.
6. Run SAFE-SVD paper reproduction from that downloaded asset directory.
7. Create tag `v0.1-paper`, make the repositories public, and copy their
   immutable commit revisions and checksums into `model_registry.yaml`.

For sizeable folders, use the Hugging Face `hf upload` command or
`huggingface_hub.upload_folder`; uploads are resumable and handle large model
files efficiently. Do not use a GitHub release as the primary weight host.

## Minimal download interface

The final README should provide commands of this form, with real identifiers
substituted before release:

```bash
# VICON
hf download SAFE-SVD/SAFE-SVD-VICON \
  --revision <pinned-commit-sha> \
  --local-dir /path/to/assets/vicon

# MPP
hf download SAFE-SVD/SAFE-SVD-MPP \
  --revision <pinned-commit-sha> \
  --local-dir /path/to/assets/mpp
```

The code should accept `--assets-dir`; it must not assume a personal training
directory such as `VICON/output/...` or `models/MPP_AViT_L`.
