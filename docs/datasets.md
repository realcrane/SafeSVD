# Dataset acquisition and preparation

Do not redistribute datasets from this repository unless every source licence
explicitly permits it. SAFE-SVD should give users a short working procedure,
state the expected local path, and link to the upstream authoritative download
instructions. A bare sentence saying “see the source code” is not sufficient
for reproducibility.

Each model route should expose this three-part contract:

1. **Source and access:** provider, landing page, licence, and any required
   account/terms acceptance.
2. **Command or exact upstream procedure:** a copyable download command when
   one exists; otherwise a direct link to the relevant upstream section.
3. **Prepared layout:** exact directory supplied to SAFE-SVD, required splits,
   and a lightweight verification command or expected file list.

## Poseidon

Poseidon datasets are published by the upstream authors on Hugging Face. The
official code README gives the dataset-to-repository mapping and the assembly
steps. From `poseidon/poseidon_main`, the basic upstream download pattern is:

```bash
huggingface-cli download camlab-ethz/<DATASET_IDENTIFIER> \
  --repo-type dataset --local-dir /path/to/assets/poseidon/data/<dataset>
```

Then follow the **Download & Assembly** instructions in the upstream Poseidon
`poseidon_main/README.md`. SAFE-SVD must name the
specific calibration and test datasets used by each published preset, rather
than asking users to download the whole collection.

## VICON

VICON's paper route currently loads datasets through
`VICON.src.dataset.all_datasets(...)`, including `NS2D`, `COMPRESSIBLE2D`, and
`EULER2D`. Before public release, add to this section:

- the upstream repository or landing-page URL for each dataset;
- the licence and access requirement for each source;
- the exact download/preprocessing command; and
- the local root passed to VICON's data configuration.

The public configuration must replace private machine paths with a parameter
such as `--data-root /path/to/assets/vicon/data` or `SAFE_SVD_DATA_ROOT`.
The current workspace does not include `VICON/src` or its Hydra configs, so
the adapter already forwards both values but the loader-side read of this
setting must be added when that source tree is migrated into SAFE-SVD.

## MPP

MPP configuration files declare `train_data_paths` and `valid_data_paths`.
The upstream MPP code documents its expected HDF5-directory dataset interface
in `mpp/multiple_physics_pretraining/README.md` in the source checkout.

Before public release, list for every dataset named in the selected MPP
configuration:

- provider and source URL;
- licence/terms and whether a user account is required;
- download and any conversion/assembly command;
- expected HDF5 directory layout; and
- which files are used for calibration versus final evaluation.

If the same data come from PDEBench, link users directly to its data-download
documentation rather than copying assets into SAFE-SVD.
