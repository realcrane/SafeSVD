# SAFE-SVD implementations

These are the source implementations now maintained in SAFE-SVD-v0. They were
copied without algorithmic rewrites from the corresponding working research
directories; no foundation-model checkpoints, datasets, caches, or experiment
outputs are included.

```text
implementations/
├── poseidon/
│   ├── paper/      # MySVD_origin: paper-compatible reference implementation
│   └── safe_rank/  # improved rank selection + accelerated compression
├── vicon/
│   └── paper/      # MySVD: paper-compatible reference implementation
└── mpp/
    └── paper/      # MySVD: paper-compatible reference implementation
```

The next migration step is to bring the three foundation-model runtime source
trees (Poseidon `poseidon_main`, VICON `VICON/src`, and MPP
`multiple_physics_pretraining`) under a declared external dependency contract,
then replace their remaining absolute/legacy-relative paths with the common
asset layout. Do not make mathematical changes while doing that migration.
