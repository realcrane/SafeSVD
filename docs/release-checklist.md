# SAFE-SVD v0 public-release checklist

## Legal and provenance

- [ ] Select a licence for SAFE-SVD source code and add `LICENSE`.
- [ ] Verify that every copied or adapted upstream source file complies with
      its licence and retain required notices.
- [ ] Confirm you have the right to redistribute VICON and MPP weights.
- [ ] Document training-data provenance and licence/terms in both model cards.
- [ ] Confirm Poseidon weight redistribution is not needed; cite the official
      upstream repository instead.

## Weights

- [ ] Create the `SAFE-SVD` Hugging Face organisation.
- [ ] Create separate private repositories for VICON and MPP.
- [ ] Upload only the canonical paper-reference checkpoint and matching config.
- [ ] Add `checksums.txt`, model cards, a licence, and a revision tag
      (`v0.1-paper`) to each repository.
- [ ] Download each repository from a clean directory and verify its checksum.
- [ ] Replace all `TODO` model IDs, revisions, filenames, and hashes in
      `configs/model_registry.yaml`.

## Reproducibility

- [ ] Replace all personal absolute checkpoint/data paths with CLI arguments
      or configuration variables.
- [ ] Write complete VICON and MPP data acquisition steps in
      `docs/datasets.md`.
- [ ] Record seeds, dependency versions, GPU/runtime assumptions, and expected
      paper metrics.
- [ ] Run each paper route from a clean environment with downloaded assets.
- [ ] Publish a small result table with tolerance bounds for regression tests.

## Scope clarity

- [ ] Label `MySVD_fast` as an optimized/non-paper preset.
- [ ] Label the improved SAFE rank selector as experimental until benchmarked.
- [ ] Ensure the default route remains the paper-reference implementation.
