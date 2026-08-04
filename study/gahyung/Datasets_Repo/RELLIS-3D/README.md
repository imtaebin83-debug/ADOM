# RELLIS-3D Preprocessing

This directory contains versioned preprocessing workflows for the
RELLIS-3D dataset.

## Versions

- `rellis3d_cost4_v1/`
  - Existing ADOM Cost4 preprocessing.
  - Target IDs: `0`, `1`, `2`, `3`, and `255`.

- `rellis3d_semantic20_v1/`
  - RELLIS-3D original semantic ontology preprocessing.
  - The ontology contains 20 classes including `void`.
  - Training masks use 19 semantic classes and `255` for `void`.

## Repository scope

This repository stores preprocessing scripts, mappings, split files,
tests, QC summaries, and representative previews.

Raw RGB images, full masks, processed datasets, archives, model weights,
and training artifacts are not tracked in Git.
