# RELLIS-3D Semantic20 Preprocessing

This directory contains preprocessing for the original RELLIS-3D
semantic ontology.

## Label policy

- Original ontology classes: 20, including `void`
- Trainable semantic classes: 19
- Train IDs: `0` through `18`
- Ignore index: `255`
- `sky` is retained as a trainable semantic class
- Only `void` is converted to `255`

## Version scope

`rellis3d_semantic20_v1` contains only RELLIS-3D data and labels.

RUGD and YCOR samples are not included in this version. Any later
multi-dataset training set will be maintained separately.

## Repository scope

This directory stores preprocessing scripts, mappings, official split
files, tests, QC summaries, and representative previews.

Raw RGB images, full masks, processed datasets, archives, model weights,
and training artifacts are not tracked in Git.
