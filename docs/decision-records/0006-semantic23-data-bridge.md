# 0006: GOOSE native preprocessing and contiguous Semantic23 data bridge

- Status: proposed for review
- Date: 2026-08-05

## Decision

GOOSE source normalization and ADOM label-space conversion are separate,
versioned stages.

`src/data/goose` reads the uploaded original train/val ZIP archives and
materializes visible windshield RGB plus the original 64-class label-ID masks.
It validates selected ZIP member CRCs, pairing, dimensions, and IDs, and writes
portable metadata. It does not remap labels or select frames.

`src/data/semantic_23` is a bridge over validated source outputs. It accepts:

- RELLIS Semantic20 indexed masks;
- RUGD original index-label masks and committed splits;
- YCOR original palette labels;
- GOOSE native output from `src/data/goose`.

The bridge converts one dataset at a time, writes a manifest and complete
zero-filled target-class statistics, then combines converted sources. Main
validation and test remain RELLIS-only; non-RELLIS validation/test samples are
diagnostic splits.

## Label contract

The existing RELLIS preprocessing has 19 trainable classes with IDs `0..18`.
The four GOOSE additions continue directly from ID 18 with no reserved or
unused output channel. Model output indices are:

- RELLIS semantic classes: `0..18`;
- snow: `19`;
- animal: `20`;
- artifact: `21`;
- cobble: `22`;
- ignore: `255`.

GOOSE artifact contains only `traffic_light`, `traffic_sign`, and `misc_sign`.
The GOOSE bridge enables mappings marked `use_for_phase1=true` in the reviewed
candidate table plus snow, animal, the artifact union, and the subsequently
approved mappings `cobble -> cobble`, `crops -> bush`,
`rail_track -> asphalt`, `moss -> dirt`, and `road_marking -> asphalt`.
`scenery_vegetation` remains ignore.

## GOOSE full-dataset materialization

Every GOOSE native image and mask is remapped and materialized. Per-image target
class counts and percentages remain in the audit CSV, but class presence and
class-ratio thresholds do not remove images. The GOOSE bridge manifest count
must equal the GOOSE-native input manifest count.

## Rationale

Separating immutable source normalization from experimental ontology mapping
allows mapping and selection policy changes without repeatedly unpacking the
large GOOSE archives. Source-specific validation remains owned by each dataset
pipeline, while the bridge has one explicit, reviewable label contract and
portable server CLI. Contiguous IDs avoid an unused decoder channel and match
the actual 19-trainable-class RELLIS implementation plus four GOOSE additions.
Full GOOSE materialization prevents selection-policy changes from silently
omitting source frames.
