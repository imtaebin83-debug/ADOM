# 0044. Repair the B5 primary mapping lock from the canonical source

- Status: Accepted
- Date: 2026-09-02
- Owners: perception training/research 담당자
- Supersedes: none; amends 0013 and 0041

## Context

The B5 static contract on the RTX 5090 correctly failed closed before any optimizer
update. It compared the current `bridge_mapping.yaml` identity with the frozen B5
primary-dataset lock and found a mismatch. The recorded mapping value ended with
`a21dd389...`, which is 65 lowercase hex characters and therefore cannot be a
SHA-256 value.

The live immutable image at Git SHA
`f00e49d640ee41a0398dd7282832542d5dfdba8e` was used only to read and hash the
canonical source file:
`/opt/adom/src/data/semantic_20/config/bridge_mapping.yaml`. Its SHA-256 is
`ecfa61662ddbf16c801bcac22db11b0e7ee2408d635e3018a21d389933a6bc55`.

## Decision

Replace the invalid 65-hex `bridge_mapping.yaml` value in the B5 static lock and
the two B2/B5 provenance tables with the raw-source 64-hex SHA-256 above. Extend
the static validator's identifier-format check to include nested mapping digests,
and pin the exact mapping SHA in its unit test.

No dataset file, dataset manifest, split, mapping content, B2 checkpoint, B5 recipe,
or GO metric is changed. This is a provenance transcription repair only.

## Rationale and evidence

The B5 contract's mapping digest is calculated from the repository's canonical
`src/data/semantic_20/config/bridge_mapping.yaml`, not a copied mapping under a
processed dataset root. The hash command on the immutable image returned the
64-hex value recorded above. The previous 65-hex string differs only by a duplicated
`d` after `a21` and is invalid by SHA-256 length alone.

The failure occurred before the active GPU memory probe and before either B5 Stage 1
or Stage 2 could begin, so no optimizer update or model artifact was produced under
the invalid lock.

## Alternatives considered

- Bypass the static contract for the current pod: rejected because it would permit a
  run without its split/mapping/data-manifest guard.
- Drop the mapping lock: rejected because it weakens the preregistered primary-data
  identity guarantee.
- Change the processed dataset or substitute a different mapping: rejected because
  the existing Semantic20 mapping and primary split remain authoritative.

## Consequences

- A new immutable Git-SHA image is required before B5 execution can continue.
- The corrected nested digest is format-validated, so a future 65-hex mapping typo
  blocks with a direct identifier error rather than appearing as dataset drift.
- All prior B2 evidence, the B5 GO artifact, RELLIS-val-only selection, and the
  Korean held-out test-only lock remain unchanged.

## Validation and rollback

Run the focused B5 contract tests and the artifact guard, then the training-image
CI allowlist and immutable-image checks. On the RTX 5090, rerun the full static
contract, active memory probe, and only then launch B5-E0 in tmux. If the corrected
lock still disagrees with the canonical source, stop before training and retain the
new static-contract report as the blocker.
