# RELLIS-3D official split snapshot

These files are the project-pinned RELLIS-3D semantic-segmentation split
snapshot. `adom.data prepare` treats the three files as an immutable input and
copies them into each canonical dataset package.

The identifiers use `<sequence>_<frame-stem>`. Samples are disjoint across
train, validation, and test, but sequences occur in more than one split. This
means neighbouring frames can be temporally correlated across splits. Every
evaluation report and model card must retain this caveat.

Expected counts:

| split | samples |
|---|---:|
| train | 3,302 |
| val | 983 |
| test | 1,672 |

Pinned SHA-256:

```text
29d6c1d7cbf7d94e18a7ce83dd20d85bedd614ff0901d93fbd19cac6329397e9  train.txt
741f67a4f181f4494bbea044ae248d587f41c72033ae97bb8952a58d84934c8c  val.txt
7a98484c9550b8d2825a7cc0b40c1789b452f30266047cb94bf757f99ad453d3  test.txt
```

Do not regenerate these files from a local processed dataset. If the upstream
official split changes, review it as a versioned data-contract change and
update these expected checksums.
