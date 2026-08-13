# TA0/TA1/TA2 parallel RunPod runbook

이 문서는 decision record 0009와 0010의 실행 계약이다. 세 condition은 같은 B0-E0
selected checkpoint, 선택된 TA recipe, dataset package, image SHA, seed와
optimizer-update budget을 사용한다.
TA checkpoint를 서로 이어 학습하거나 평균·ensemble하지 않는다.

## 1. Immutable package

원본 확인 결과 `/workspace/adom/datasets/raw/adomdata`에는 9개 capture group, 총
215 pair가 있다. raw package는 학습 입력이 아니다. 2026-08-13에 standalone
converter/validator가 133 train, 21 val, 61 test와 `_SUCCESS`를 생성했고 validator
`PASS`를 확인했다. 같은 날 released E1과 released standalone으로 공통 superset을
생성하고 validator `PASS`를 확인했다.

2026-08-12 실제 Network Volume snapshot과 lookup 명령은
[RunPod dataset inventory](../../status/runpod-dataset-inventory-2026-08-12.md)에
기록돼 있다. 최초 snapshot에서는 E1만 release 상태였고, 2026-08-13 후속 snapshot에서
ADOM standalone과 TA superset release를 확인했다. marker 없는 18 GB E1 staging
tree는 계속 학습 입력에서 제외한다.

```bash
export E1_ROOT=/workspace/adom/datasets/processed/adom_semantic20_rellis_rugd_ycor_v1
export ADOM_ROOT=/workspace/adom/datasets/processed/adom_zed2i_semantic20_v1
export TA_ROOT=/workspace/adom/datasets/processed/adom_semantic20_target_adaptation_v1

python -m adom.data.target_adaptation build \
  --e1-root "$E1_ROOT" \
  --standalone-root "$ADOM_ROOT" \
  --output-root "$TA_ROOT"

python -m adom.data.target_adaptation validate \
  --input-root "$TA_ROOT" \
  --write-success-marker
```

Expected split counts are TA0 4,435, TA1 4,568, TA2 10,001, canonical val 900,
canonical test 899, ADOM diagnostic val 21 and diagnostic test 61. `build` refuses
non-empty output; failed or changed inputs require a new package version, not in-place
mutation. Training Pods treat `TA_ROOT` read-only.

Verified package digests are manifest `183dda705e76b451dc383a81f517d36df3d6032f00002ab225421b9ae316b9dd`,
images `f07e1ed3a463ade04834f6de8e5c80c531d2b67be2ca1df78f3de4d0fe57ef87`, and
masks `975209f763326e5c86d9d54e55474997a76489ea6dcbf1ed51fe61f830c1bc69`.
All 14,636 image and mask entries are hardlinks, so source and target release roots are
immutable after validation.

TA0의 source exposure는 RELLIS 100%이지만 현재 config는 공통 package의
`splits/ta0_train.txt`와 `manifest.csv`를 요구하고 hook이 이를 강제한다. 따라서 TA0도
superset build/validate 이후 시작한다. 이 구조는 standalone sample을 TA0에 노출하지
않으며, 세 condition이 동일 package digest와 canonical RELLIS val/test를 공유하게 한다.
TA1/TA2 full run은 TA0에서 선택한 recipe commit을 상속하고 검증된 동일 `TA_ROOT`를
사용해야 한다.

## 2. E0 warm-start lock

Use the one selected B0-E0 checkpoint for all runs and record its SHA-256.

```bash
export E0_CKPT=/workspace/adom/checkpoints/b0-e0-selected.pth
export E0_SHA=$(sha256sum "$E0_CKPT" | awk '{print $1}')
export IMAGE_SHA=<integration-commit-full-sha>
export IMAGE=<dockerhub-user>/adom-mmseg:${IMAGE_SHA}
```

The runtime rejects a missing/mismatched SHA, a non-B0 encoder, or a decode head other
than 19 classes. The common scaffold currently supports a 1,000-update frozen-encoder
stage followed by a 5,000-update full-model stage. These are provisional ceilings until
the TA0 method-selection gate is complete. The selected recipe must use equal total
optimizer-update budgets across its controls. After the recipe is frozen, condition
branches may change only their train split, fixed source weights and identity.

## 3. Parallel gates

After the three condition commits are integrated, build one SHA-tagged image. Launch
one single-GPU Pod per condition. All Pods may read the Network Volume, but each must use
a distinct output and W&B run identity.

```bash
python -m adom.runtime.semantic20_cycle \
  --dataset "$TA_ROOT" \
  --experiment ta0 \
  --models b0 \
  --output /workspace/adom/runs/semantic20/target-adaptation-v1/ta0/seed42/smoke \
  --gate smoke \
  --seed 42 \
  --initial-checkpoint "$E0_CKPT" \
  --expected-initial-checkpoint-sha256 "$E0_SHA" \
  --expected-image-sha "$IMAGE_SHA"
```

Run the same command with `ta1` and `ta2` and their own output roots. Required order:

1. Three 50-update smoke runs in parallel; verify dataset/checkpoint contract,
   `source_exposure.json`, finite loss and backbone audit.
2. Three 500-update mini runs in parallel after smoke review.
3. Full runs for seeds 42, 43 and 44. Either use nine Pods concurrently or three Pods
   in three seed waves; the latter is cheaper and keeps dataset I/O predictable.
4. Select by canonical RELLIS validation with the non-degradation constraint. Do not
   unlock canonical test during tuning.
5. If a recipe is accepted, retrain one `TA-final` independently from the same E0
   checkpoint and evaluate canonical RELLIS test and ADOM held-out diagnostics once.

TA0 requests RELLIS 1.0. TA1 requests RELLIS 0.75 / ADOM 0.25. TA2 requests RELLIS
0.4375 / RUGD 0.25 / YCOR 0.0625 / ADOM 0.25. The deterministic sampler writes actual
draw counts every 100 iterations and restores them on resume. A run may resume only in
its original output root with the same experiment, seed, dataset content SHA and E0 SHA.

## 4. Deployment

Training changes weights, not the B0 graph, so TA0/1/2 have the same expected inference
latency class as E0. Export only the selected candidate to ONNX. Build the TensorRT
engine on the target Jetson, then compare accuracy artifacts and p50/p95 latency against
E0 under identical ZED resolution, TensorRT precision, clocks and subscriber settings.
Do not deploy an ensemble. Keep E0 as the rollback artifact until the selected TA model
passes file inference, live ZED inference and Go/Stop watchdog tests.
