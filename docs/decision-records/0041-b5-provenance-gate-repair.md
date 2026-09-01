# 0041. Repair B5 provenance locks from raw B2 artifacts

- Status: Accepted
- Date: 2026-09-01
- Owners: perception training/research 담당자
- Supersedes: none; amends 0013 and extends 0014

## Context

0013은 보존된 B2 run record의 primary dataset과 evaluation identifier 여덟 개가
65 hex로 기록됐기 때문에 B5 static/GO gate를 의도적으로 닫았다. 2026-09-01에
Network Volume의 B2 full-run `dataset_contract.json`, fresh-evaluation summaries,
`dataset_manifest_summary.json`, `checkpoint_manifest.json`과 per-class CSV를
읽기 전용으로 재감사했다. 각 65-hex identifier에는 중복 문자 하나가 있었으며,
실제 artifact에는 서로 일치하는 64-hex SHA-256이 남아 있었다.

또한 B0와 B2 fresh evaluation은 같은 evaluator policy를 사용하지만 resolved model
architecture가 evaluation-contract payload에 포함된다. 따라서 B0 contract SHA와
B2 contract SHA는 의도대로 다르며, B2 metric으로 B5를 승인하는 GO artifact에는
B2-resolved contract SHA를 사용해야 한다.

## Decision

Primary matched-legacy dataset lock을 B2 full-run artifact에서 재확인한 다음 값으로
교정한다.

- split contract: `fab9c136c81081464d9db099656680dac3bf2921a4ae2bbd7605c383b309ab93`
- manifest: `183dda705e76b451dc383a81f517d36df3d6032f00002ab225421b9ae316b9dd`
- image content: `ce06265e6146bcd37692938786386cbd9b844e9742f831284ee5d26aedd15305`
- mask content: `5ae15ab1eff69921168b15811683edab41472456a439b58aa6384c6d472c377e`
- combined content: `a70c6b9467b692a4797976659c6dcd501c80938626226000a6c214efcdec5e42`

B5 GO evidence lock은 다음 B2 fresh-evaluation provenance를 사용한다.

- B2 resolved evaluation contract: `4adfcb3ae550274ed3436c695c872e030c804bb8c16c09025958797312d8d592`
- ordered RELLIS test manifest: `2e078a3ac89d870b4dfb5838f8cc2772e788ecdd7cb011c309d59b4ca6a66918`
- ordered Korean held-out manifest: `1eb86ff65620fb5c0afc1d58c572c517cacc937468ebd865375aaa26d81eb782`

B0 resolved evaluation contract
`096467321246732da9d2f4a31ad8f75626b1aba0500e0680ba4ddd778241635e`는 B0
reference metric provenance로 보존하지만 B2 GO evidence field에 대입하지 않는다.
향후 B5 final evaluation도 resolved B5 model을 포함한 별도 contract SHA를 기록한다.

재감사한 Korean common-supported mIoU로 계산한 `DID02`는
`38.41105305671128 pp`이므로 0013의
`abs_b2_difference_in_differences_ge_10pp` GO trigger를 충족한다. GO artifact는
RunPod output에만 생성하고 repository template은 계속 `NO_GO` placeholder로 둔다.

## Rationale and evidence

교정값은 65자 문자열을 임의로 자른 값이 아니다. B2-E-ADOM full output의
`dataset_contract.json`에 기록된 다섯 field, fresh evaluation의 ordered-manifest
summary, 각 B2 metric summary의 provenance를 직접 읽어 확인했다. B2 checkpoint audit도
PASS이며 E0 SHA는 `c47288019185e18fffdb856d2f47f56936adb06db7579416271ab468b3849f4f`,
E-ADOM SHA는 `b1b9cded88fa091d503fb48c0fd1f9fafd3df938030bb767a04d7a9aab96707b`다.

GO 계산은 B0/B2 Korean held-out을 recipe나 checkpoint 선택에 사용하지 않는다. 두 모델의
RELLIS-val checkpoint가 이미 freeze된 뒤 수행된 direct fresh evaluation 결과를 이용해
B5 실행 필요성만 판단한다.

## Alternatives considered

- 65자 값을 앞/뒤에서 임의로 자르기: raw artifact identity를 증명하지 못해 거절.
- B0 evaluation-contract SHA를 B2 GO evidence로 사용하기: resolved architecture가 다른
  contract를 같은 것으로 표시하므로 거절.
- validator를 우회해 현재 image에서 B5를 직접 실행하기: preregistration과 provenance
  보호를 깨므로 거절.
- Korean held-out metric으로 B5 checkpoint나 threshold를 선택하기: test leakage이므로 거절.

## Consequences

- 새 Git-SHA image에서 static contract와 B5 GO validator를 통과할 수 있다.
- B5 실행은 여전히 exact GPU profile, effective batch 16, active memory probe와 official
  MiT-B5 ImageNet initialization을 요구한다.
- Korean held-out은 GO 여부 외 recipe, early stopping, threshold, checkpoint selection에
  사용하지 않으며 B5 final evaluation 전까지 다시 잠근다.
- B0/B2 checkpoint, dataset, predictions와 metric artifact는 수정하지 않는다.

## Validation and rollback

Unit tests는 모든 corrected identifier가 정확히 64 lowercase hex인지, valid GO payload가
실제 frozen 값으로 통과하는지, dataset drift와 sub-threshold trigger가 계속 거부되는지
검사한다. 새 image에서는 full dataset static contract, RTX 5090 doctor, two-iteration
memory probe를 차례로 실행한다. 어느 단계든 실패하면 B5 training을 시작하지 않고 해당
artifact를 blocker로 보존한다.
