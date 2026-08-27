# 0013. Gate B5 capacity × domain study on unresolved B2 evidence

- Status: Accepted
- Date: 2026-08-27
- Owners: perception training/research 담당자
- Supersedes: none; extends 0005, 0011, and the B2 preregistration

## Context

B0-E0의 Korean field failure가 capacity, domain supervision 또는 상호작용 때문인지
분리하기 위해 B2-E-ADOM 실험이 진행 중이다. B2 결과가 나오기 전에 B5를 추가하면
compute만 늘리고 post-hoc hypothesis를 만들 위험이 있다. 또한 A100 40/80GB,
RTX A6000 48GB, RTX PRO 6000 Blackwell 96GB가 모호한 이름 하나로 취급되고 있었다.

## Decision

B5-E0와 B5-E-ADOM config를 B2 counterpart의 architecture-only extension으로
사전등록한다. official MMSegmentation MiT-B5 ImageNet initialization을 사용하고
B0/B2 experiment checkpoint warm-start를 금지한다. primary는 기존 4,568-row
matched-legacy split이며 4,556 clean sensitivity와 분리한다. Korean held-out 61장은
checkpoint freeze 뒤 test-only다.

B5의 모든 runtime gate는 frozen B2 evaluation provenance와 사전등록한 10 pp
capacity/interaction 또는 class-discordance trigger를 증명하는 별도 GO artifact를
요구한다. GPU는 A100-40GB, A100-80GB, RTX-A6000-48GB,
RTX-PRO-6000-Blackwell-96GB exact profile 중 하나를 runtime doctor에서 name과 VRAM으로
검증한다. profile별 micro-batch order는 memory probe 전 proposal이며 effective batch는
항상 16을 유지한다.

## Rationale and evidence

B2와 B5는 embed/decoder width가 같고 official config의 주요 차이가 MiT depth와
pretrained checkpoint이므로 세 경로 allowlist로 capacity change를 고립할 수 있다.
B5 실행을 B2 uncertainty에 조건화하면 독립 Korean sequence/negative annotation보다
정보 가치가 낮은 compute를 피할 수 있다. exact hardware profile은 잘못된 VRAM 가정과
Blackwell/Ampere image compatibility 혼동을 fail closed한다.

## Alternatives considered

- B2 결과와 무관하게 B5 즉시 실행: 연구 질문을 좁히지 못해 거절.
- “A100” 또는 “PRO A6000” substring만 검사: 40/80GB 및 제품 세대를 구분하지 못해 거절.
- B2-E-ADOM 또는 B5-E0 checkpoint warm-start: data/capacity 효과와 curriculum을 섞어 거절.
- clean 4,556 split을 primary로 교체: 기존 B0/B2 matched comparison을 깨므로 sensitivity로 유지.

## Consequences

- 현재 B2 full run은 변경하거나 중단하지 않는다.
- B5 config와 static tests가 있어도 valid GO artifact 없이는 B5 probe가 실행되지 않는다.
- historical B0/B2 E0와 E-ADOM의 checkpoint-selection provenance 차이는 숨기지 않고
  exploratory limitation으로 보고한다.
- 시간·비용 숫자는 각 exact profile의 probe/mini 측정 뒤에만 기록한다.
- B2 run record의 dataset/evaluation identifier 일부가 65 hex로 확인돼 SHA-256 lock으로
  사용할 수 없다. 원본 artifact 재감사와 새 provenance amendment 전까지 B5는 blocked다.

## Validation and rollback

architecture allowlist, non-architecture fingerprint, split/manifest/mapping/content digest,
freeze/update hook, Stage 2 handoff, RELLIS-val selection, canonical test lock, GPU profile과
effective-batch parity를 unit/static contract로 검사한다. 계약이 실패하면 B5를 실행하지
않으며 B0/B2 artifact와 기존 D-5 B0 배포 계약은 그대로 유지한다.
