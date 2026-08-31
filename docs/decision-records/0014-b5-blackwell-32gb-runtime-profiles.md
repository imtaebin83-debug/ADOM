# 0014. Add exact Blackwell 32GB runtime profiles for conditional B5 runs

- Status: Accepted
- Date: 2026-09-01
- Owners: perception training/research 담당자
- Supersedes: none; extends 0013

## Context

B5 capacity × domain 사전등록은 A100 40/80GB, RTX A6000 48GB와 RTX PRO 6000
Blackwell 96GB만 exact profile로 허용했다. 실제 공급 가능한 GPU는 RTX PRO 4500
Blackwell 32GB와 GeForce RTX 5090 32GB이며 A100과 RTX 4090은 재고가 없다. RTX PRO
4500 Server Edition은 16GB MIG slice도 제공하므로 제품명 substring만으로는 전체
32GB 장치를 증명하지 못한다. 두 Blackwell 제품은 compute capability 12.0이지만 현재
immutable training image의 PyTorch는 native `sm_120` binary를 포함하지 않는다.

## Decision

`rtx-pro-4500-blackwell-32gb`와 `rtx-5090-32gb`를 별도 B5 runtime profile로
추가한다. runtime doctor는 exact family name, 29–34GiB full-device range와 compute
capability 12.0을 함께 검사한다. PRO 4500의 16GB MIG slice, RTX 5090/PRO 4500 간
상호 대체, 다른 compute capability는 fail closed한다.

두 profile의 memory-probe order는 `8/2 → 4/4 → 2/8 → 1/16` proposal이며 effective
batch는 항상 16이다. current PyTorch build에 native `sm_120`이 없으면 doctor는
PTX-JIT compatibility를 provisional warning으로 남긴다. 이 warning은 active B5
forward/backward와 2 runner-iteration memory probe를 생략할 권한을 주지 않는다.

## Rationale and evidence

NVIDIA는 RTX PRO 4500 Blackwell Workstation/Server Edition과 GeForce RTX 5090을
각각 32GB 제품으로 명시한다. 실제 RunPod RTX PRO 4500은 32,623MiB와 compute
capability 12.0으로 확인됐고 B5 512×512 forward/backward는 PTX JIT 경로에서
통과했다. name, memory와 capability를 함께 잠그면 MIG slice와 잘못된 제품 선택을
기계적으로 거부하면서 현재 공급 가능한 hardware를 조건부로 사용할 수 있다.

## Alternatives considered

- 두 제품을 하나의 `blackwell-32gb` profile로 합치기: 제품별 성능·provenance가
  사라져 거절.
- VRAM만 32GB 이상 검사하기: PRO 4500/5090 식별과 compute architecture 검증을
  하지 못해 거절.
- native `sm_120` 부재를 숨기고 정상 지원으로 표시하기: PTX JIT startup과 kernel
  compatibility 위험을 감춰 거절.
- 즉시 CUDA/PyTorch base를 변경하기: 기존 B2/B5 non-architecture fingerprint를
  바꾸므로 별도 migration과 B2 재검증 없이 수행하지 않는다.

## Consequences

- 두 32GB profile은 probe 전 capacity나 실행시간을 보장하지 않는다.
- native `sm_120`이 없는 image에서는 process startup마다 PTX JIT 비용이 발생할 수
  있다.
- profile 추가는 B2 uncertainty GO gate, dataset provenance repair, RELLIS-val-only
  selection과 Korean test-only 규칙을 완화하지 않는다.
- GPU별 provider, exact name, observed VRAM, driver, image SHA와 probe 결과를 run
  metadata에 기록한다.

## Validation and rollback

unit tests는 두 exact name, 32GB full-device range, compute capability 12.0, 16GB MIG,
wrong-product rejection과 native-arch/PTX warning을 검사한다. 실제 Pod에서는 doctor와
B5 active memory probe를 실행한다. 검증 실패 시 해당 profile을 사용하지 않으며 기존
0013 profile과 B0/B2 artifact는 변경하지 않는다.
