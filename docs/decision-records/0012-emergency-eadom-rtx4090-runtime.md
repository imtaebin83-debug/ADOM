# 0012. Run emergency E-ADOM on RTX 4090

- Status: Accepted
- Date: 2026-08-13
- Owners: 태빈 및 perception training 담당자
- Amends: 0011의 긴급 실행 hardware 계약

## Context

E-ADOM full training 직전 A100 재고가 소진됐다. 현재 immutable image의 PyTorch 2.1과
CUDA 12.2는 RTX PRO 6000/RTX 5090 Blackwell을 지원하지 않지만 RTX 4090 Ada는
지원한다. 기존 cycle은 모든 실험에서 A100 이름과 최소 75GiB를 하드코딩해 호환 가능한
RTX 4090에서도 학습 전에 runtime doctor가 실패했다.

## Decision

E-ADOM wrapper는 runtime doctor에 GPU name substring `RTX 4090`과 최소 physical memory
22GiB를 명시한다. cycle은 이 두 값을 명시적으로 받을 수 있게 하되 기본값은 A100/75GiB로
유지한다. E-ADOM의 micro-batch 16, effective batch 16, seed 42, Stage 1 4,000 및 Stage 2
40,000 optimizer update는 변경하지 않는다.

## Consequences

- 기존 E0/E1/E2/TA cycle의 A100 runtime contract는 변하지 않는다.
- RTX 4090 OOM 시 implicit fallback하지 않고 실패시킨 뒤, 별도 승인된 micro-batch 8 /
  accumulation 2 재실행으로 effective batch 16을 보존한다.
- RTX 4090은 ECC가 없으므로 finite-loss, checkpoint SHA와 validation artifact 감사를
  그대로 유지하며 B0-E0 rollback package를 보존한다.
