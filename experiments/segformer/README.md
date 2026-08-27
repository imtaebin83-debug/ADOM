# SegFormer Experiments

SegFormer-B0/B2 기반 비교와, B2 evidence에 조건부인 B5 사전등록을 관리합니다.

초기 목표:

- pretrained checkpoint와 config 출처 기록
- RELLIS-3D class ontology에 맞는 head 구성
- B0/B2 accuracy-runtime-power trade-off 비교
- TensorRT export 가능성을 고려한 모델 변형 최소화

계획 문서:

- [B2-E-ADOM capacity × domain study](b2-eadom-capacity-domain-study.md)
- [B2-E-ADOM seed 42 run record](b2-eadom-capacity-domain-seed42-run.md)
- [B5-E0/B5-E-ADOM preregistration](b5-eadom-capacity-domain-preregistration.md)
- [B5 GO/NO-GO artifact template](b5-go-decision.template.json)
