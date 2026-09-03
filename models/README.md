# Models

체크포인트와 export 산출물의 로컬 배치 규칙을 기록합니다.

실제 모델 파일은 git에 커밋하지 않습니다.

권장 로컬 구조:

```text
models/
├── checkpoints/
│   ├── b0-e0/        # Semantic20 B0-E0 배포 프로파일
│   └── eadom/        # Semantic20 B0-E-ADOM 배포 프로파일
├── exports/
│   ├── onnx/
│   └── tensorrt/
└── cards/
```

## 배포 프로파일 계약

`scripts/run_jetson_t4.sh`가 쓰는 프로파일 디렉터리에는 **정확히 한 개의 `.pth`만** 둔다.
실행 전에 파일 수와 SHA-256을 검증하며, 다른 위치를 쓰려면 `ADOM_CHECKPOINT`에 절대경로를
지정한다. 의도적인 신규 artifact는 `ADOM_EXPECTED_CHECKPOINT_SHA256`을 함께 명시해야
통과한다. RunPod에서 Jetson까지의 전달 절차는
[`docs/setup-guides/jetson-model-checkpoint-handoff.md`](../docs/setup-guides/jetson-model-checkpoint-handoff.md)를
따른다.

모델을 공유해야 할 때는 model card에 다음을 기록합니다.

- model architecture
- training dataset and split
- checkpoint source
- metrics
- export command
- known limitations
