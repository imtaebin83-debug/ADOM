# ADOM 밀리테크 최종 연구발표 스토리보드

> 기본 가정: 발표 12–15분, 질의응답 별도. 시간이 확정되면 슬라이드 수를 조정한다.

## 1. Research framing

### 연구 질문

1. **Data:** 한국형·임무 후보 장면과 여러 공개 off-road source를 추가하면
   RELLIS-only baseline의 희소 클래스 실패를 개선할 수 있는가?
2. **Deployment:** Semantic20 픽셀 결정을 FP16 TensorRT 변환 후에도 보존하면서
   target Jetson에서 bounded latency로 실행할 수 있는가?
3. **System:** semantic perception을 저속 RC Car의 보수적 Go/Stop 판단으로 연결하고,
   입력이나 명령이 소실되면 안전 상태로 복귀시킬 수 있는가?

### 검증할 가설

- H1: TA1/TA2는 continued-training 통제군 TA0보다 target/rare-class IoU 또는 Recall을
  개선하면서 canonical RELLIS 주요 성능을 보존한다.
- H2: PyTorch↔ONNX argmax agreement는 99.9% 이상, ONNX↔TensorRT valid-region
  agreement는 99.0% 이상이다.
- H3: latest-frame 파이프라인은 입력을 누적하지 않고 frame age와 camera-to-command
  p95를 제한하며, watchdog과 manual reset 안전 계약을 만족한다.

### 주장 경계

| 피할 표현 | 발표 표현 |
| --- | --- |
| 데이터 부족이 유일한 근본 원인 | domain representation과 class imbalance를 주요 병목 가설로 둔다 |
| Foundation Model의 성능을 개선했다 | pretrained SegFormer 기반 off-road domain adaptation을 수행했다 |
| TensorRT 양자화 | FP16 TensorRT engine 최적화 |
| 온보드 자율 복귀 구현 | server-free perception과 low-speed safety-stop PoC |
| semantic segmentation이 주행 가능성을 결정 | semantic prior를 이용한 보수적 정지 판단 |
| 30 FPS는 자율주행의 절대 기준 | source FPS, frame age, p95 latency와 control rate를 분리 평가 |

Semantic label만으로 slope, roughness, 침하 가능성을 알 수 없으므로 완전한
traversability나 autonomous return을 주장하지 않는다. Depth, vehicle interaction,
costmap과 Nav2는 후속 연구다.

## 2. 기존 착수 발표에서 바꿀 부분

| 기존 요소 | 최종 발표 처리 |
| --- | --- |
| CNN vs Transformer 대결 | SegFormer 선택 근거만 짧게 제시하고 data ablation을 중심에 둔다 |
| Decoupled head와 Focal Loss | 실제 ablation이 없으면 본문에서 제거한다 |
| SegFormer-B2 + FP16 최적점 예상 | 실제 배포 위험을 고려해 B0를 선택한 pivot과 실측을 제시한다 |
| Depth, Nav2, semantic costmap | 구현 결과가 아니므로 roadmap으로 이동한다 |
| 자동 복귀 | 저속 Go/Stop과 watchdog 범위로 축소한다 |
| Semantic23와 web MLOps | 완료 결과와 분리한 post-presentation roadmap으로 둔다 |
| 2주 노력 강조 | 날짜별 gate, artifact, 대표 오류의 영구 수정으로 보여준다 |

## 3. Recommended slide sequence

### Slide 1. Cold open

동일 고정 장면에서 `E0 실패 → selected TA model 성공 → RC Car STOP`을 20–30초로
보여준다. 아직 영상이 없다면 E0 overlay, TensorRT overlay, 실제 RC Car 사진을
3분할로 사용한다.

발표 시작 문장:

> 저희는 2주 동안 모델 정확도 숫자에서 끝내지 않고, 직접 수집한 데이터가 어떤
> 실패를 고치는지 통제 실험하고, 그 모델이 target Jetson과 실제 차량까지 동일한
> 의미로 전달되는지를 검증했습니다.

### Slide 2. Operational problem and bounded scope

- MUM-T에서 unmanned ground vehicle은 복잡한 비정형 지형과 contested network를
  함께 고려해야 한다.
- 완전 자율주행이 아니라 `ZED RGB → semantic perception → low-speed STOP`을 검증한다.
- 완전한 통신 두절 작전이나 EMP 내성을 구현했다고 말하지 않는다.

### Slide 3. Research gap and data hypothesis

- Off-road dataset은 urban dataset보다 작고 domain/class coverage가 제한적이다.
- RELLIS-3D도 class imbalance와 environmental topography를 난점으로 보고한다.
- dominant grass/tree/sky와 rare pole/log/rubble 등의 support 차이를 실제 ADOM
  baseline 수치로 보여준다.

권장 그림은 class별 train image count를 로그축으로 그린 막대그래프와 같은 class
순서의 IoU 그래프다.

### Slide 4. Three research questions

```text
직접 수집/공개 데이터 → 통제 fine-tuning → ONNX/TRT → latest-frame → Go/Stop
       RQ1                                    RQ2                  RQ3
```

### Slide 5. Two-week engineering sprint

Data, Model/MLOps, Edge, Sensor/Control 네 swimlane으로 날짜별 gate를 표시한다.
commit 수를 성과로 내세우기보다 다음 대표 사례로 연구 밀도를 보여준다.

1. H,W/W,H 혼동으로 640×640 export → config 계약 분리
2. A100 PyTorch와 CPU ONNX Runtime 비교 오해 → same-backend graph parity 분리
3. TensorRT workspace 단위 오류로 attention tactic 탈락 → 2048 MiB build 성공

### Slide 6. Data and ontology pipeline

- Semantic20 train IDs `0..18`, ignore `255`.
- 공개 source package는 RELLIS/RUGD/YCOR를 정규화하며 main val/test는 RELLIS로 유지한다.
- standalone ZED dataset은 현재 문서 계약상 215 labeled frames, sequence-disjoint
  train/val/test `133/21/61`이다.
- source별 보유량과 실제 training sampler draw count를 구분한다.
- Raw RGB → CVAT mask → mapping → ID audit → sequence split → package SHA를 그린다.

### Slide 7. Frozen B0-E0 baseline

| Metric | Result |
| --- | ---: |
| Validation raw mIoU, selected iter 6,000 | `51.07%` |
| Validation fixed-supported mIoU | `58.93%` |
| Canonical test raw mIoU | `43.35%` |
| Canonical test supported11 mIoU | `59.11%` |
| Test aAcc | `89.78%` |

대표 class:

| Class | Test IoU | Test Recall | Interpretation |
| --- | ---: | ---: | --- |
| grass | 83.92 | 96.76 | dominant, strong |
| tree | 76.09 | 94.68 | dominant, strong |
| pole | 0.00 | 0.00 | complete failure |
| log | 40.33 | 63.97 | rare and unstable |
| mud | 44.51 | 47.23 | weak terrain-hazard recall |
| rubble | 53.34 | 54.88 | improvement candidate |

Raw mIoU와 fixed-supported mIoU의 분모가 다른 점을 설명하고 숫자를 섞지 않는다.

### Slide 8. Controlled target-adaptation matrix

| Condition | Initial checkpoint | Additional train data | Question |
| --- | --- | --- | --- |
| Frozen E0 | B0-E0 | none | Original deployment baseline |
| TA0 | Same E0 | RELLIS | Continued-update effect |
| TA1 | Same E0 | RELLIS + standalone | Directly collected data effect |
| TA2 | Same E0 | RELLIS + RUGD + YCOR + standalone | Multi-source diversity effect |

조건은 같은 optimizer-update budget, seed, canonical RELLIS val/test를 사용한다.
TA-final은 checkpoint merge가 아니다. 선택된 recipe를 frozen E0에서 독립 재학습하는
confirmatory run이다.

### Slide 9. Model results

가능하면 3 seeds mean±std를 보고한다. 시간이 부족하면 seed 42 preliminary임을 명시한다.

- OverallSupported mIoU/Recall
- RareRisk-4
- AugmentedRisk-2
- pole/log/barrier/rubble/puddle/mud class IoU·Recall
- absent-class FP pixel/image rate
- standalone held-out target performance
- TA0 대비 delta와 non-degradation

Grouped bar와 class-delta heatmap을 권장한다. 가장 좋은 모델 하나만 보여주지 않는다.

### Slide 10. Deployment semantic preservation

| Gate | Evidence | Result |
| --- | --- | --- |
| PyTorch↔ONNX | 12 frozen images | 100% argmax, max logit error `1.0347e-4` |
| ONNX↔TensorRT | 12 frozen tensors/masks | overall valid `99.9816%` |
| Worst TensorRT reference | 12 images | `99.9521%` |
| Class-area preservation | IDs 0..18 | max `0.0318%p` |

Engine size는 `8.82 MiB`, target-built SHA는
`3e815c93a3f6b63265e449d85174c0a3164f636d5bce8e5fcbb1bc9ba272735d`다.

Parity는 model accuracy나 mIoU가 아니라 deployment semantic preservation이다.

### Slide 11. Edge runtime and latest-frame

현재 file runtime:

| Stage | Mean | p95 |
| --- | ---: | ---: |
| H2D | 0.76 ms | 0.99 ms |
| TensorRT engine | 14.23 ms | 17.02 ms |
| D2H | 3.89 ms | 5.32 ms |
| Runtime total | 18.90 ms | 22.11 ms |

평균 reciprocal `52.92 FPS`는 camera/ROS/resize/pad를 제외한 file tensor 결과다.
HD720 30/60 latest-frame에서는 source/received/processed/overwritten frames, frame age,
각 stage와 end-to-end p50/p95/p99, power/temperature를 보고한다.

### Slide 12. RC Car and safety architecture

- 실제 차량 사진에 ZED 2i, Jetson, PCA9685, ESC/servo, battery를 callout한다.
- 확정된 ROS topic만 architecture에 넣는다.
- Low speed ≤0.3 m/s, Keep Last 1/latest-frame, 0.25초 command timeout, STOP 후
  manual reset, physical E-stop 담당자를 표시한다.
- wheels-off, process-kill, camera-loss 결과가 있을 때만 PASS로 표시한다.

### Slide 13. End-to-end A/B result

같은 camera pose, light, target distance와 frozen ROI/threshold에서 E0와 TA-final을
비교한다.

필수 표:

- target trials와 성공률
- negative trials와 false-stop rate
- perception-to-command p95
- stop distance
- watchdog/neutral 결과

편집한 성공 영상과 함께 전체 무편집 trial을 appendix 또는 QR로 제공한다.

### Slide 14. Limitations, roadmap, conclusion

한계:

- small standalone dataset와 제한된 지역·계절
- legacy E0는 single seed, deterministic=false
- semantic class만으로 geometry-aware traversability를 보장하지 못함
- live-camera와 repeated vehicle trial은 실측 완료 전 claim 불가

연구 우선순위:

1. TA 3-seed confirmatory result
2. sequence-held-out field test와 negative-scene false-stop
3. geometry/depth 기반 traversability
4. Semantic23 ontology/provenance
5. deployment/CVAT/W&B web automation

결론은 세 문장으로 제한한다.

1. 직접 수집 데이터의 효과를 continued-training과 분리하는 실험을 설계했다.
2. ONNX→TensorRT에서 valid-pixel 결정의 `99.9816%`를 보존했다.
3. Server-free perception을 실제 RC Car의 보수적 safety action으로 연결하고 있다.

## 4. Likely questions

### Foundation Model 연구인가?

공식 과제명은 유지하지만 현재 결과는 SegFormer-B0 기반이다. 엄밀히는 범용 VFM을
새로 제안한 연구가 아니라 pretrained transformer segmentation model의 domain
adaptation과 deployment validation이다. VFM 확장은 future work다.

### 데이터 부족이 정말 원인인가?

유일한 근본 원인이라고 주장하지 않는다. RELLIS-3D가 class imbalance와 topography
challenge를 보고하고 cross-dataset 연구가 multi-source generalization 향상을
보고한 점에 근거해 검증 가능한 가설로 설정했다. TA0가 단순 continued training을
통제한다.

### Semantic segmentation으로 주행 가능성을 판단할 수 있는가?

완전히는 불가능하다. 같은 grass나 mud도 경사·거칠기·함몰에 따라 주행성이 다르다.
이번 연구는 semantic prior와 저속 safety stop PoC이며 geometry-aware traversability는
후속 연구다.

### 왜 B2가 아니라 B0인가?

짧은 integration 기간에 target-device risk와 latency를 우선해 학습·평가된 B0를
배포 baseline으로 선택했다. B0 engine은 target Jetson에서 parity와 file runtime
gate를 통과했다. B2는 별도 연구 비교군이다.

### 52.92 FPS이면 60 FPS camera도 충분한가?

아직 알 수 없다. 이 값은 frozen tensor file runtime reciprocal이다. HD720 30/60
latest-frame에서 camera/ROS/preprocess를 포함한 frame age와 p95를 측정한 뒤 판단한다.

### Single seed를 일반화할 수 있는가?

없다. E0는 audited legacy baseline이다. Confirmatory claim은 TA 3-seed mean±std와
sequence-held-out test를 통과한 뒤에만 한다. 그 전 결과는 preliminary PoC다.

## 5. Primary references

- DARPA RACER, unstructured off-road UGV autonomy and iterative field testing:
  <https://www.darpa.mil/research/programs/robotic-autonomy-in-complex-environments-with-resiliency>
- U.S. Army MUM-T contested network experiment:
  <https://www.army.mil/article/224497/army_assesses_network_needs_for_manned_unmanned_teaming>
- RELLIS-3D dataset, class imbalance and environmental topography:
  <https://arxiv.org/abs/2011.12954>
- RUGD unstructured outdoor dataset: <http://rugd.vision/>
- Cross-dataset off-road semantic segmentation evaluation:
  <https://doi.org/10.1016/j.eswa.2026.132656>
- SegFormer, NeurIPS 2021:
  <https://proceedings.neurips.cc/paper_files/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html>
- STONE, geometry-aware scalable off-road traversability:
  <https://arxiv.org/abs/2603.09175>
