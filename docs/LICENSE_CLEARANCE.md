# AVTR-1 라이선스 정리 현황

- 기준일: 2026-09-11
- 대상 커밋: `eb1e5a8` (https://github.com/avaturn-live/avtr-1, main)
- 전제: 고객사 현장 PC 1대에 설치하는 **상업 배포**. 라이선스 취득 주체의 연매출(계열사 합산)은 1,000만 달러 미만으로 가정.
- 근거는 모두 이 저장소 안의 라이선스 파일과 공개 업스트림 라이선스이며, 법률 자문이 아닙니다. 최종 판단 전 법무 검토가 필요합니다.

## 1. 판정 요약

| 판정 | 의미 |
| --- | --- |
| **해결** | 아래 "공통 의무" 를 지키면 추가 계약 없이 상업 사용 가능 |
| **조건부 해결** | 상업 사용 가능하나 수치 기준·금지 조항 등 별도 확인 항목이 있음 |
| **미해결** | 현 상태로는 상업 사용 불가. 계약 또는 대체 필요 |

## 2. 해결 · 조건부 해결 구성요소

### 2.1 모델 및 번들 가중치

| 구성요소 | 파일 | 라이선스 | 판정 | 비고 |
| --- | --- | --- | --- | --- |
| AVTR-1 모션 생성기 가중치 | `build_artifacts/avtr1.scripted.pt`, `avtr1_normalizer.safetensors` | AVTR-1 Community License (`LICENSE-MODEL.md`) | **조건부 해결** | 연매출 1,000만 달러 미만(계열사 합산)이면 로열티 없음. Attachment A 사용 제한, 경쟁 제품 금지(A-20), 파생물 배포 조건(3장) 적용 |
| 빌드·데모 스크립트 | `scripts/` | AVTR-1 Community License | **조건부 해결** | 위와 동일 조건 |
| HuBERT 음성 인코더 | `build_artifacts/hubert-lbs-avtr1.onnx` | Apache-2.0 (Meta 원본, Goodsize 파인튜닝) | **해결** | AVTR-1 입력 특징을 만드는 핵심 부품이며 Apache-2.0 이므로 그대로 유지 가능. `THIRD-PARTY-NOTICES.md` 근거 |
| MODNet 매팅 | `build_artifacts/modnet.onnx` | Apache-2.0 | **해결** | 배경 분리(알파) 처리에 재사용 가능 |
| grid_sample 3D TensorRT 플러그인 | `renderer_runtime_artifacts/libgrid_sample_3d_plugin.so` | Apache-2.0 | **해결** | 워프 네트워크용 |

### 2.2 LivePortrait 계열 ONNX (ditto-talkinghead 에서 내려받음)

| 구성요소 | 파일 | 라이선스 | 판정 | 비고 |
| --- | --- | --- | --- | --- |
| 외형 추출기 | `appearance_extractor.onnx` | LivePortrait MIT, ONNX 재패키징 Apache-2.0 | **해결** | 저작권 고지 유지 필요 |
| 모션 추출기 | `motion_extractor.onnx` | 동일 | **해결** | |
| 203점 랜드마크 | `landmark203.onnx` | 동일 (LivePortrait 자체 모델) | **해결** | InsightFace 와 무관 |
| 워프 네트워크 | `warp_network.onnx`, `warp_network_ori.onnx` | 동일 | **해결** | |
| 스티치 네트워크 | `stitch_network.onnx` | 동일 | **해결** | |
| SPADE 디코더 | `decoder.onnx` | 동일 | **해결** | |

### 2.3 얼굴 검출·랜드마크 (이번 작업으로 대체)

| 구성요소 | 파일 | 라이선스 | 판정 | 비고 |
| --- | --- | --- | --- | --- |
| MediaPipe BlazeFace (short-range) | `blaze_face.onnx` | Apache-2.0 (Google, 모델 카드 명시) | **해결** | 아티팩트 설정에 이미 포함되어 있었으나 코드에서 미사용이었음. 이번 작업으로 검출기로 채택 |
| MediaPipe Face Mesh V2 (478점) | `face_mesh.onnx` | Apache-2.0 (Google, 모델 카드 명시) | **해결** | 106점 랜드마크 대체. 아바타 등록 시 1회만 실행 |

### 2.4 스트리밍 · 런타임 라이브러리

| 구성요소 | 라이선스 | 판정 | 비고 |
| --- | --- | --- | --- |
| aiortc | BSD-3-Clause | **해결** | 현재 Streamer 가 이미 사용 중. 자체 Streamer 구현 시에도 사용 가능 |
| PyTorch | BSD-3-Clause | **해결** | |
| onnxruntime-gpu | MIT | **해결** | |
| OpenCV, kornia, scikit-image, numpy, FastAPI, uvicorn, PyAV 등 | Apache-2.0 / BSD / MIT | **해결** | 고지 유지 |
| TensorRT (pip wheel) | NVIDIA TensorRT 라이선스 | **확인 필요** | 고객사 PC 재배포 조건은 NVIDIA 약관 검토 필요. 이 문서 범위 밖 |

## 3. 미해결 구성요소

| 구성요소 | 경로 | 라이선스 | 문제 | 선택지 |
| --- | --- | --- | --- | --- |
| Avaturn Renderer (추론 파이프라인 코드) | `src/avtr1_renderer/` | PolyForm Noncommercial 1.0.0 (`LICENSE-RENDERER.md`) | 매출과 무관하게 상업 사용 불가. 단, 폴더 내 모든 `.py` 헤더는 `LicenseRef-AVTR-1-Community` 로 표기되어 폴더 라이선스와 불일치 | (a) Renderer Commercial License 계약, (b) 제작사에 헤더 기준 적용 여부 서면 확인, (c) LivePortrait 원본 기반 자체 재구현 |
| Avaturn Streamer (세션·동기화 백엔드) | `src/avaturn_live_streamer/` | PolyForm Noncommercial 1.0.0 + 특허 유보 (`LICENSE-STREAMER.md`, `PATENTS.md`) | 상업 사용 불가. **독자 재구현·클린룸 구현에도 특허 실시권이 없음** (PATENTS.md 3-5) | (a) Streamer Commercial License 계약, (b) 특허 청구범위 검토 후 회피 설계. 현장 PC 로컬 출력으로 단순화해도 특허 검토는 별도 필요 |
| InsightFace SCRFD 검출기 | `insightface_det.onnx` | 코드 MIT, 모델 비상업 연구 전용 | 상업 사용 불가 | **이번 작업으로 제거, MediaPipe BlazeFace 로 대체** |
| InsightFace 2D106 랜드마크 | `landmark106.onnx` | 동일 | 상업 사용 불가 | **이번 작업으로 제거, MediaPipe Face Mesh 로 대체** |

주의: 모델 호출 코드(`avtr1_motion_generator.py`, `models/avtr1.py`, `components/hubert.py`)는 Renderer 폴더 안에 있습니다. "모델 단독 무료 사용" 은 이 호출 코드를 자체 작성하거나 제작사 확인을 받는 것을 전제로 합니다.

## 4. 공통 의무 (AVTR-1 Community License)

고객사 PC 설치는 제3자 배포에 해당하므로 `LICENSE-MODEL.md` 3장이 그대로 적용됩니다.

- [ ] 고객사와의 계약에 Attachment A 사용 제한을 구속력 있는 조항으로 포함 (3-a)
- [ ] 고객사에 `LICENSE-MODEL.md` 전문 사본 제공 (3-b)
- [ ] 수정한 파일마다 변경 사실을 명시하는 고지 추가 (3-c). 이번 작업에서 수정한 파일은 헤더에 `Modified` 줄을 추가함
- [ ] 저작권·특허·상표·귀속 고지 유지 (3-d). `THIRD-PARTY-NOTICES.md` 갱신 포함
- [ ] 아바타 출력물이 기계 생성임을 사용자에게 명확히 고지하는 UI/문구 (Attachment A-5)
- [ ] 제작사(Avaturn.Live)의 상업 제품과 직접 경쟁하는 용도가 아님을 확인 (Attachment A-20)
- [ ] 연매출 산정 시 자회사·계열사·공동 지배 회사 합산 (1장 Entity 정의)
- [ ] Apache-2.0 구성요소(HuBERT, MODNet, MediaPipe 모델, ditto ONNX)의 라이선스 사본·고지 동봉
- [ ] LivePortrait(MIT) 저작권 고지 동봉

## 5. 작업 단계

| 단계 | 내용 | 상태 |
| --- | --- | --- |
| 1 | InsightFace 검출기·106점 랜드마크를 MediaPipe BlazeFace + Face Mesh 로 대체. 아티팩트 목록에서 InsightFace 모델 제거. 고지 문서 갱신 | **완료 (2026-09-11, 미커밋)**. 검증 결과는 6장 |
| 2 | Renderer: 모델 단독 견적과 모델+Renderer 견적을 hello@avaturn.me 에 동시 요청. 외부 렌더러 연결 허용, 모델 호출 코드 사용권, 헤더 불일치 해석을 질의 | **초안 완료** `docs/AVATURN_INQUIRY_DRAFT.md` (2026-09-14). 회사 정보 기입 후 발송 필요 |
| 3 | Streamer: 특허 청구범위 검토. 계약 또는 회피 설계 결정 | **공개 조사 완료** `docs/PATENT_REVIEW.md`. Streamer 출원은 미공개라 청구항 확인 불가. 회신 후 변리사 FTO 필요 |
| 4 | (2·3 결과에 따라) LivePortrait 원본 기반 자체 렌더러, 로컬 재생기 구현 | **라이선스 무관 부분 구현 완료** (2026-09-14): `src/lpavatar/` 에 등록 캐스케이드·키포인트 변환·stitch/warp/decoder 래퍼·페이스트백·매팅·픽셀 포맷·런타임 래퍼 구현, CPU 검증 완료(자기 재구성 PSNR 31.8 dB). 모델 호출기(R1)와 실시간 최적화(R6)는 견적 회신·GPU 확보 후. 상세는 `docs/REIMPLEMENTATION_PLAN.md` 5장 |

## 6. 1단계 변경 내역과 검증 결과

### 변경 파일

| 구분 | 파일 | 내용 |
| --- | --- | --- |
| 추가 | `src/avtr1_renderer/components/mediapipe_face.py` | BlazeFace 전·후처리(앵커 896개, 박스 디코드, 가중 NMS, 레터박스), Face Mesh ROI 구성·478점 역투영. ditto-talkinghead(Apache-2.0) 코드 기반 |
| 추가 | `src/avtr1_renderer/models/blaze_face.py`, `models/face_mesh.py` | ONNX 입출력 계약 |
| 추가 | `tests/test_mediapipe_face.py` | CPU 단위 테스트 11개 |
| 수정 | `avatar_loader.py` | 등록 캐스케이드: BlazeFace → Face Mesh(478) → 224 크롭 → landmark203 → 512 크롭 |
| 수정 | `components/source_crop.py` | 478점용 눈·입 중심 축 파서 추가 |
| 수정 | `components/face_landmarks.py` | landmark106 제거, scikit-image 의존 제거 |
| 수정 | `pipeline.py`, `artifact_configs/v1.py`, `models/__init__.py` | 아티팩트 키 교체, InsightFace 다운로드 목록 삭제 |
| 삭제 | `components/face_detection.py`, `models/face_detection.py`, `models/landmark106.py` | InsightFace 래퍼·계약 |
| 수정 | `THIRD-PARTY-NOTICES.md`, `README.md` | MediaPipe·ditto 고지 추가, InsightFace 제거 명시 |

수정한 Goodsize 파일에는 모두 헤더에 `Modified 2026-09-11:` 줄을 추가했습니다 (`LICENSE-MODEL.md` 3-c).

### 구 경로 대비 정합성 (CPU, ONNX Runtime)

두 경로는 모두 LivePortrait landmark203 → 512 크롭으로 끝나므로, 렌더러가 실제로 소비하는 203점 랜드마크와 512 크롭 아핀을 비교했습니다. 테스트 이미지는 ditto-talkinghead 예제 초상(1432×1432).

| 케이스 | 203점 평균 차이 (px) | 512 크롭 평행이동 차이 (px) | 512 크롭 픽셀 MAE (/255) |
| --- | --- | --- | --- |
| 정방형 uint8 | 0.96 | 3.6 | 4.0 |
| 정방형 float64 (로더 입력 형식) | 0.90 | 3.4 | 3.8 |
| 1280×720 캔버스에 축소 배치 | 0.39 | 1.1 | 2.6 |
| 20° 기울인 초상 | 0.92 | 3.1 | 3.2 |

얼굴 폭 약 520 px 기준으로 203점 차이는 0.2 % 수준입니다. 시드 단계(224 크롭)는 478점이 이마까지 포함해 약 8 % 더 넓게 잡히지만, landmark203 이 이를 흡수하여 최종 결과는 동일하게 수렴합니다.

### 이 환경에서 확인하지 못한 것

- GPU·TensorRT 환경에서의 `AvatarLoader` 실제 실행 (로컬 PC 에 CUDA 없음). 코드 경로는 기존 ONNX Runtime CUDA 엔진 래퍼를 그대로 사용하므로 `pixi run download` 후 `pixi run generate_offline` 로 확인 필요
- 실제 Avaturn 기준 아바타(HF 게이트 저장소)에 대한 재검증

## 7. 근거 파일

- `LICENSE.md` — 구성요소별 라이선스 지도, 건별 협상 문구
- `LICENSE-MODEL.md` — 매출 기준(2장), 배포 조건(3장), Attachment A
- `LICENSE-RENDERER.md` — `src/avtr1_renderer/` 전체 비상업
- `LICENSE-STREAMER.md`, `PATENTS.md` — `src/avaturn_live_streamer/` 비상업, 독자 구현 특허 실시권 불인정
- `THIRD-PARTY-NOTICES.md` — HuBERT·MODNet·플러그인 Apache-2.0, LivePortrait MIT, InsightFace 비상업
- MediaPipe 모델 카드: BlazeFace (Short Range), Face Mesh V2 — 모두 Apache-2.0 링크 명시
  - https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf
  - https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf
- ditto-talkinghead (ONNX 재패키징 출처): https://github.com/antgroup/ditto-talkinghead — Apache-2.0
- aiortc: https://github.com/aiortc/aiortc/blob/main/LICENSE — BSD-3-Clause
- LivePortrait: https://github.com/KwaiVGI/LivePortrait/blob/main/LICENSE — MIT
