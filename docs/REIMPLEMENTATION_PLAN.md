# 자체 구현 범위·공수 산정 (Renderer / Streamer 대체안)

- 작성일: 2026-09-14
- 목적: Avaturn 견적(`docs/AVATURN_INQUIRY_DRAFT.md`)과 비교할 기준. 계약이 유리하면 이 문서의 구현은 하지 않습니다
- 전제: AVTR-1 가중치 + HuBERT(Apache-2.0) + MODNet(Apache-2.0) + LivePortrait ONNX(MIT) + MediaPipe(Apache-2.0)는 그대로 사용. 고객사 현장 PC 1대, 로컬 화면 출력, 외부 WebRTC 스트리밍 없음
- 공수는 공개 코드 구조를 바탕으로 한 추정치입니다. 실제 성능 검증 결과가 아니며 GPU 실측 후 재산정이 필요합니다

## 1. 현재 코드 자산 분류

`src/avtr1_renderer/` 5,427행, `src/avaturn_live_streamer/` 4,625행, `scripts/` 2,172행 (2026-09-14 기준, 1단계 변경 반영).

| 분류 | 파일 | 행수 | 라이선스 | 자체 구현 시 처리 |
| --- | --- | --- | --- | --- |
| **A. 그대로 사용 가능** | `scripts/build_avtr1_engines.py` | 536 | Community | 모델 TensorRT 엔진 빌드. 인코드/디코드 I/O 계약(입력 `x, kp_tokens, past_context, past_last, audio_self, audio_other, t, w_self, w_other, w_kp`, 출력 `output`)이 문서화되어 있어 호출 코드 재작성의 기준이 됨 |
| | `scripts/build_hubert_engine.py`, `build_renderer_engines.py`, `download_artifacts.py` | 953 | Community | 엔진 빌드·다운로드. ONNX 수술(surgery) 로직 포함 |
| | `scripts/generate_offline.py` | 199 | Community | 오프라인 검증용 참조 |
| | `components/mediapipe_face.py`, `models/blaze_face.py`, `models/face_mesh.py` | ~450 | Apache-2.0 (1단계 작성) | 그대로 사용 |
| **B. 업스트림(MIT/Apache)에서 재도출 가능** | `components/source_crop.py` | 242 | 폴더 라이선스 대상 | LivePortrait `crop.py` 와 동일 수학. 원본 기준으로 재작성 |
| | `components/liveportrait/motion_stitch.py`, `motion_extractor.py` | 286 | 〃 | LivePortrait 키포인트 변환 `x_d = s·((x_c+δ)@R)+t` 와 stitching. 원본 기준 재작성 |
| | `components/putback.py`, `components/matting.py`, `backgrounds.py` | ~200 | 〃 | 페이스트백·MODNet 매팅·배경 합성. ditto-talkinghead(Apache-2.0)에 동등 구현 존재 |
| | `runtime/onnxrt.py`, `runtime/trt.py`, `runtime/loader.py` | 520 | 〃 | 범용 IOBinding/TensorRT 래퍼. 일반적 패턴이라 자체 작성 부담 낮음 |
| | `artifact_manage_basic.py`, `avtr1_artifact_manager.py`, `artifact_configs/` | 416 | 〃 | HF 다운로드·경로 관리. 단순 유틸 |
| **C. Goodsize 고유 로직 (재작성 핵심)** | `avtr1_motion_generator.py`, `models/avtr1.py` | 629 | 〃 (헤더는 Community) | HuBERT 3+5+5 윈도우, 75프레임 오디오 이력, z-score 정규화 상태, AR(1) 상관 노이즈, Euler ODE, CFG 가중치. **가장 중요한 재작성 대상**. 2.1 질의 결과에 따라 재사용 가능할 수도 있음 |
| | `intro_motion.py`, `constants.py` | 220 | 〃 | 인트로 모션 재생, 39개 립싱크 좌표 인덱스. `LIPSYNC_COORDS` 는 normalizer safetensors 에도 들어 있어 데이터로 취득 가능 |
| | `avatar_loader.py` | 433 | 〃 | 등록 캐스케이드. 1단계에서 MediaPipe 로 교체한 부분 + LivePortrait 추출기 호출 + 마스크 사전 워프 |
| | `renderer.py`, `pipeline.py`, `frame_sink.py`, `pixel_format.py`, `types.py` | 846 | 〃 | 청크 오케스트레이션, 5프레임 배치 워프 후 프레임별 디코드, 픽셀 포맷 변환 |
| | `api/` | 396 | 〃 | HTTP API. 로컬 배포에서는 불필요 |
| **D. Streamer** | `src/avaturn_live_streamer/` 전체 | 4,625 | PolyForm NC + 특허 | 사용 불가. 로컬 재생용으로 **축소 재설계** (아래 3장). 코드 참조도 피할 것 |

## 2. 자체 렌더러 (Renderer 대체) 설계

```
portrait.png ──MediaPipe det/mesh──► 224 crop ──lm203──► 512 crop
             ──appearance/motion extractor(LivePortrait)──► f_s, KPInfo
audio(16k) ──HuBERT(Apache)──► 25Hz feat ──AVTR-1 enc/dec(TRT)──► R, exp (5 frames)
R, exp ──stitch(LivePortrait)──► x_s, x_d ──warp──► decoder ──► 512 RGB
512 RGB ──pasteback(M_c2o)──► 720p ──MODNet──► alpha ──► composite/bg
```

### 2.1 재작성 항목과 공수

| 항목 | 내용 | 근거 자료 | 추정 공수 |
| --- | --- | --- | --- |
| R1. 모델 호출기 | 오디오 윈도우·이력 상태, 정규화/역정규화, ODE 스텝, CFG 가중치, 노이즈 스케줄 | `build_avtr1_engines.py` 의 I/O 계약과 normalizer 사이드카(Community). 하이퍼파라미터(chunk 5, past 75, future 5, ODE 스텝 수, 노이즈 상관 계수)는 실측으로 재확인 필요 | 3–4 주 |
| R2. LivePortrait 렌더 스테이지 | stitch → warp → SPADE decoder, 5프레임 배치 | LivePortrait 원본(MIT), ditto ONNX 그래프(Apache-2.0), `build_renderer_engines.py` (Community) | 2 주 |
| R3. 등록 캐스케이드 | MediaPipe(완료) + lm203 + 512 크롭 + 추출기 + 마스크 워프 | LivePortrait `cropper.py`, 1단계 코드 | 1 주 |
| R4. 합성·출력 | 페이스트백, MODNet 매팅, 배경, 픽셀 포맷 | ditto `putback`, MODNet 공식 코드 | 1 주 |
| R5. 런타임 래퍼 | TensorRT/ORT IOBinding, 스트림 처리 | 일반 패턴 | 1 주 |
| R6. 실시간 최적화 | 25 fps 유지(RTX 4060 Ti 급에서 청크당 ≤200 ms), 스트림 중첩, 메모리 고정 | README 성능표를 목표치로 | 2–3 주 |
| R7. 검증 | 원본 렌더러 출력과 프레임 단위 비교(비상업 사용은 내부 검증 목적), 립싱크 지연 측정 | `generate_offline.py` | 1 주 |
| **합계** | | | **11–13 인주** (1인 기준, GPU 개발 환경 전제) |

R1 은 2.1 질의에서 모델 호출 코드의 Community 적용이 확인되면 0.5 주(정리·검토)로 줄어듭니다.

### 2.2 위험 요소

- **하이퍼파라미터 복원**: ODE 스텝 수, CFG 가중치 기본값, AR(1) 노이즈 계수는 스크립트에 일부만 노출. 잘못 잡으면 립싱크 품질 저하. 원본 렌더러로 내부 비교 검증 필수
- **인트로 모션 pkl**: `intro_motion` 데이터가 HF 게이트 저장소의 아바타 아티팩트에 포함되는지 확인 필요. 없으면 인트로 기능 생략
- **TensorRT 재배포**: NVIDIA 약관 확인(`LICENSE_CLEARANCE.md` 2.4)
- **품질 회귀**: Goodsize 의 미공개 튠(마스크 형태, 배경 합성 파라미터)을 재현하지 못할 수 있음

## 3. 로컬 재생기 (Streamer 대체) 설계

원본 Streamer 의 특허 고지 대상 4가지(오디오 스케줄러, 드리프트 보정 스트림 클록, 프레임 정밀 이벤트 발행, 대칭 이중 스트림 렌더 조정)는 **모두 비대면 원격 스트리밍과 대화 세션 관리**에 관한 것입니다. 현장 PC 로컬 출력에서는 다음처럼 문제를 단순화할 수 있습니다.

| 원본 기능 | 로컬 재생기 처리 | 특허 회피 관점 메모 |
| --- | --- | --- |
| WebRTC 송출(aiortc) | 없음. 프레임을 로컬 윈도우(OpenGL/SDL) 또는 Unity/Unreal 텍스처로 직접 전달 | 스트리밍 전송 자체를 하지 않음 |
| 드리프트 보정 클록 | 오디오 출력 장치 클록을 마스터로 두고 프레임을 오디오 재생 위치에 맞춰 표시(일반적 A/V 동기) | 표준 미디어 플레이어 방식. 그래도 청구항 확인 필요 |
| 오디오 스케줄러·세그먼트 이벤트 | TTS 출력 큐 → 6480 샘플 청크로 절단 → 모델 호출. 세그먼트 시작/끝은 큐 경계로 판단 | 단순 FIFO |
| 끼어들기(barge-in) | 사용자 음성 감지 시 큐 비우기 + 모델 상태를 listen 모드로 전환 | |
| 이중 스트림(speak/listen) | 모델 입력의 `audio_speech`/`audio_listen` 두 트랙을 채우는 것으로 충분. 상대 화자 오디오는 마이크 입력 | 모델 계약이 요구하는 입력 형식 자체는 Community 범위 |
| 재연결·세션 관리 | 없음 |

추정 공수: **3–4 인주** (오디오 I/O, 청크 절단, barge-in, 로컬 렌더 표시, LLM/TTS 연결 제외).

**특허 주의**: 위 단순화가 미공개 출원의 청구항을 회피하는지는 청구항을 보기 전까지 판단할 수 없습니다. `docs/PATENT_REVIEW.md` 참조. 구현 착수 전 변리사 FTO(freedom-to-operate) 검토를 권장합니다.

## 4. 단계 제안

| 단계 | 내용 | 선행 조건 |
| --- | --- | --- |
| P0 | Avaturn 회신 수령, 견적 vs 본 문서 공수 비교 | 문의 발송 |
| P1 | GPU 개발 머신에서 원본 파이프라인 실행(내부 평가·비상업). 1단계 MediaPipe 변경 실측 검증 | Linux + RTX, `pixi run download` |
| P2 | R3·R4·R5 (라이선스 무관, 즉시 착수 가능) | P1 |
| P3 | R1·R2 (2.1 회신에 따라 R1 범위 확정) | P0 |
| P4 | 로컬 재생기 | FTO 검토 |
| P5 | R6·R7 최적화·검증, 고객사 EULA(Attachment A 포함) 준비 | P3, P4 |

## 5. 진행 현황 (2026-09-14): `src/lpavatar/` 패키지

라이선스와 무관한 항목(R2·R3·R4·R5 의 CPU 검증 가능 부분)을 새 패키지 `src/lpavatar/` 로 구현했습니다. 기존 PolyForm 폴더와 import 관계가 없고, LivePortrait(MIT)·ditto-talkinghead(Apache-2.0)·MediaPipe(Apache-2.0) 만을 근거로 작성했습니다. 고지는 `src/lpavatar/NOTICE.md`.

| 모듈 | 내용 | 대응 항목 | 상태 |
| --- | --- | --- | --- |
| `runtime/onnx_engine.py` | ONNX Runtime 엔진 래퍼(CPU/CUDA), 입력 이름·형상 검증 | R5 | 완료. TensorRT 구현은 GPU 확보 후 |
| `registration/face.py` | MediaPipe BlazeFace + Face Mesh(478점) | R3 | 완료 |
| `registration/crop.py` | LivePortrait 크롭 기하(478·203점 축 파서, 유사변환) | R3 | 완료 |
| `registration/landmark203.py`, `extractors.py` | 203점 랜드마크, 외형·모션 추출기, 머리 자세 66빈 → 각도 → 회전행렬 | R3 | 완료 |
| `registration/avatar.py` | 초상 → `Avatar`(프레임, 512 크롭, `m_c2o`, `f_s`, `KPInfo`, `x_s`, 마스크) | R3 | 완료 |
| `render/keypoints.py` | 암시적 키포인트 변환 `x = s·(kp@R + exp) + t` | R2 | 완료 |
| `render/stage.py` | stitch → warp → SPADE decoder(프레임 단위, numpy) | R2 | 완료. 5프레임 배치·TensorRT 최적화(R6)는 미착수 |
| `compose/pasteback.py`, `matting.py`, `pixel_format.py` | 페이스트백, MODNet 매팅, BGR/I420 변환 | R4 | 완료. MODNet 은 공식 ONNX 미확보로 실행 검증 못 함 |
| `artifacts.py` | 공개 ONNX 8종 다운로드(ditto HF, 로그인 불필요) | R5 | 완료 |

### CPU 검증 결과 (ditto 예제 초상, 1280×720 프레임)

| 항목 | 결과 |
| --- | --- |
| 등록 시간(CPU) | 1.5 s |
| stitch(x_s, x_s) 최대 변위 | 0.0055 (항등 입력에서 델타 ≈ 0) |
| 자기 재구성(x_d = x_s) PSNR vs 512 크롭 | 31.8 dB, MAE 3.9/255 |
| 페이스트백 후 마스크 내부 MAE vs 원 프레임 | 4.1/255 |
| 구동 프레임(요 +15°, 입 변위) | 머리 회전·입 변화가 프레임에 이음새 없이 합성됨 (시각 확인) |
| 단위 테스트 | `tests/lpavatar/` 13개 통과(기하 10, 실제 모델 CPU 3) |

### 남은 항목

- **R1 모델 호출기**: 2.1 질의 회신 대기. 착수 시 `KPInfo`·`driving_keypoints` 인터페이스에 `(R, exp)` 시퀀스를 공급하는 형태로 연결
- **R6 실시간 최적화**: GPU 머신에서 TensorRT 엔진, 5프레임 배치 워프, 프레임별 디코드 스트리밍
- **MODNet**: 공식 저장소의 ONNX 내보내기 확보 후 `ModnetMatting` 실행 검증
- **크롭 파라미터**: LivePortrait 기본값(2.3 / −0.125)을 사용. AVTR-1 모션 모델이 학습된 크롭 규약과 다를 수 있어 R1 단계에서 재확인

## 6. 비교 기준

Renderer 상업 라이선스 비용(연간 또는 배포 건별)과 **11–13 인주 + 유지보수**를 비교합니다. 견적이 개발비의 약 절반 이하이고 특허 확인(질의 2.2)이 명확하면 계약이 유리합니다. Streamer 는 어느 경우든 로컬 재생기로 대체하는 쪽이 단순하지만, 특허 청구항 확인이 선행되어야 합니다.
