# 실시간 대화형 아바타 (ditto 기반) 신규 프로젝트 인수인계 문서

- 작성일: 2026-09-18
- 목적: Avaturn/AVTR-1 을 사용하지 않고, 완전 오픈 라이선스(Apache-2.0 / MIT) 구성으로 고객사 현장 PC 1대에서 동작하는 실시간 대화형 얼굴 아바타를 만든다.
- 이 문서 하나로 새 저장소에서 작업을 시작할 수 있도록 결정 사항, 가져올 코드, 새로 만들 코드, 모델 입출력, 검증 기준을 담았다.
- 2026-09-18 보완: AVTR-1 실시간 코드와 ditto 온라인 원본 코드를 대조해 지연·청크 절단·끼어들기·대기 동작 4개 항목을 수정했다. 11장 참조. 6.1/6.2 의 해당 문장에는 `[11장]` 표시를 달았다.
- 이전 작업 저장소: `E:\Work\_Test\avtr-1_license-update\avtr-1` (avaturn-live/avtr-1 포크). 이 저장소는 참조 기록으로만 남기고 제품 저장소로 쓰지 않는다.

---

## 1. 결정 사항과 근거

| 결정 | 근거 |
| --- | --- |
| AVTR-1 (Avaturn) 미사용 | 모델은 연매출 1,000만 달러 이상이면 유료, Renderer·Streamer 는 매출 무관 비상업 전용, Streamer 는 특허 유보(독자 구현도 실시권 없음). 상세: 포크 저장소 `docs/LICENSE_CLEARANCE.md`, `docs/PATENT_REVIEW.md` |
| 음성→모션 모델은 ditto-talkinghead LMDM 사용 | 코드·모델 Apache-2.0, 실시간 온라인 모드 제공, 오디오 청크 규격이 동일(16 kHz, 6,480 샘플 → 25 fps 5프레임) |
| 렌더러는 LivePortrait 계열 ONNX + 자체 코드 `lpavatar` | LivePortrait MIT. `lpavatar` 는 이미 구현·CPU 검증 완료(자기 재구성 PSNR 31.8 dB) |
| 얼굴 검출·랜드마크는 MediaPipe | InsightFace 모델이 비상업 전용이라 대체. 이미 구현 완료 |
| 출력은 로컬 화면 + 스피커 | 외부 WebRTC 스트리밍 없음. Streamer 특허 회피 여부를 따질 필요가 없어짐 |
| 잃는 것 | AVTR-1 의 "상대 음성에 반응하는 경청 동작". 규칙 기반 대기 동작(눈 깜빡임·감정 조건·머리 자세 오프셋)으로 대체 |

## 2. 라이선스 구성 (전부 상업 사용 가능)

| 구성요소 | 라이선스 | 출처 |
| --- | --- | --- |
| ditto-talkinghead 코드, LMDM 모델, 설정 | Apache-2.0 | https://github.com/antgroup/ditto-talkinghead , https://huggingface.co/digital-avatar/ditto-talkinghead |
| LivePortrait ONNX (appearance/motion extractor, landmark203, stitch, warp, decoder) | MIT (모델), ONNX 재패키징 Apache-2.0 | 위 HF 저장소 `ditto_onnx/` |
| HuBERT (facebook/hubert-base-ls960 계열, ditto 스트리밍 ONNX) | Apache-2.0 | HF 저장소 `ditto_onnx/hubert.onnx`, `ditto_pytorch/aux_models/hubert_streaming_fix_kv.onnx` |
| MediaPipe BlazeFace, Face Mesh V2 | Apache-2.0 | HF 저장소 `ditto_onnx/blaze_face.onnx`, `face_mesh.onnx` |
| MODNet (배경 분리) | Apache-2.0 | https://github.com/ZHKKKe/MODNet (ONNX 직접 내보내기 필요) |
| lpavatar (자체 코드) | Apache-2.0 | 포크 저장소 `src/lpavatar/` |
| onnxruntime-gpu / PyTorch / OpenCV / numpy | MIT / BSD / Apache | pip |
| TensorRT | NVIDIA 라이선스 | 고객사 PC 재배포 조건은 NVIDIA 약관 확인 필요 |

의무: Apache-2.0·MIT 저작권 고지 동봉(`NOTICE.md`), 변경 파일 표시. 매출 기준, 사용 제한, 특허 고지, 라이선스 전문 배포 의무 없음.

**사용 금지**: `ditto_pytorch/aux_models/2d106det.onnx`, `det_10g.onnx` (InsightFace, 비상업). 포크 저장소의 `src/avtr1_renderer/`, `src/avaturn_live_streamer/` (PolyForm 비상업) 코드는 새 저장소에 복사·참조하지 않는다.

## 3. 시스템 구조

```
[마이크] → VAD/STT → LLM → 스트리밍 TTS ─┐
                                          ▼ 16 kHz mono float32
                          ┌─────────── 로컬 재생기 (신규) ───────────┐
                          │ TTS 오디오 큐 → 6,480샘플 창, 3,200샘플 간격  │
                          │ 무음 주입(대기), 끼어들기 시 큐 비움           │
                          └────────────┬───────────────────────────┘
                                       ▼ 청크
                  ┌────── lpavatar.motion (신규, ditto 어댑터) ──────┐
                  │ HuBERT 스트리밍 → 25 Hz 특징 (5프레임/청크)          │
                  │ 조건 결합 (1024 + emo 8 + eye_open 2 + eye_ball 6 + sc 63 = 1103)
                  │ 신규 5~10프레임 누적 → LMDM DDIM 10스텝, 80프레임 창   │
                  │ 온라인 융합·스무딩 → 265차원 모션 → (R, exp, t, scale)  │
                  └────────────┬───────────────────────────────────┘
                               ▼ x_d (1,21,3) 프레임별
                  lpavatar.render: stitch → warp → decoder → 512 RGB
                               ▼
                  lpavatar.compose: 페이스트백(m_c2o) → [MODNet 알파] → 배경 합성
                               ▼
                  로컬 표시 (OpenGL/SDL 창 또는 Unity/Unreal 텍스처) + 스피커, A/V 동기
```

등록(오프라인, 1회): 초상 → MediaPipe 검출·478점 → 224 크롭 → landmark203 → 512 크롭(scale 2.3, vy −0.125) → appearance/motion extractor → `Avatar`(f_s, kp_info, x_s, m_c2o, mask).

## 4. 새 저장소 구조 (제안)

```
<project>/
  pyproject.toml              # 패키지: lpavatar, avatar_player
  NOTICE.md                   # 3장 표의 고지 전문
  README.md
  src/lpavatar/               # 포크에서 그대로 이동 (아래 5장)
    runtime/  registration/  render/  compose/  artifacts.py  NOTICE.md
    motion/                   # 신규 (6.1)
      __init__.py
      hubert.py               # HuBERT 스트리밍 특징
      condition.py            # 1103차원 조건 결합, emo/eye/sc
      lmdm.py                 # DDIM 샘플러 + ONNX/TRT 호출
      ditto_motion.py         # 온라인 스트리밍 상태기: 청크 → x_d 시퀀스
      vector.py               # 265차원 ↔ (scale, pitch, yaw, roll, t, exp) 변환
  src/avatar_player/          # 신규 (6.2)
    audio_queue.py            # TTS 오디오 수신, 6,480 청크 절단, 무음 주입
    sync.py                   # 오디오 장치 클록 기준 프레임 표시 타이밍
    display.py                # 로컬 창 출력 (초기: OpenCV/SDL, 이후 Unity/Unreal 연동)
    session.py                # 말하기/대기/끼어들기 상태 머신
    main.py                   # 데모 진입점
  tests/lpavatar/             # 포크에서 이동 + motion 테스트 추가
  tests/avatar_player/
  scripts/
    download_models.py        # lpavatar.artifacts.ensure_models + LMDM/HuBERT
    register_avatar.py        # 초상 → Avatar 캐시(npz)
    offline_render.py         # wav + 초상 → mp4 (CPU 검증용)
    build_trt.py              # ONNX → TensorRT 엔진 (GPU)
  docs/
    LICENSE_CLEARANCE.md, PATENT_REVIEW.md, REIMPLEMENTATION_PLAN.md   # 포크에서 이동 (검토 기록)
    DITTO_AVATAR_HANDOFF.md   # 이 문서
```

## 5. 포크에서 가져올 것

포크 저장소 `E:\Work\_Test\avtr-1_license-update\avtr-1` 기준 경로.

| 원본 | 새 위치 | 비고 |
| --- | --- | --- |
| `src/lpavatar/**` | `src/lpavatar/` | 전부 Apache-2.0. `avtr1_renderer` import 없음 |
| `tests/lpavatar/test_geometry.py`, `test_models_cpu.py` | `tests/lpavatar/` | 13개 테스트, CPU 통과 |
| `docs/LICENSE_CLEARANCE.md`, `docs/PATENT_REVIEW.md`, `docs/REIMPLEMENTATION_PLAN.md`, `docs/AVATURN_INQUIRY_DRAFT.md` | `docs/` | 왜 AVTR-1 을 쓰지 않는지의 기록 |
| `pyproject.toml` 의 `[tool.setuptools.packages.find]` 패턴 | 새 pyproject | `include = ["lpavatar*", "avatar_player*"]` 로 변경 |

가져오지 않는 것: `src/avtr1_renderer/`, `src/avaturn_live_streamer/`, `scripts/`, `LICENSE*.md`, `PATENTS.md`, `pixi.toml`, `tests/test_mediapipe_face.py`(avtr1_renderer 의존).

`lpavatar` 공개 API 요약:

```python
from lpavatar.runtime import OnnxEngine                      # OnnxEngine(path, device="cpu"|"cuda")(**inputs) -> {name: ndarray}
from lpavatar.registration.avatar import AvatarRegistrar, RegistrationEngines, RegistrationConfig
from lpavatar.render.keypoints import KPInfo, driving_keypoints, rotation_matrix, bin66_to_degree
from lpavatar.render.stage import RenderEngines, render_face   # (f_s, x_s, x_d) -> (512,512,3) float RGB
from lpavatar.compose.pasteback import pasteback               # (frame, face, m_c2o, mask_frame) -> uint8 frame
from lpavatar.artifacts import ensure_models                   # 공개 ONNX 8종 다운로드
```

`Avatar` 필드: `frame (H,W,3)`, `crop_512`, `m_c2o (3,3)`, `lmk203 (203,2)`, `kp_info: KPInfo(kp, exp, scale, t, pitch, yaw, roll, rot)`, `x_s (1,21,3)`, `f_s (1,32,16,64,64)`, `mask_frame (H,W)`.

## 6. 새로 만들 것

### 6.1 `lpavatar.motion` — ditto 음성→모션 어댑터 (추정 1~2인주)

참조 구현(Apache-2.0): ditto `core/atomic_components/{wav2feat,condition_handler,audio2motion}.py`, `core/models/lmdm.py`, `core/utils/eye_info.py`, `stream_pipeline_online.py`. 로직을 `lpavatar` 스타일(numpy, `Engine` API)로 옮긴다.

**모델 파일**

| 파일 | 위치 (HF digital-avatar/ditto-talkinghead) | 입력 | 출력 |
| --- | --- | --- | --- |
| `hubert_streaming_fix_kv.onnx` (1.4 GB) | `ditto_pytorch/aux_models/` | `input_values (1, 6480)` float32 | `encoding_out` (프레임, 1024) 50 Hz |
| `lmdm_v0.4_hubert.onnx` (192 MB) | `ditto_onnx/` | `x (1,80,265)`, `cond_frame (1,265)`, `cond (1,80,1103)`, `time_cond (1,) int64` | `pred_noise (1,80,265)`, `x_start (1,80,265)` |
| `v0.4_hubert_cfg_trt_online.pkl` | `ditto_cfg/` | 온라인 설정값·`emo` 배열(600,8)·`delta_eye_arr (15,63)`·`v_min_max_for_clip (4,265)` | pickle, numpy 만 필요 |

**오디오 특징 (hubert.py)**: 청크 = 6,480 샘플 = (과거 3 + 현재 5 + 미래 2) 프레임 × 640 + 80. HuBERT 출력 50 Hz 중 `[-14:-4]` 10프레임을 (5, 2, 1024) 로 묶어 평균 → 25 Hz 5프레임 × 1024. 오프라인(wav 전체)은 앞에 (6480 − 7×640) 샘플, 뒤에 6480 샘플의 무음 패딩 후 5프레임씩 슬라이딩.

**조건 결합 (condition.py)**: 프레임마다 `[hubert 1024 | emo 8 | eye_open 2 | eye_ball 6 | sc 63]` = 1103.
- `emo`: 8클래스 softmax(선택 라벨에 8, 나머지 0). 라벨 순서 Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise, Contempt. 기본 4(Neutral). 온라인 설정의 `emo (600,8)` 배열을 프레임 인덱스로 순환 사용해도 됨.
- `eye_open (2)`, `eye_ball (6)`: 512 크롭에 MediaPipe 478점을 다시 돌려 계산. 눈 인덱스 L: 33/133(폭), 145-159, 144-160, 153-158(높이), 홍채 468; R: 263/362, 374-386, 373-387, 380-385, 홍채 473. open = 세 높이 합 / 폭. ball_move = (홍채 − 눈 중심) 방향에서 눈 방향을 뺀 벡터 × 거리. 이미지 아바타는 1개 값을 모든 프레임에 복제. (`lpavatar.registration.face` 재사용)
- `sc (63)`: 소스 `kp_info.kp.flatten()`.

**모션 벡터 265 (vector.py)**: 순서 `[scale−1 (1) | pitch (66) | yaw (66) | roll (66) | t (3) | exp (63)]`. `kp` 제외. pitch/yaw/roll 은 66빈 로짓 그대로이며 `bin66_to_degree` 로 각도 변환 후 `rotation_matrix`. `v_min_max_for_clip[0]`, `[1]` 로 클리핑.

**LMDM 샘플러 (lmdm.py)**: 코사인 베타 스케줄(n=1000, s=8e-3), DDIM eta=1, 온라인 10스텝(오프라인 50). `times = linspace(−1, 999, steps+1)` 를 역순 정수로, 각 스텝 `x = x_start·√ᾱ_next + c·pred_noise + σ·noise`, 마지막은 `x = x_start`. 시작 `x ~ N(0,1) (1,80,265)`. 스텝별 noise 는 setup 시 고정 생성(ditto 동일).

**온라인 상태기 (ditto_motion.py)**: ditto 배포 설정값 `seq_frames=80, overlap_v2=70 → valid_clip_len=10, sampling_timesteps=10, fix_kp_cond=1, fix_kp_cond_dim=[0,202], smo_k_d=3`, `overall_ctrl_info={delta_pitch: 2}`. `overlap_v2` 는 생성자 인자로 두고 기본값을 75(valid_clip_len=5) 로 한다. `[11장 §1]`
1. 시작 시 무음 `overlap×640` 샘플로 특징 70프레임을 만들어 버퍼 초기화.
2. 청크마다 특징 5프레임을 큐에 넣고, 신규 `valid_clip_len` 프레임이 쌓이면 최근 80프레임 창으로 LMDM 실행. 유효 출력은 새 예측의 꼬리가 아니라 직전 결과와 융합한 구간이므로 출력이 창 끝보다 `fuse_length` 프레임 뒤에 있다. 이것이 지연의 주원인이다. `[11장 §1]`
3. 첫 클립은 전부 버리고 `d0`(상대 모션 기준)만 저장. 이후 클립은 융합 구간 `valid_clip_len` 프레임을 유효 출력으로 사용, 직전 결과와 `fuse_length=min(overlap_v2, valid_clip_len)` 선형 융합, 처음 202차원(scale~roll)에 3프레임 이동평균.
4. `kp_cond`: 클립마다 소스 벡터로 리셋하되 `[0:202]` 는 직전 예측값 유지(`fix_kp_cond=1`).
5. 출력 265차원 → dict → `relative_d`: `x_d = x_s + (x_d − d0)` (exp, pitch, yaw, roll, t 에 대해), exp 는 입 인덱스 [6,12,14,17,19,20]·눈 인덱스 [11,13,15,16,18] 만 구동값, 나머지는 소스값 (`_fix_exp_for_x_d_info_v2`). 눈 깜빡임은 `delta_eye_arr (15,63)` 를 60~100프레임 간격으로 삽입.
6. `driving_keypoints(kp_info, R_d, exp_d)` 로 x_d 생성 → `render_face`.
7. 상태기에 `rewind_pending()` 을 둔다: 끼어들기 시 아직 출력하지 않은 예측 꼬리와 미소비 특징을 되감아 "취소된 오디오가 오지 않은 것" 과 같은 상태로 만든다. `[11장 §3]`
8. 프레임별 제어 입력을 받는다: `vad_alpha`(입 exp 를 소스로 되돌리는 비율), `delta_pitch/yaw/roll`, kp_cond 전체 리셋 플래그. ditto `motion_stitch.ctrl_vad`/`ctrl_motion` 과 같은 의미. `[11장 §4]`

마지막 청크 처리: 스트림 종료 신호에서 부족한 프레임을 마지막 특징으로 패딩해 1회 더 실행.

**검증**: `scripts/offline_render.py` 로 ditto `example/audio.wav` + `example/image.png` 를 렌더한 mp4 가 ditto 원본 `inference.py` 결과와 시각적으로 동등한지 확인(CPU 가능, 수 분 소요). 무음 입력에서 입이 닫혀 있고 미세 움직임만 있는지 확인.

### 6.2 `avatar_player` — 로컬 재생기 (추정 2~3인주)

- **입력**: TTS 스트리밍 PCM(임의 샘플레이트) → 16 kHz mono float32 리샘플(soxr) → 링 버퍼. 모델 입력은 **6,480 샘플 창을 3,200 샘플(5프레임) 간격으로 미는 슬라이딩 창**이다: 과거 1,920 + 현재 3,200 + 미래 1,360(2프레임 + 80). 이웃 창과 과거 3프레임·미래 2프레임이 겹친다. 겹침 없이 6,480 씩 자르면 405 ms 오디오마다 200 ms 영상이 나와 입이 오디오의 절반 속도로 움직인다. 매 창 5프레임(200 ms) 생성. `[11장 §2]`
- **대기 상태**: TTS 큐가 비면 무음 청크를 계속 주입해 대기 동작 유지. 무음 구간 프레임에는 `vad_alpha` 를 5프레임 램프로 1→0 으로 내려 입 exp 를 소스로 고정하고, 발화 시작 시 0→1 로 올린다. 대기가 길어지면(예: 30 s) kp_cond 를 소스 벡터로 전체 리셋해 머리 자세 랜덤워크를 막는다. 선택적으로 `emo` 라벨·`delta_pitch` 등으로 "듣는 자세" 연출. `[11장 §4]`
- **끼어들기**: VAD 가 사용자 발화를 감지하면 (1) 오디오 장치 재생 위치 p 를 읽고, (2) TTS 수신 중단·오디오 링 버퍼 비움, (3) 표시 큐와 렌더 큐에서 p 이후 프레임 폐기, (4) 상태기 `rewind_pending()` 호출, (5) 무음 주입 재개와 `vad_alpha` 램프 다운. LMDM 의 융합·kp_cond 상태는 되감기 후 유지되어 동작이 이어진다. `[11장 §3]`
- **A/V 동기**: 오디오 출력 장치 재생 위치를 마스터 클록으로 두고, 프레임 i 의 표시 시각 = 해당 청크 오디오의 재생 시작 + i×40 ms. 오디오는 모션 선행 필요량 + 연산 시간만큼 버퍼링 후 재생 시작. 선행 필요량은 `overlap_v2=75` 에서 480 ms, `70` 에서 880 ms 이다(11장 표). 재생 위치보다 늦은 프레임은 버리고, 단계 간 큐는 1~2 청크로 제한한다(ditto 원본의 큐 크기 100 은 오프라인용). `[11장 §1, §5]`
- **표시**: 1단계 OpenCV 창(검증용) → 2단계 SDL/OpenGL 전체화면 → 3단계 Unity/Unreal 에 텍스처 공유(공유 메모리 또는 Spout/NDI 대체는 라이선스 확인).
- **파이프라인 스레딩**: ditto 온라인 파이프라인과 같이 음성→모션 / 스티치·워프·디코드 / 페이스트백·표시를 스레드+큐로 분리. LMDM 은 `valid_clip_len` 프레임마다 10 DDIM 스텝을 몰아서 돌기 때문에 연산이 불규칙하다. 모션 스레드와 렌더 스레드 사이에 최소 1청크 출력 버퍼를 둔다. `[11장 §5]`

### 6.3 GPU 최적화 (추정 2~3인주, Linux + RTX 필요)

- 모든 ONNX 를 TensorRT fp16 엔진으로 변환(`scripts/build_trt.py`, ditto `scripts/cvt_onnx_to_trt.py` 참고). warp 는 `warp_network_ori.onnx`(표준 GridSample) 사용, 커스텀 플러그인 불필요.
- 5프레임 배치 워프 + 프레임별 디코드로 첫 프레임 지연 최소화.
- 목표: 청크당(5프레임) 총 연산 ≤ 200 ms → 25 fps 유지. 측정 GPU: RTX 4060 Ti 급.
- `lpavatar.runtime` 에 `TensorRTEngine`(같은 `Engine` 프로토콜) 추가.

### 6.4 부수 작업

- MODNet 공식 ONNX 내보내기 → `lpavatar.compose.matting.ModnetMatting` 실행 검증. RGBA 초상만 필요.
- `NOTICE.md` 작성(3장 표 전문 + 각 라이선스 원문 링크).
- 아바타 등록 결과 캐시(npz) 와 다중 아바타 전환.

## 7. 마일스톤

| 단계 | 내용 | 완료 기준 |
| --- | --- | --- |
| M0 | 새 저장소 생성, lpavatar·테스트·문서 이동, 모델 다운로드 스크립트 | `pytest tests/lpavatar` 13개 통과 (CPU) |
| M1 | `lpavatar.motion` 어댑터, 오프라인 wav→mp4 | ditto 예제와 동등한 립싱크 영상 (CPU) |
| M2 | 온라인 상태기 + 로컬 재생기(OpenCV 창), 마이크 없이 TTS 파일 스트리밍 | TTS 샘플 도착→표시 지연 ≤ 0.8 s (`overlap_v2=75`), 프레임 드랍 없음 (GPU). 끼어들기 후 입 움직임 잔류 ≤ 1프레임. 30분 무음 세션에서 입 닫힘 유지 |
| M3 | TensorRT 엔진, 25 fps 확보 | RTX 4060 Ti 에서 청크당 ≤ 200 ms. LMDM 10스텝을 매 청크 실행하는 `overlap_v2=75` 기준으로 측정. 미달 시 `sampling_timesteps` 6~8 또는 `overlap_v2=70` 으로 후퇴하고 지연 수치를 갱신 |
| M4 | STT/LLM/TTS 연결, 끼어들기, 대기 동작, 매팅·배경 | 현장 데모 시나리오 통과 |
| M5 | 패키징, 고지 파일, 설치 문서 | 고객사 PC 설치 리허설 |

## 8. 개발 환경

- Python 3.12, numpy 2.x, opencv-python-headless, onnxruntime-gpu ≥ 1.20, scipy(softmax), soxr(리샘플), pytest, ruff
- GPU 단계: CUDA 12.x, TensorRT 10.x (ditto 원본은 8.6 기준이라 변환 스크립트 조정 필요), torch 는 필수 아님
- CPU 개발 PC 에서도 등록·오프라인 렌더·단위 테스트는 전부 가능 (Windows 포함). 이전 세션에서 Windows CPU 로 lpavatar E2E 를 검증했음.
- 모델 저장 위치: 환경변수 `LPAVATAR_MODELS` (기본 `<repo>/artifacts/lpavatar`)

## 9. 위험과 확인 사항

| 위험 | 대응 |
| --- | --- |
| ditto 립싱크 품질이 AVTR-1 데모에 못 미칠 수 있음 | M1 에서 ditto 원본 결과와 비교, 필요 시 `emo`·스무딩 파라미터 조정 |
| 응답 지연이 AVTR-1 보다 길다 (선행 필요량 400 ms → 480 ms, 배포 설정 그대로면 880 ms) | `overlap_v2=75` 기본, GPU 실측 후 확정. 11장 |
| 무음 입력에서 입 떨림·장시간 머리 자세 드리프트 | `vad_alpha` 램프, 대기 중 kp_cond 주기 리셋. M2 에서 30분 무음 세션으로 검증 |
| 경청 동작 없음 | 규칙 기반 대기 동작으로 대체. 요구사항 확정 필요 |
| RTX 4060 급에서 25 fps 미달 가능 | 해상도(512 크롭 유지, 출력 720p), fp16, 배치 워프. 실측 후 GPU 사양 결정 |
| Ant Group 특허 여부 미확인 | 공개 검색 미실시. 필요 시 변리사 확인. Apache-2.0 은 기여자 특허 실시권을 포함함 |
| TensorRT 재배포 | NVIDIA 약관 확인 또는 onnxruntime-gpu 로 대체 가능성 검토 |
| HuBERT 스트리밍 ONNX 1.4 GB | 설치 용량 고려. 필요 시 fp16 변환 |

## 10. 참고 자료

- ditto-talkinghead 코드: https://github.com/antgroup/ditto-talkinghead (Apache-2.0)
- ditto 모델·설정: https://huggingface.co/digital-avatar/ditto-talkinghead (Apache-2.0, 로그인 불필요)
- LivePortrait: https://github.com/KwaiVGI/LivePortrait (MIT)
- MediaPipe 모델 카드: BlazeFace https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf , Face Mesh V2 https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf
- MODNet: https://github.com/ZHKKKe/MODNet (Apache-2.0)
- 이전 검토 기록(포크 저장소 `docs/`): 라이선스 판정, Goodsize 특허 조사, AVTR-1 경로 공수 산정, Avaturn 문의 초안

---

## 11. 실시간 검토 보완 (2026-09-18, AVTR-1 대조)

포크 저장소의 AVTR-1 실시간 코드(`avtr1_motion_generator.py`, `speech/speech_scheduler.py`, `worklets/rendering.py`, `renderer/models.py`)와 ditto 원본(`stream_pipeline_online.py`, `audio2motion.py`, `motion_stitch.py`, `inference.py`) 및 HF `ditto_cfg/v0.4_hubert_cfg_trt_online.pkl` 실제 값을 대조한 결과. AVTR-1 코드는 동작 요구사항 확인에만 참조했고 복사하지 않았다.

### §1 지연: 배포 설정 그대로면 880 ms

온라인 모드 출력 타이밍을 코드로 따라가면 다음과 같다.

- `_audio2motion_worker` 는 신규 특징이 `valid_clip_len` 프레임 쌓여야 LMDM 을 돌린다. 배포 설정은 10프레임 = 400 ms.
- 첫 클립은 폐기하고 `res_kp_seq_valid_start = 80 - fuse_length` 로 잡는다. 이후 유효 출력은 직전 클립 꼬리와 융합한 구간이라 창 끝보다 `fuse_length` 프레임 뒤다.
- 따라서 모션 프레임 k 를 내보내려면 오디오 특징이 k+20 프레임까지 있어야 하고, 마지막 창의 HuBERT 미래 2프레임을 더하면 오디오는 k+22 프레임까지 필요하다.

| 항목 | AVTR-1 (포크 코드) | ditto `overlap_v2=70` | ditto `overlap_v2=75` |
| --- | --- | --- | --- |
| 오디오 선행 필요량, 블록 첫 프레임 | 10프레임 = 400 ms | 22프레임 = 880 ms | 12프레임 = 480 ms |
| 오디오 선행 필요량, 블록 마지막 프레임 | 6프레임 = 240 ms | 13프레임 = 520 ms | 8프레임 = 320 ms |
| 모션 모델 실행 주기 | 200 ms 마다 encode 1 + decode 4 | 400 ms 마다 LMDM 10스텝 | 200 ms 마다 LMDM 10스텝 |
| RTX 4060 Ti 청크당 연산 | 166 ms 실측 (README) | 미측정 | 미측정, LMDM 비용 2배 |

AVTR-1 스트리머는 present 5프레임 + future 5프레임 + 80샘플을 모델에 주고(`renderer/models.py`), 다음 청크 렌더를 현재 청크 소진 100 ms 전에 시작한다. 이 문서의 이전 판에 있던 "약 480 ms" 는 융합 지연 10프레임을 빠뜨린 값이었다.

**결정**: `overlap_v2=75` 를 기본으로 한다. `valid_clip_len=5, fuse_length=5` 가 되어 원본 `_fuse`/`_smo` 코드 변경 없이 동작한다. 융합 구간이 200 ms 로 짧아지는 품질 영향과 LMDM 2배 연산은 M1·M3 에서 확인한다. 출력 버퍼 1청크를 더하면 TTS 샘플 도착→표시 지연 목표는 약 0.7~0.8 s 다. 참고로 AVTR-1 은 같은 기준 약 0.6 s 였다.

### §2 청크 절단: 슬라이딩 창

ditto `inference.py` 온라인 경로는 `range(0, len(audio), 5*640)` 으로 이동하며 `audio[i:i+6480]` 을 자른다. 창 6,480, 간격 3,200 이고 앞에 무음 3×640 을 붙여 시작한다. AVTR-1 도 창 6,480(현재 5 + 미래 5 + 80), 간격 3,200 이며 과거 3프레임은 상태에서 가져온다. 두 방식 모두 이웃 창과 겹친다. 이전 판 6.2 의 "청크 간 겹침 없음" 은 오류이며, 그대로 구현하면 오디오 405 ms 당 영상 200 ms 가 나와 립싱크 속도가 절반이 된다.

`audio_queue.py` 규격: 16 kHz 링 버퍼에서 hop = 3,200 마다 `[t-1920, t+3200+1360)` 창을 만든다. 미래 1,360 샘플이 아직 없으면 대기한다(TTS 스트리밍 중) 또는 무음으로 채운다(대기 상태·발화 종료). 발화 종료 후 무음이 이어지므로 ditto 오프라인의 "마지막 청크 패딩" 은 세션 종료 시에만 필요하다.

### §3 끼어들기: 상태 되감기

AVTR-1 스트리머 `SpeechScheduler.interrupt()` 는 큐를 비울 때 이미 모델에 future 로 보여준 구간만 남겨, 모델이 본 오디오와 재생되는 오디오를 일치시킨다. ditto 에는 대응 개념이 없어 재생기와 상태기가 직접 처리해야 한다. 끼어들기 시점에 상태기 안에 취소된 오디오가 세 곳 남는다.

| 위치 | 내용 | 방치 시 증상 |
| --- | --- | --- |
| `item_buffer` | LMDM 에 아직 안 들어간 특징 ≤ `valid_clip_len` 프레임 | 취소된 말의 입 모양이 다음 클립에 반영 |
| `res_kp_seq` 꼬리 `fuse_length` 프레임 | 취소된 오디오로 예측했지만 미출력 | 다음 무음 클립과 융합되어 최대 200~400 ms 입 움직임 잔류 |
| HuBERT 창의 과거 3프레임 | 취소된 오디오의 꼬리 | 다음 창 특징에 미세 오염 |

`rewind_pending()` 절차: `local_idx -= valid_clip_len`, `audio_feat = audio_feat[: local_idx + (80 - valid_clip_len)]`, `res_kp_seq = res_kp_seq[:, :-fuse_length]`, `item_buffer` 비움, HuBERT 과거 꼬리를 무음으로 교체. `res_kp_seq_valid_start`·`kp_cond`·`d0`·깜빡임 인덱스는 유지한다. 되감기 직전에 이미 출력된 프레임까지는 상태가 일관되므로 이후 무음 클립과 자연스럽게 융합된다. 재생기 쪽은 오디오 장치 재생 위치 p 이후의 표시 큐·렌더 큐 프레임을 폐기한다. AVTR-1 과 마찬가지로 p 직후 표시 버퍼에 이미 올라간 1~2프레임은 되돌릴 수 없다.

### §4 대기 동작: 무음 검증과 드리프트

- AVTR-1 은 무음 두 트랙에서 대기 미세 동작을 모델이 직접 만들고(`generate_offline --duration`), `w_kp=3.0` CFG 로 소스 자세로 되돌리는 힘이 모델 안에 있다.
- ditto LMDM 의 무음 출력은 검증되지 않았다. 원본 `motion_stitch.py` 에 `vad_alpha<1` 일 때 입 인덱스 exp 를 소스로 되돌리는 `ctrl_vad` 가 있어 저자들도 무음 보정을 두었다. 재생기는 무음 프레임에 `vad_alpha` 를 5프레임 램프로 0 까지 내리고 발화 시작에 1 로 올린다. 램프는 오디오 타임라인 기준으로 프레임에 붙여 상태기에 전달한다.
- `fix_kp_cond_dim=[0,202]` 는 scale·pitch·yaw·roll 을 클립마다 직전 예측값으로 넘기므로 수 시간 세션에서 머리 자세가 랜덤워크할 수 있다. `v_min_max_for_clip` 이 상한을 막지만 경계에 붙을 수 있다. 대기 30 s 마다 kp_cond 를 소스 벡터로 전체 리셋하고, 리셋 직후 클립은 융합으로 자연히 이어진다.
- 깜빡임 인덱스는 `N_d=-1` 이면 3,000프레임 목록을 만들어 나머지 연산으로 순환하므로 장시간 문제 없다. 배포 설정의 `overall_ctrl_info={delta_pitch: 2}` 는 유지한다. `emo (600,8)` 배열은 프레임 인덱스로 순환한다.
- M2 검증 항목에 "30분 무음 세션에서 입 닫힘 유지, 머리 자세가 소스 ±5° 이내" 를 추가한다.

### §5 스레딩과 A/V 동기

- ditto 의 6단계 스레드 구조는 오프라인용이어서 큐 크기가 100 이다. 한 단계가 느려지면 영상이 최대 4 s 오디오 뒤로 밀린다. 재생기는 큐를 1~2 청크로 제한하고, 재생 위치보다 늦은 프레임은 버린다.
- LMDM 은 `valid_clip_len` 프레임마다 10스텝을 몰아서 돌아 연산이 불규칙하다. AVTR-1 은 매 청크 encode 1 + decode 4 로 균등하다. 모션 스레드와 렌더 스레드를 분리하고 사이에 1청크 출력 버퍼를 둔다.
- 스티치·워프·디코더는 AVTR-1 과 같은 LivePortrait ONNX 이므로 렌더 비용은 동일하다. 4060 Ti 실측 166 ms 의 대부분이 이 구간이라 ditto 의 추가 비용은 HuBERT 차이와 LMDM 만이다.

### §6 구현 현황 (2026-09-18)

- `src/lpavatar/motion/` 구현 완료: `hubert.py`(슬라이딩 창, 오프라인 패딩), `condition.py`(1103 결합, 눈 특징, ditto 참조 캐릭터 `ch_info` 조건), `vector.py`, `lmdm.py`(DDIM), `ditto_motion.py`(온라인 상태기 `Audio2MotionOnline` + `rewind()`, 리타겟 `MotionRetarget` + `FrameControl(vad_alpha, delta_*)`, 파사드 `DittoMotionStream`).
- `KPInfo` 에 66빈 로짓(`pitch_bins` 등) 추가. ditto 는 로짓 위에서 상대 모션을 계산하므로 필요.
- 테스트 30개 통과(가짜 엔진으로 §1 타이밍·§3 되감기 검증, 실제 HuBERT+LMDM 무음 4창 검증). `tests/lpavatar/test_motion.py` 의 `test_online_output_timing` 이 §1 표의 12/22 프레임 값을 고정한다.
- `scripts/offline_render.py` 로 ditto 예제 4 s 렌더: 립싱크 상관(오디오 RMS vs 입 변위) 0.39, 입 열림·페이스트백 정상. CPU 실측 모션 1.5 s/창, 렌더 2.4 s/프레임.
- 미완: ditto 원본 `inference.py` 결과와의 시각 비교(GPU 필요), `condition_on="avatar"` 모드 품질 확인, `rewind()` 의 실제 오디오 끼어들기 연동(M2).

### §7 GPU 실시간 실측 (RTX 5080, onnxruntime-gpu 1.30 / CUDA 13, 2026-09-18)

`scripts/realtime_demo.py` 로 ditto 예제 15.75 s 를 스피커 재생 + OpenCV 창 표시로 실행한 결과. TensorRT 없이 ONNX Runtime CUDA EP 만 사용.

| 항목 | 값 |
| --- | --- |
| 표시 fps | 24.3~25.7 (드랍 0, 표시 전 스킵 0) |
| A/V 오프셋 | 평균 +10 ms, 최대 +20 ms (오디오 장치 클록 기준) |
| TTS 샘플 도착 → 첫 프레임 표시 (`--live`) | 0.64 s (`overlap_v2=75`, 선행 480 ms + 연산) |
| 모션 창당 (HuBERT + LMDM 10스텝, `--live`) | 70 ms (예산 200 ms) |
| 렌더 프레임당, 워커 1개 기준 | 45~50 ms (워커 2개 병렬로 25 fps 확보) |

이를 위해 필요했던 변경:

- **워프 그래프 opset 20 변환** (`lpavatar.artifacts.ensure_cuda_warp`): 원본 opset 17 의 `GridSample` 은 CUDA EP 에서 4-D 만 지원해 5-D 볼륨 샘플링에서 실행 오류. opset 20 변환 후 CUDA 18 ms (CPU 560 ms), 원본과 최대 오차 3.5e-4.
- **fp16 변환** (`ensure_fp16`, onnxconverter-common, I/O 는 fp32 유지): 디코더 27.7 → 15.7 ms (PSNR 78 dB), 워프 18.8 → 9.9 ms (`GridSample` 은 fp32 유지, 디코드 PSNR 59 dB).
- **CUDA 그래프 + IO 바인딩** (`OnnxEngine(cuda_graph=True)`, 정적 형상 그래프에 자동 적용): LMDM 스텝 15.5 → 5.5 ms. 렌더 그래프에는 이득 없음. **주의**: ORT 는 스레드별로 그래프를 캡처하며, 캡처가 다른 스레드의 GPU 작업과 겹치면 `operation not permitted when stream is capturing` 으로 실패한다. 워커 스레드마다 시작 시 한 번에 하나씩 워밍업(렌더 워커 → 모션 스레드)한 뒤 동시 실행해야 한다. 데모의 `capture_lock` + `Barrier` 가 그 순서를 강제한다.
- **페이스트백** `cv2.warpAffine(uint8)` + `cv2.blendLinear`: 38 ms → 3.4 ms (float 참조 대비 최대 오차 2/255).
- **재생기 규칙**: 오디오 장치 재생 위치가 마스터 클록. 이미 늦은 프레임은 렌더 전에 스킵(`skip`), 렌더 후 늦으면 표시에서 폐기(`drop`). 이 규칙이 없으면 렌더가 예산을 조금만 넘어도 지연이 누적되어 몇 초 뒤 전 프레임이 늦어진다(실측: 42.7 ms/프레임에서 8 s 후 붕괴).

M2 완료 기준 "지연 ≤ 0.8 s, 프레임 드랍 없음" 은 이 데모로 충족. 남은 M2 항목: 실제 스트리밍 TTS 입력, 마이크 VAD 끼어들기 연동(데모의 `b` 키가 `rewind()` 경로를 호출함), 30분 무음 세션 검증. 목표 GPU(RTX 4060 Ti 급)에서의 재측정은 M3.

### §8 문서와 일치하는 부분

HuBERT 입력 규격과 `[-14:-4]` 슬라이스, 1103 조건 결합, 265 벡터 순서, 코사인 DDIM 스케줄과 setup 시 고정 noise, `_fix_exp_for_x_d_info_v2` 의 입·눈 인덱스, 세션 시작 시 무음 70프레임 워밍업과 첫 클립 폐기는 원본 코드와 일치한다. 워밍업은 대기 상태에서 한 번만 일어나므로 지속 무음 주입 설계와 맞는다. 경청 동작 부재는 1장에서 이미 인지한 손실이며, AVTR-1 의 AR(1) 상관 노이즈 역할은 ditto 에서 70프레임 중첩 융합이 대신한다.
