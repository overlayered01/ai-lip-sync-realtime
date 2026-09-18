# ai-lip-sync

실시간 대화형 얼굴 아바타. 초상 사진 1장과 TTS 오디오 스트림으로 25 fps 립싱크 영상을
고객사 현장 PC 1대에서 로컬 화면에 출력한다. 모든 구성요소가 Apache-2.0 / MIT 이다.

- 음성→모션: ditto-talkinghead LMDM (Apache-2.0), HuBERT 스트리밍 특징
- 렌더러: LivePortrait ONNX (MIT) + 자체 코드 `lpavatar`
- 얼굴 검출·랜드마크: MediaPipe BlazeFace / Face Mesh (Apache-2.0)
- 배경 분리(선택): MODNet (Apache-2.0)

설계·결정·마일스톤은 `docs/DITTO_AVATAR_HANDOFF.md` 에 있다. 특히 11장(AVTR-1 대조
실시간 검토)은 구현 전에 읽어야 한다. 고지는 `NOTICE.md`.

## 구조

```
src/lpavatar/        등록(registration) · 렌더(render) · 합성(compose) · 런타임(runtime)
src/lpavatar/motion/ ditto 음성→모션 어댑터 (M1)
src/avatar_player/   로컬 재생기: 오디오 큐, A/V 동기, 표시, 세션 상태 머신 (M2)
scripts/             모델 다운로드, 아바타 등록, 오프라인 렌더, TensorRT 빌드
tests/               pytest (CPU)
docs/                인수인계 문서, 라이선스·특허 검토 기록
example/             ditto 예제 초상·오디오 (Apache-2.0)
```

## 시작하기 (CPU, Windows 포함)

```bash
python -m pip install -e ".[dev]"
python scripts/download_models.py            # 렌더 그래프 8종 + HuBERT + LMDM + cfg (약 2.3 GB)
set LPAVATAR_MODELS=%CD%\artifacts\lpavatar  # 기본값이므로 생략 가능
set LPAVATAR_TEST_PORTRAIT=%CD%\example\image.png
python -m pytest tests -q
```

`tests/lpavatar` 30개는 CPU 에서 약 25 s 에 통과한다. 모델 파일이 없으면 실제 모델 테스트
4개는 건너뛴다.

오프라인 렌더(온라인 상태기를 그대로 사용, CPU 가능):

```bash
python scripts/register_avatar.py example/image.png            # -> artifacts/avatars/image.npz
python scripts/offline_render.py --audio example/audio.wav --avatar artifacts/avatars/image.npz --out out/example.mp4 --max-seconds 4
# 스냅샷 PNG 와 립싱크 상관계수(오디오 RMS vs 입 변위)를 함께 출력한다
```

CPU(Windows, Python 3.14) 실측: 모션 1.5 s/창(HuBERT + LMDM 10스텝), 렌더 2.4 s/프레임.

## 실시간 테스트 (GPU)

```bash
python -m pip uninstall -y onnxruntime
python -m pip install -e ".[gpu,player]"
# onnxruntime-gpu 1.30 은 CUDA 13 빌드. 런타임은 pip 로 받는다 (약 1.5 GB):
python -m pip install nvidia-cuda-runtime nvidia-cudnn-cu13 nvidia-cublas nvidia-cufft nvidia-curand nvidia-cuda-nvrtc nvidia-nvjitlink
python scripts/realtime_demo.py --audio example/audio.wav --avatar artifacts/avatars/image.npz --live
```

첫 실행 때 워프 그래프를 opset 20 으로, 워프·디코더를 fp16 으로 변환해 `artifacts/lpavatar` 에 저장한다(수십 초).
키: `q` 종료, `b` 끼어들기(오디오를 무음으로 끊고 모션 상태 되감기), `v` 무음 입 고정 토글.

RTX 5080 실측(2026-09-18): 25 fps, 드랍 0, A/V 오프셋 +10 ms, TTS 샘플 도착→표시 0.64 s, 모션 70 ms/창, 렌더 45 ms/프레임(워커 2개).
상세와 주의점(스레드별 CUDA 그래프 캡처)은 `docs/DITTO_AVATAR_HANDOFF.md` 11장 §7.

## 마일스톤 현황

| 단계 | 내용 | 상태 |
| --- | --- | --- |
| M0 | 저장소 구성, lpavatar·테스트·문서 이동, 모델 다운로드 | 완료 (2026-09-18) |
| M1 | `lpavatar.motion` ditto 어댑터, 오프라인 wav→mp4 | 구현 완료, CPU 검증 (2026-09-18). ditto 원본 영상과의 비교는 GPU 세션에서 |
| M2 | 온라인 상태기 + 로컬 재생기 | 데모 완료 (2026-09-18): RTX 5080 에서 25 fps, 드랍 0, 지연 0.64 s. 남은 것: 스트리밍 TTS 입력, 마이크 VAD |
| M3 | TensorRT, 25 fps | ORT CUDA EP + fp16 + CUDA 그래프로 이미 25 fps. TensorRT 는 4060 Ti 급 실측 후 결정 |
| M4 | STT/LLM/TTS 연결, 끼어들기, 대기 동작, 매팅 | |
| M5 | 패키징, 고지, 설치 문서 | |

## 라이선스

자체 코드는 Apache-2.0. 제3자 구성요소는 `NOTICE.md` 참조. InsightFace 계열 모델
(`2d106det.onnx`, `det_10g.onnx`)은 비상업 전용이므로 사용하지 않는다.
