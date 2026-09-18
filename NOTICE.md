# ai-lip-sync — 저작권 및 제3자 고지

이 저장소의 자체 코드(`src/lpavatar/`, `src/avatar_player/`, `scripts/`, `tests/`)는
Apache License, Version 2.0 으로 배포합니다 (https://www.apache.org/licenses/LICENSE-2.0).

이 저장소는 Avaturn AVTR-1 모델, `avtr1_renderer`, `avaturn_live_streamer` 의 코드나
가중치를 포함하지 않으며 참조하지도 않습니다. 배경은 `docs/LICENSE_CLEARANCE.md`,
`docs/PATENT_REVIEW.md` 를 참조하십시오.

## 사용하는 제3자 구성요소

| 구성요소 | 라이선스 | 저작권 | 출처 |
| --- | --- | --- | --- |
| ditto-talkinghead 코드·LMDM 모델(`lmdm_v0.4_hubert.onnx`)·설정(`ditto_cfg/*.pkl`) | Apache-2.0 | © 2024 Ant Group Co., Ltd. | https://github.com/antgroup/ditto-talkinghead , https://huggingface.co/digital-avatar/ditto-talkinghead |
| LivePortrait 모델 (appearance_extractor, motion_extractor, landmark203, stitch_network, warp_network_ori, decoder) | MIT | © 2024 Kuaishou Visual Generation and Interaction Center | https://github.com/KwaiVGI/LivePortrait ; ONNX 재패키징은 위 ditto HF 저장소 `ditto_onnx/` (Apache-2.0) |
| HuBERT 스트리밍 ONNX (`hubert_streaming_fix_kv.onnx`, facebook/hubert 계열) | Apache-2.0 | © Facebook AI Research | 위 ditto HF 저장소 `ditto_pytorch/aux_models/` |
| MediaPipe BlazeFace (short range), Face Mesh V2 (`blaze_face.onnx`, `face_mesh.onnx`) | Apache-2.0 | © The MediaPipe Authors / Google LLC | https://github.com/google-ai-edge/mediapipe ; ONNX 는 위 ditto HF 저장소 `ditto_onnx/` |
| MODNet (배경 분리, 선택) | Apache-2.0 | © 2020 Zhanghan Ke et al. | https://github.com/ZHKKKe/MODNet (ONNX 는 공식 저장소에서 직접 내보내기) |
| onnxruntime | MIT | © Microsoft Corporation | https://github.com/microsoft/onnxruntime |
| numpy | BSD-3-Clause | © NumPy Developers | https://numpy.org |
| OpenCV | Apache-2.0 | © OpenCV team | https://opencv.org |
| scipy | BSD-3-Clause | © SciPy Developers | https://scipy.org |
| python-soxr | LGPL-2.1-or-later (동적 링크) | © Myungchul Keum | https://github.com/dofuuz/python-soxr |
| sounddevice / PortAudio | MIT | © Matthias Geier / Ross Bencina, Phil Burk | https://github.com/spatialaudio/python-sounddevice |
| TensorRT (GPU 단계, 선택) | NVIDIA Software License | © NVIDIA Corporation | https://developer.nvidia.com/tensorrt — 재배포 조건은 NVIDIA 약관 확인 필요 |

## 사용하지 않는 파일

ditto HF 저장소의 `ditto_pytorch/aux_models/2d106det.onnx`, `det_10g.onnx` 는 InsightFace
모델로 비상업 연구 전용이며, 이 프로젝트는 다운로드·사용하지 않습니다. 얼굴 검출·랜드마크는
MediaPipe 로 대체했습니다.

## 라이선스 전문

- Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt
- MIT License (LivePortrait): https://github.com/KwaiVGI/LivePortrait/blob/main/LICENSE
- BSD-3-Clause (numpy): https://github.com/numpy/numpy/blob/main/LICENSE.txt
- BSD-3-Clause (scipy): https://github.com/scipy/scipy/blob/main/LICENSE.txt
- LGPL-2.1 (soxr): https://github.com/dofuuz/python-soxr/blob/main/LICENSE.txt

## 변경 사항 표시 (Apache-2.0 §4(b))

- `src/lpavatar/` 는 LivePortrait·ditto-talkinghead 의 수학과 ONNX 입출력 규격을 numpy 로
  독립 재작성한 것입니다. 파일 헤더의 SPDX 태그를 참조하십시오.
- `src/lpavatar/motion/` 은 ditto-talkinghead 의 `core/atomic_components/{wav2feat,
  condition_handler, audio2motion, motion_stitch}.py`, `core/models/lmdm.py`,
  `stream_pipeline_online.py` 의 로직을 옮기고 실시간 대화용으로 변경(슬라이딩 창, 되감기,
  `vad_alpha` 램프, `overlap_v2=75`)한 것입니다. 변경 내용은 `docs/DITTO_AVATAR_HANDOFF.md` 11장에 있습니다.
