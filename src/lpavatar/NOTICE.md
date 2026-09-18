# lpavatar — third-party notices

`src/lpavatar/` is licensed under the Apache License, Version 2.0
(https://www.apache.org/licenses/LICENSE-2.0). It is written from the
following upstream projects only. It contains no code from Avaturn's
`avtr1_renderer` / `avaturn_live_streamer` packages (PolyForm-Noncommercial);
those were consulted, in a separate repository, only to establish behavioural
requirements (tensor shapes, crop parameters, frame layout).

| Upstream | License | Used for |
| --- | --- | --- |
| LivePortrait — https://github.com/KwaiVGI/LivePortrait | MIT, © 2024 Kuaishou Visual Generation and Interaction Center | crop geometry (`registration/crop.py`), implicit keypoint transform and head-pose decoding (`render/keypoints.py`), model semantics of appearance / motion extractor, stitching, warping and SPADE decoder |
| ditto-talkinghead — https://github.com/antgroup/ditto-talkinghead | Apache-2.0, © 2024 Ant Group Co., Ltd. | portable ONNX graphs (`digital-avatar/ditto-talkinghead` on Hugging Face), numpy reference wrappers, paste-back mask, MediaPipe pre/post-processing |
| MediaPipe — https://github.com/google-ai-edge/mediapipe | Apache-2.0, © The MediaPipe Authors | BlazeFace short-range detector and Face Mesh V2 models (`registration/face.py`) |
| MODNet — https://github.com/ZHKKKe/MODNet | Apache-2.0, © 2020 Zhanghan Ke et al. | portrait matting (`compose/matting.py`) |

```
Copyright (c) 2024 Kuaishou Visual Generation and Interaction Center
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction ... (MIT License, full text at
https://github.com/KwaiVGI/LivePortrait/blob/main/LICENSE)
```

The audio-to-motion stage (`lpavatar/motion/`) is a port of ditto-talkinghead's
`core/atomic_components/{wav2feat, condition_handler, audio2motion, motion_stitch}.py`,
`core/models/lmdm.py` and `stream_pipeline_online.py` (Apache-2.0, © 2024 Ant
Group Co., Ltd.), modified for live conversation (sliding audio window,
rewind on barge-in, `vad_alpha` ramp, `overlap_v2=75`). The HuBERT streaming
ONNX (`hubert_streaming_fix_kv.onnx`) and LMDM ONNX come from the same
Hugging Face repository under Apache-2.0. The repository-level `NOTICE.md`
lists every third-party component.
