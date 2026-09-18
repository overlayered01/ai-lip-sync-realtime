# SPDX-FileCopyrightText: 2026 ai-lip-sync contributors
# SPDX-License-Identifier: Apache-2.0

"""Real HuBERT + LMDM on CPU. Skipped unless ``LPAVATAR_MODELS`` holds
``hubert_streaming_fix_kv.onnx``, ``lmdm_v0.4_hubert.onnx``,
``v0.4_hubert_cfg_trt_online.pkl`` plus the registration graphs, and
``LPAVATAR_TEST_PORTRAIT`` points at a portrait. Takes ~1 min (1.4 GB HuBERT)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from lpavatar.artifacts import PUBLIC_MODELS

_ROOT = os.environ.get("LPAVATAR_MODELS")
_PORTRAIT = os.environ.get("LPAVATAR_TEST_PORTRAIT")
_MOTION = (
    "hubert_streaming_fix_kv.onnx",
    "lmdm_v0.4_hubert.onnx",
    "v0.4_hubert_cfg_trt_online.pkl",
)


def _ready() -> bool:
    if not (_ROOT and _PORTRAIT and Path(_PORTRAIT).is_file()):
        return False
    root = Path(_ROOT)
    return all((root / f"{n}.onnx").is_file() for n in PUBLIC_MODELS) and all(
        (root / n).is_file() for n in _MOTION
    )


pytestmark = pytest.mark.skipif(not _ready(), reason="motion models / portrait not available")


@pytest.fixture(scope="module")
def stream_and_avatar():
    from lpavatar.motion.ditto_motion import DittoMotionStream
    from lpavatar.registration.avatar import AvatarRegistrar, RegistrationEngines
    from lpavatar.runtime import OnnxEngine

    root = Path(_ROOT)
    eng = RegistrationEngines(
        blaze_face=OnnxEngine(root / "blaze_face.onnx"),
        face_mesh=OnnxEngine(root / "face_mesh.onnx"),
        landmark203=OnnxEngine(root / "landmark203.onnx"),
        appearance_extractor=OnnxEngine(root / "appearance_extractor.onnx"),
        motion_extractor=OnnxEngine(root / "motion_extractor.onnx"),
    )
    avatar = AvatarRegistrar(eng).register(_PORTRAIT, avatar_id="test")
    stream = DittoMotionStream.from_ditto_cfg(
        root / "v0.4_hubert_cfg_trt_online.pkl",
        avatar,
        hubert=OnnxEngine(root / "hubert_streaming_fix_kv.onnx"),
        lmdm=OnnxEngine(root / "lmdm_v0.4_hubert.onnx"),
        seed=0,
    )
    return stream, avatar


def test_silence_windows_produce_closed_mouth_frames(stream_and_avatar):
    from lpavatar.motion.ditto_motion import LIP_KP
    from lpavatar.motion.hubert import WINDOW_SAMPLES

    stream, avatar = stream_and_avatar
    assert stream.cfg.overlap_v2 == 75 and stream.audio_lookahead_frames == 12
    outs = []
    for _ in range(4):
        outs += stream.push_window(np.zeros(WINDOW_SAMPLES, np.float32))
    assert len(outs) == 15  # windows 1..3 each release a 5-frame block
    assert [o.frame_idx for o in outs] == list(range(15))
    for o in outs:
        assert o.x_d.shape == (1, 21, 3) and np.isfinite(o.x_d).all()
        assert np.isfinite(o.vector).all()
    # relative motion: first frames stay close to the source keypoints
    d = np.abs(outs[0].x_d - avatar.x_s).max()
    assert d < 0.15, f"frame 0 drifted {d:.3f} from the source keypoints"
    lip_delta = np.stack([o.motion.exp.reshape(21, 3)[list(LIP_KP)] for o in outs])
    lip_delta = np.abs(lip_delta - avatar.kp_info.exp.reshape(21, 3)[list(LIP_KP)]).max()
    assert lip_delta < 0.05, f"silence moved the lips by {lip_delta:.3f}"
