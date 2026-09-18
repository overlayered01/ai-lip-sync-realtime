# SPDX-FileCopyrightText: 2024 Ant Group Co., Ltd. (ditto-talkinghead)
# SPDX-FileCopyrightText: 2026 ai-lip-sync contributors
# SPDX-License-Identifier: Apache-2.0

"""Audio -> motion for a live avatar: a port of the ditto-talkinghead online pipeline.

Modules (all numpy, engines behind ``lpavatar.runtime.Engine``)::

    hubert.py        6,480-sample window -> 5 HuBERT features at 25 Hz
    condition.py     [hubert 1024 | emo 8 | eye_open 2 | eye_ball 6 | sc 63] = 1103
    vector.py        265-d motion vector <-> (scale, pitch, yaw, roll, t, exp)
    lmdm.py          DDIM sampler around the LMDM ONNX/TensorRT graph
    ditto_motion.py  online state machine: windows in, driving keypoints out
                     (+ rewind for barge-in, per-frame controls, retargeting)

Timing (see docs/DITTO_AVATAR_HANDOFF.md §11): with ``overlap_v2=75`` the
first frame of every 5-frame block needs audio 12 frames (480 ms) ahead;
with ditto's shipped ``overlap_v2=70`` it is 22 frames (880 ms).
"""

from lpavatar.motion.condition import ConditionBuilder, ConditionSource, emo_from_label
from lpavatar.motion.ditto_motion import (
    Audio2MotionOnline,
    DittoMotionStream,
    FrameControl,
    MotionOutput,
    MotionRetarget,
    MotionStreamConfig,
)
from lpavatar.motion.hubert import (
    CHUNK_FUTURE,
    CHUNK_PAST,
    FRAME_SAMPLES,
    HOP_SAMPLES,
    WINDOW_SAMPLES,
    HubertFeatures,
    iter_windows,
)
from lpavatar.motion.lmdm import LMDMSampler
from lpavatar.motion.vector import MOTION_DIM, MotionInfo, motion_to_vector, vector_to_motion

__all__ = [
    "CHUNK_FUTURE",
    "CHUNK_PAST",
    "FRAME_SAMPLES",
    "HOP_SAMPLES",
    "MOTION_DIM",
    "WINDOW_SAMPLES",
    "Audio2MotionOnline",
    "ConditionBuilder",
    "ConditionSource",
    "DittoMotionStream",
    "FrameControl",
    "HubertFeatures",
    "LMDMSampler",
    "MotionInfo",
    "MotionOutput",
    "MotionRetarget",
    "MotionStreamConfig",
    "emo_from_label",
    "iter_windows",
    "motion_to_vector",
    "vector_to_motion",
]
