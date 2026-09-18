# SPDX-FileCopyrightText: 2024 Ant Group Co., Ltd. (ditto-talkinghead)
# SPDX-FileCopyrightText: 2026 ai-lip-sync contributors
# SPDX-License-Identifier: Apache-2.0
#
# Follows ``_cvt_LP_motion_info`` in ditto ``core/atomic_components/audio2motion.py``.

"""265-d LMDM motion vector <-> LivePortrait motion parameters.

Layout::

    [scale-1 (1) | pitch (66) | yaw (66) | roll (66) | t (3) | exp (63)]

``pitch`` / ``yaw`` / ``roll`` are the raw 66-bin head-pose logits (decode
with ``bin66_to_degree``). ``kp`` (the canonical keypoints) is *not* part of
the vector; it is the ``sc`` condition instead. The first 202 dims
(scale..t) are what ditto smooths and carries in ``kp_cond`` across clips.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from lpavatar.render.keypoints import KPInfo

MOTION_DIM = 265
POSE_DIMS = 202  # scale + pitch + yaw + roll + t
SL_SCALE = slice(0, 1)
SL_PITCH = slice(1, 67)
SL_YAW = slice(67, 133)
SL_ROLL = slice(133, 199)
SL_T = slice(199, 202)
SL_EXP = slice(202, 265)


@dataclass(slots=True)
class MotionInfo:
    """One frame of LivePortrait motion, batch dim kept at 1 (ditto ``x_d_info``)."""

    scale: np.ndarray  # (1, 1)
    pitch: np.ndarray  # (1, 66) logits
    yaw: np.ndarray  # (1, 66)
    roll: np.ndarray  # (1, 66)
    t: np.ndarray  # (1, 3)
    exp: np.ndarray  # (1, 63)

    def copy(self) -> MotionInfo:
        return replace(
            self,
            scale=self.scale.copy(),
            pitch=self.pitch.copy(),
            yaw=self.yaw.copy(),
            roll=self.roll.copy(),
            t=self.t.copy(),
            exp=self.exp.copy(),
        )


def motion_to_vector(info: MotionInfo) -> np.ndarray:
    """``MotionInfo`` -> ``(265,)`` float32 (``scale`` stored as ``scale - 1``)."""
    v = np.empty(MOTION_DIM, dtype=np.float32)
    v[SL_SCALE] = np.asarray(info.scale, np.float32).reshape(1) - 1.0
    v[SL_PITCH] = np.asarray(info.pitch, np.float32).reshape(66)
    v[SL_YAW] = np.asarray(info.yaw, np.float32).reshape(66)
    v[SL_ROLL] = np.asarray(info.roll, np.float32).reshape(66)
    v[SL_T] = np.asarray(info.t, np.float32).reshape(3)
    v[SL_EXP] = np.asarray(info.exp, np.float32).reshape(63)
    return v


def vector_to_motion(vec: np.ndarray) -> MotionInfo:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.shape[0] < MOTION_DIM:
        raise ValueError(f"motion vector needs {MOTION_DIM} dims, got {v.shape[0]}")
    return MotionInfo(
        scale=(v[SL_SCALE] + 1.0).reshape(1, 1),
        pitch=v[SL_PITCH].reshape(1, 66).copy(),
        yaw=v[SL_YAW].reshape(1, 66).copy(),
        roll=v[SL_ROLL].reshape(1, 66).copy(),
        t=v[SL_T].reshape(1, 3).copy(),
        exp=v[SL_EXP].reshape(1, 63).copy(),
    )


def motion_from_kpinfo(info: KPInfo) -> MotionInfo:
    """Source motion from the registered avatar. Needs the raw pose logits."""
    if info.pitch_bins is None or info.yaw_bins is None or info.roll_bins is None:
        raise ValueError(
            "KPInfo has no head-pose logits; re-register the avatar with the current "
            "lpavatar (extract_motion now keeps pitch_bins / yaw_bins / roll_bins)"
        )
    return MotionInfo(
        scale=np.asarray(info.scale, np.float32).reshape(1, 1),
        pitch=np.asarray(info.pitch_bins, np.float32).reshape(1, 66),
        yaw=np.asarray(info.yaw_bins, np.float32).reshape(1, 66),
        roll=np.asarray(info.roll_bins, np.float32).reshape(1, 66),
        t=np.asarray(info.t, np.float32).reshape(1, 3),
        exp=np.asarray(info.exp, np.float32).reshape(1, 63),
    )


def motion_from_dict(d: dict[str, np.ndarray]) -> MotionInfo:
    """From a ditto ``x_s_info`` dict (as stored in the cfg pickle's ``ch_info``)."""
    return MotionInfo(
        scale=np.asarray(d["scale"], np.float32).reshape(1, 1),
        pitch=np.asarray(d["pitch"], np.float32).reshape(1, 66),
        yaw=np.asarray(d["yaw"], np.float32).reshape(1, 66),
        roll=np.asarray(d["roll"], np.float32).reshape(1, 66),
        t=np.asarray(d["t"], np.float32).reshape(1, 3),
        exp=np.asarray(d["exp"], np.float32).reshape(1, 63),
    )


__all__ = [
    "MOTION_DIM",
    "POSE_DIMS",
    "SL_EXP",
    "SL_PITCH",
    "SL_ROLL",
    "SL_SCALE",
    "SL_T",
    "SL_YAW",
    "MotionInfo",
    "motion_from_dict",
    "motion_from_kpinfo",
    "motion_to_vector",
    "vector_to_motion",
]
