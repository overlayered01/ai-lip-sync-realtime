# SPDX-FileCopyrightText: 2024 Kuaishou Visual Generation and Interaction Center (LivePortrait)
# SPDX-FileCopyrightText: 2026 lpavatar contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Head-pose decoding and the implicit-keypoint transform follow LivePortrait
# ``src/utils/camera.py`` (``headpose_pred_to_degree``, ``get_rotation_matrix``)
# and ``src/live_portrait_wrapper.py`` (``transform_keypoint``), MIT.

"""Implicit keypoint math (LivePortrait Eqn. 2).

    x = scale * (kp @ R + exp) + t[:, :2]

``kp`` are the canonical 3-D keypoints of the source portrait, ``R`` the
head rotation, ``exp`` the per-keypoint expression deltas, ``t`` the
translation (z is ignored). Source keypoints ``x_s`` use the source's own
pose/expression; driving keypoints ``x_d`` reuse the source ``kp``,
``scale`` and ``t`` but take ``R`` and ``exp`` from the motion generator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NUM_KP = 21
NUM_EXP = NUM_KP * 3  # 63


@dataclass(slots=True, frozen=True)
class KPInfo:
    """Source-side pose / keypoint bundle, batch dim kept at 1."""

    kp: np.ndarray  # (1, 21, 3)
    exp: np.ndarray  # (1, 21, 3)
    scale: np.ndarray  # (1, 1)
    t: np.ndarray  # (1, 3)
    pitch: np.ndarray  # (1,) degrees
    yaw: np.ndarray  # (1,)
    roll: np.ndarray  # (1,)
    rot: np.ndarray  # (1, 3, 3)
    # Raw 66-bin head-pose logits from the motion extractor. Optional: only
    # motion models that work on the logits (ditto LMDM) need them.
    pitch_bins: np.ndarray | None = None  # (1, 66)
    yaw_bins: np.ndarray | None = None  # (1, 66)
    roll_bins: np.ndarray | None = None  # (1, 66)


def bin66_to_degree(pred: np.ndarray) -> np.ndarray:
    """Soft-argmax over 66 head-pose bins (3 deg each) -> degrees ``(B,)``."""
    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim == 2 and pred.shape[1] == 66:
        z = pred - pred.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        idx = np.arange(66, dtype=np.float32)
        return (p * idx).sum(axis=1) * 3.0 - 97.5
    return pred.reshape(-1)


def rotation_matrix(pitch_deg: np.ndarray, yaw_deg: np.ndarray, roll_deg: np.ndarray) -> np.ndarray:
    """``(B, 3, 3)``; LivePortrait convention ``(Rz Ry Rx)^T`` for row-vector points."""
    p = np.deg2rad(np.asarray(pitch_deg, dtype=np.float32).reshape(-1))
    y = np.deg2rad(np.asarray(yaw_deg, dtype=np.float32).reshape(-1))
    r = np.deg2rad(np.asarray(roll_deg, dtype=np.float32).reshape(-1))
    b = p.shape[0]
    one = np.ones(b, dtype=np.float32)
    zero = np.zeros(b, dtype=np.float32)

    rot_x = np.stack(
        [one, zero, zero, zero, np.cos(p), -np.sin(p), zero, np.sin(p), np.cos(p)], axis=1
    ).reshape(b, 3, 3)
    rot_y = np.stack(
        [np.cos(y), zero, np.sin(y), zero, one, zero, -np.sin(y), zero, np.cos(y)], axis=1
    ).reshape(b, 3, 3)
    rot_z = np.stack(
        [np.cos(r), -np.sin(r), zero, np.sin(r), np.cos(r), zero, zero, zero, one], axis=1
    ).reshape(b, 3, 3)
    rot = rot_z @ rot_y @ rot_x
    return np.transpose(rot, (0, 2, 1)).astype(np.float32)


def transform_keypoints(
    kp: np.ndarray, rot: np.ndarray, exp: np.ndarray, scale: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """``(B, 21, 3)`` = ``scale * (kp @ rot + exp) + t[:, :2]``.

    ``kp`` may be ``(1, 21, 3)`` and broadcast against a batch of ``rot`` /
    ``exp`` (one source, many driving frames).
    """
    kp = np.asarray(kp, dtype=np.float32).reshape(-1, NUM_KP, 3)
    rot = np.asarray(rot, dtype=np.float32).reshape(-1, 3, 3)
    exp = np.asarray(exp, dtype=np.float32).reshape(-1, NUM_KP, 3)
    scale = np.asarray(scale, dtype=np.float32).reshape(-1, 1, 1)
    t = np.asarray(t, dtype=np.float32).reshape(-1, 3)
    out = (kp @ rot + exp) * scale
    out = out.copy()
    out[:, :, :2] += t[:, None, :2]
    return out.astype(np.float32)


def source_keypoints(info: KPInfo) -> np.ndarray:
    return transform_keypoints(info.kp, info.rot, info.exp, info.scale, info.t)


def driving_keypoints(info: KPInfo, rot_d: np.ndarray, exp_d: np.ndarray) -> np.ndarray:
    """Driving keypoints for ``N`` motion frames: ``rot_d`` (N,3,3), ``exp_d`` (N,21,3)
    or (N,63). Source ``kp`` / ``scale`` / ``t`` are reused."""
    exp_d = np.asarray(exp_d, dtype=np.float32).reshape(-1, NUM_KP, 3)
    return transform_keypoints(info.kp, rot_d, exp_d, info.scale, info.t)


__all__ = [
    "NUM_EXP",
    "NUM_KP",
    "KPInfo",
    "bin66_to_degree",
    "driving_keypoints",
    "rotation_matrix",
    "source_keypoints",
    "transform_keypoints",
]
