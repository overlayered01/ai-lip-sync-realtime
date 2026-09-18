# SPDX-FileCopyrightText: 2024 Ant Group Co., Ltd. (ditto-talkinghead)
# SPDX-FileCopyrightText: 2026 lpavatar contributors
# SPDX-License-Identifier: Apache-2.0
#
# Follows ditto-talkinghead ``core/atomic_components/putback.py`` (Apache-2.0).

"""Paste the rendered 512 px face back into the full frame."""

from __future__ import annotations

import cv2
import numpy as np


def pasteback(
    frame_rgb: np.ndarray,
    face_rgb: np.ndarray,
    m_c2o: np.ndarray,
    mask_frame: np.ndarray,
) -> np.ndarray:
    """``frame_rgb`` (H,W,3) uint8, ``face_rgb`` (512,512,3) float/uint8 in
    [0,255], ``m_c2o`` (3,3) crop->frame, ``mask_frame`` (H,W) float32 already
    warped into frame space. Returns (H,W,3) uint8."""
    h, w = frame_rgb.shape[:2]
    face_warped = cv2.warpAffine(
        np.asarray(face_rgb, dtype=np.float32), m_c2o[:2, :], (w, h), flags=cv2.INTER_LINEAR
    )
    m = mask_frame[:, :, None]
    out = m * face_warped + (1.0 - m) * frame_rgb.astype(np.float32)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def composite_alpha(fg_rgb: np.ndarray, alpha: np.ndarray, bg_rgb: np.ndarray) -> np.ndarray:
    """``fg * alpha + bg * (1 - alpha)``; ``alpha`` (H,W) in [0,1]. uint8 out."""
    a = np.asarray(alpha, dtype=np.float32)[:, :, None]
    out = fg_rgb.astype(np.float32) * a + bg_rgb.astype(np.float32) * (1.0 - a)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


__all__ = ["composite_alpha", "pasteback"]
