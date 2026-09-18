# SPDX-FileCopyrightText: 2024 Ant Group Co., Ltd. (ditto-talkinghead)
# SPDX-FileCopyrightText: 2026 lpavatar contributors
# SPDX-License-Identifier: Apache-2.0
#
# Follows ditto-talkinghead ``core/atomic_components/putback.py`` (Apache-2.0).

"""Paste the rendered 512 px face back into the full frame.

``pasteback`` is the stateless reference (one call = one frame). ``PasteBack``
caches the frame-space inverse mask and is what the real-time path uses: it
does the warp on uint8 and the blend with ``cv2.blendLinear`` (SIMD, multi-
threaded), about 8x faster than the float32 numpy blend on a 720p frame.
"""

from __future__ import annotations

import cv2
import numpy as np


def _face_u8(face_rgb: np.ndarray) -> np.ndarray:
    if face_rgb.dtype == np.uint8:
        return face_rgb
    return np.clip(face_rgb, 0.0, 255.0).astype(np.uint8)  # truncation matches ditto's cast


class PasteBack:
    """Frame-space compositor for one avatar: ``PasteBack(frame, m_c2o, mask_frame)(face)``."""

    def __init__(self, frame_rgb: np.ndarray, m_c2o: np.ndarray, mask_frame: np.ndarray) -> None:
        self.frame = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
        self.h, self.w = self.frame.shape[:2]
        self.m = np.ascontiguousarray(m_c2o[:2, :], dtype=np.float64)
        self.mask = np.ascontiguousarray(mask_frame, dtype=np.float32)
        self.inv_mask = np.ascontiguousarray(1.0 - self.mask, dtype=np.float32)

    def __call__(self, face_rgb: np.ndarray) -> np.ndarray:
        """``face_rgb`` (512,512,3) float [0,255] or uint8 -> (H,W,3) uint8 RGB."""
        warped = cv2.warpAffine(
            _face_u8(face_rgb), self.m, (self.w, self.h), flags=cv2.INTER_LINEAR
        )
        return cv2.blendLinear(warped, self.frame, self.mask, self.inv_mask)


def pasteback(
    frame_rgb: np.ndarray,
    face_rgb: np.ndarray,
    m_c2o: np.ndarray,
    mask_frame: np.ndarray,
) -> np.ndarray:
    """``frame_rgb`` (H,W,3) uint8, ``face_rgb`` (512,512,3) float/uint8 in
    [0,255], ``m_c2o`` (3,3) crop->frame, ``mask_frame`` (H,W) float32 already
    warped into frame space. Returns (H,W,3) uint8."""
    return PasteBack(frame_rgb, m_c2o, mask_frame)(face_rgb)


def composite_alpha(fg_rgb: np.ndarray, alpha: np.ndarray, bg_rgb: np.ndarray) -> np.ndarray:
    """``fg * alpha + bg * (1 - alpha)``; ``alpha`` (H,W) in [0,1]. uint8 out."""
    a = np.asarray(alpha, dtype=np.float32)[:, :, None]
    out = fg_rgb.astype(np.float32) * a + bg_rgb.astype(np.float32) * (1.0 - a)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


__all__ = ["PasteBack", "composite_alpha", "pasteback"]
