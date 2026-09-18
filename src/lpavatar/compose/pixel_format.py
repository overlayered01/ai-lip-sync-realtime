# SPDX-FileCopyrightText: 2026 lpavatar contributors
# SPDX-License-Identifier: Apache-2.0

"""Frame pixel-format conversion for local display / encoder hand-off."""

from __future__ import annotations

import cv2
import numpy as np


def rgb_to_bgr(frame_rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(frame_rgb[:, :, ::-1])


def rgb_to_i420(frame_rgb: np.ndarray) -> np.ndarray:
    """Planar YUV 4:2:0 (BT.601 limited range), shape ``(H * 3 // 2, W)`` uint8.

    ``H`` and ``W`` must be even.
    """
    h, w = frame_rgb.shape[:2]
    if h % 2 or w % 2:
        raise ValueError(f"I420 needs even dimensions, got {h}x{w}")
    return cv2.cvtColor(np.ascontiguousarray(frame_rgb), cv2.COLOR_RGB2YUV_I420)


def i420_to_rgb(frame_i420: np.ndarray, width: int) -> np.ndarray:
    return cv2.cvtColor(frame_i420.reshape(-1, width), cv2.COLOR_YUV2RGB_I420)


__all__ = ["i420_to_rgb", "rgb_to_bgr", "rgb_to_i420"]
