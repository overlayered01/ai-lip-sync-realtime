# SPDX-FileCopyrightText: 2024 Ant Group Co., Ltd. (ditto-talkinghead)
# SPDX-FileCopyrightText: 2026 lpavatar contributors
# SPDX-License-Identifier: Apache-2.0
#
# Follows ditto-talkinghead ``core/utils/get_mask.py`` (Apache-2.0).

"""Feathered paste-back mask for the 512 px face crop.

Ones in the central ``ratio`` box, linear falloff to the edges, radial
falloff in the corners. ``(H, W)`` float32 in [0, 1].
"""

from __future__ import annotations

import numpy as np


def feathered_mask(
    width: int = 512, height: int = 512, ratio_w: float = 0.9, ratio_h: float = 0.9
) -> np.ndarray:
    w = int(width * ratio_w)
    h = int(height * ratio_h)
    x1 = (width - w) // 2
    x2 = x1 + w
    y1 = (height - h) // 2
    y2 = y1 + h

    mask = np.ones((height, width), dtype=np.float32)
    top = np.linspace(0.0, 1.0, y1, dtype=np.float32)[:, None]
    bottom = np.linspace(1.0, 0.0, height - y2, dtype=np.float32)[:, None]
    left = np.linspace(0.0, 1.0, x1, dtype=np.float32)[None, :]
    right = np.linspace(1.0, 0.0, width - x2, dtype=np.float32)[None, :]

    mask[0:y1, x1:x2] = np.broadcast_to(top, (y1, w))
    mask[y2:height, x1:x2] = np.broadcast_to(bottom, (height - y2, w))
    mask[y1:y2, 0:x1] = np.broadcast_to(left, (h, x1))
    mask[y1:y2, x2:width] = np.broadcast_to(right, (h, width - x2))

    def corner(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return 1.0 - np.clip(np.sqrt(rows**2 + cols**2), 0.0, 1.0).astype(np.float32)

    mask[0:y1, 0:x1] = corner(np.linspace(1, 0, y1)[:, None], np.linspace(1, 0, x1)[None, :])
    mask[0:y1, x2:width] = corner(
        np.linspace(1, 0, y1)[:, None], np.linspace(0, 1, width - x2)[None, :]
    )
    mask[y2:height, 0:x1] = corner(
        np.linspace(0, 1, height - y2)[:, None], np.linspace(1, 0, x1)[None, :]
    )
    mask[y2:height, x2:width] = corner(
        np.linspace(0, 1, height - y2)[:, None], np.linspace(0, 1, width - x2)[None, :]
    )
    return mask


__all__ = ["feathered_mask"]
