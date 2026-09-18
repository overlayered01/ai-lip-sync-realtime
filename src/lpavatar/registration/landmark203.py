# SPDX-FileCopyrightText: 2026 lpavatar contributors
# SPDX-License-Identifier: Apache-2.0

"""LivePortrait 203-point landmark model (``landmark203.onnx``, MIT model).

Input ``input`` (1, 3, 224, 224) RGB in [0, 1]; output ``landmarks``
(1, 406) normalised to the 224 crop.
"""

from __future__ import annotations

import numpy as np

from lpavatar.registration.crop import transform_pts
from lpavatar.runtime.onnx_engine import Engine

LM203_SIZE = 224


def landmark203(
    img_crop_rgb: np.ndarray, *, engine: Engine, m_c2o: np.ndarray | None = None
) -> np.ndarray:
    """``(203, 2)`` landmarks; in original-frame pixels when ``m_c2o`` given."""
    if img_crop_rgb.shape[:2] != (LM203_SIZE, LM203_SIZE):
        raise ValueError(f"landmark203 expects a {LM203_SIZE}px crop, got {img_crop_rgb.shape}")
    inp = (img_crop_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    out = engine(input=np.ascontiguousarray(inp))["landmarks"]
    pts = out.reshape(-1, 2).astype(np.float32) * LM203_SIZE
    if m_c2o is not None:
        pts = transform_pts(pts, m_c2o).astype(np.float32)
    return pts


__all__ = ["LM203_SIZE", "landmark203"]
