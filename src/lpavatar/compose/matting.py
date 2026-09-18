# SPDX-FileCopyrightText: 2026 lpavatar contributors
# SPDX-License-Identifier: Apache-2.0

"""MODNet portrait matting (Apache-2.0 model) -> per-frame alpha.

Follows the official MODNet ONNX inference recipe: resize so the longer
side is ``ref_size`` and both sides are multiples of 32, normalise to
``[-1, 1]``, run, resize the matte back to the frame size.

The ONNX input/output names differ between MODNet exports, so they are
discovered from the graph instead of hard-coded.
"""

from __future__ import annotations

import cv2
import numpy as np

from lpavatar.runtime.onnx_engine import Engine


def _fit_size(h: int, w: int, ref_size: int) -> tuple[int, int]:
    if max(h, w) < ref_size or min(h, w) > ref_size:
        if w >= h:
            rw = ref_size
            rh = int(h / w * ref_size)
        else:
            rh = ref_size
            rw = int(w / h * ref_size)
    else:
        rh, rw = h, w
    rw = max(32, rw - rw % 32)
    rh = max(32, rh - rh % 32)
    return rh, rw


class ModnetMatting:
    def __init__(self, engine: Engine, *, ref_size: int = 512) -> None:
        if len(engine.inputs) != 1 or len(engine.outputs) != 1:
            raise ValueError("MODNet graph must have exactly one input and one output")
        self.engine = engine
        self.input_name = next(iter(engine.inputs))
        self.output_name = next(iter(engine.outputs))
        self.ref_size = ref_size

    def preprocess(self, frame_rgb: np.ndarray) -> np.ndarray:
        h, w = frame_rgb.shape[:2]
        rh, rw = _fit_size(h, w, self.ref_size)
        img = cv2.resize(frame_rgb, (rw, rh), interpolation=cv2.INTER_AREA)
        x = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

    def __call__(self, frame_rgb: np.ndarray) -> np.ndarray:
        """``(H, W)`` float32 alpha in [0, 1]."""
        h, w = frame_rgb.shape[:2]
        matte = self.engine(**{self.input_name: self.preprocess(frame_rgb)})[self.output_name]
        matte = np.asarray(matte, dtype=np.float32).reshape(matte.shape[-2], matte.shape[-1])
        return np.clip(cv2.resize(matte, (w, h), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)


__all__ = ["ModnetMatting"]
