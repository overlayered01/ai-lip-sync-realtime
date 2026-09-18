# SPDX-FileCopyrightText: 2026 lpavatar contributors
# SPDX-License-Identifier: Apache-2.0

"""Keypoints -> 512 px face: stitching -> warping -> SPADE decoder.

ONNX graphs (LivePortrait via ditto-talkinghead):

- ``stitch_network.onnx``: ``kp_source`` (1,21,3), ``kp_driving`` (1,21,3)
  -> ``out`` (1,21,3) stitched driving keypoints (the graph already adds
  the predicted delta to ``kp_driving``).
- ``warp_network.onnx``: ``feature_3d`` (1,32,16,64,64), ``kp_source``,
  ``kp_driving`` -> ``out`` (1,256,64,64) warped 2-D feature.
- ``decoder.onnx``: ``feature`` (1,256,64,64) -> ``output`` (1,3,512,512)
  RGB in [0, 1].

This module is deliberately per-frame and numpy-based. Batched TensorRT
execution for 25 fps is a separate optimisation step (R6).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lpavatar.runtime.onnx_engine import Engine


@dataclass(slots=True)
class RenderEngines:
    stitch: Engine
    warp: Engine
    decoder: Engine


def stitch(x_s: np.ndarray, x_d: np.ndarray, *, engine: Engine) -> np.ndarray:
    """Seam-reducing refinement of one driving frame. ``(1, 21, 3)`` in and out."""
    out = engine(
        kp_source=np.asarray(x_s, dtype=np.float32).reshape(1, 21, 3),
        kp_driving=np.asarray(x_d, dtype=np.float32).reshape(1, 21, 3),
    )
    return out["out"].reshape(1, 21, 3).astype(np.float32)


def warp(f_s: np.ndarray, x_s: np.ndarray, x_d: np.ndarray, *, engine: Engine) -> np.ndarray:
    out = engine(
        feature_3d=np.asarray(f_s, dtype=np.float32),
        kp_source=np.asarray(x_s, dtype=np.float32).reshape(1, 21, 3),
        kp_driving=np.asarray(x_d, dtype=np.float32).reshape(1, 21, 3),
    )
    return out["out"]


def decode(feature: np.ndarray, *, engine: Engine) -> np.ndarray:
    """``(1,256,64,64)`` -> ``(512, 512, 3)`` float32 RGB in [0, 255]."""
    out = engine(feature=np.asarray(feature, dtype=np.float32))["output"]
    img = np.transpose(out[0], (1, 2, 0))
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.float32)


def render_face(
    f_s: np.ndarray,
    x_s: np.ndarray,
    x_d: np.ndarray,
    *,
    engines: RenderEngines,
    use_stitch: bool = True,
) -> np.ndarray:
    """One frame: ``(512, 512, 3)`` float32 RGB in [0, 255]."""
    if use_stitch:
        x_d = stitch(x_s, x_d, engine=engines.stitch)
    feat = warp(f_s, x_s, x_d, engine=engines.warp)
    return decode(feat, engine=engines.decoder)


def load_render_engines(
    device: str = "cpu", root: Path | None = None, *, fp16: bool = True
) -> RenderEngines:
    """Engines for ``device`` with the right graph variants (see
    ``lpavatar.artifacts.render_model_paths``). On CUDA the sessions use graph
    capture, so keep passing the same tensor shapes."""
    from lpavatar.artifacts import render_model_paths
    from lpavatar.runtime.onnx_engine import OnnxEngine

    p = render_model_paths(device, root, fp16=fp16)
    return RenderEngines(
        stitch=OnnxEngine(p["stitch"], device=device),
        warp=OnnxEngine(p["warp"], device=device),
        decoder=OnnxEngine(p["decoder"], device=device),
    )


__all__ = ["RenderEngines", "decode", "load_render_engines", "render_face", "stitch", "warp"]
