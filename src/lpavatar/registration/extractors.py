# SPDX-FileCopyrightText: 2026 lpavatar contributors
# SPDX-License-Identifier: Apache-2.0

"""LivePortrait appearance / motion extractors on the 256 px source crop.

``appearance_extractor.onnx``: ``image`` (1,3,256,256) [0,1] -> ``pred``
(1,32,16,64,64) volumetric feature ``f_s``.

``motion_extractor.onnx``: ``image`` -> ``pitch``/``yaw``/``roll`` (1,66)
head-pose bins, ``t`` (1,3), ``exp`` (1,63), ``scale`` (1,1), ``kp`` (1,63).
Bins are decoded to degrees and the rotation matrix built here so the
returned ``KPInfo`` is ready for the keypoint transform.
"""

from __future__ import annotations

import cv2
import numpy as np

from lpavatar.render.keypoints import KPInfo, bin66_to_degree, rotation_matrix
from lpavatar.runtime.onnx_engine import Engine

FEATURE_SIZE = 256


def to_bchw256(img_crop_rgb: np.ndarray) -> np.ndarray:
    """Any-size RGB crop -> ``(1, 3, 256, 256)`` float32 in [0, 1] (INTER_AREA)."""
    rgb = cv2.resize(img_crop_rgb, (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray((rgb.astype(np.float32) / 255.0)[None].transpose(0, 3, 1, 2))


def extract_appearance(rgb_bchw: np.ndarray, *, engine: Engine) -> np.ndarray:
    return engine(image=rgb_bchw)["pred"].astype(np.float32)


def extract_motion(rgb_bchw: np.ndarray, *, engine: Engine) -> KPInfo:
    out = engine(image=rgb_bchw)
    pitch = bin66_to_degree(out["pitch"])
    yaw = bin66_to_degree(out["yaw"])
    roll = bin66_to_degree(out["roll"])
    return KPInfo(
        kp=out["kp"].reshape(1, 21, 3).astype(np.float32),
        exp=out["exp"].reshape(1, 21, 3).astype(np.float32),
        scale=out["scale"].reshape(1, 1).astype(np.float32),
        t=out["t"].reshape(1, 3).astype(np.float32),
        pitch=pitch.astype(np.float32),
        yaw=yaw.astype(np.float32),
        roll=roll.astype(np.float32),
        rot=rotation_matrix(pitch, yaw, roll).astype(np.float32),
        pitch_bins=out["pitch"].reshape(1, 66).astype(np.float32),
        yaw_bins=out["yaw"].reshape(1, 66).astype(np.float32),
        roll_bins=out["roll"].reshape(1, 66).astype(np.float32),
    )


__all__ = ["FEATURE_SIZE", "extract_appearance", "extract_motion", "to_bchw256"]
