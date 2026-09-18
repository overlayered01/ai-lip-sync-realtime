# SPDX-FileCopyrightText: 2024 Kuaishou Visual Generation and Interaction Center (LivePortrait)
# SPDX-FileCopyrightText: 2026 lpavatar contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Crop geometry follows LivePortrait ``src/utils/crop.py`` (MIT):
# ``parse_pt2_from_pt*``, ``parse_rect_from_landmark``,
# ``_estimate_similar_transform_from_pts``, ``crop_image``. Rewritten for
# the landmark sets this package uses (MediaPipe 478, LivePortrait 203).

"""Face-aligned square crops from 2-D landmarks.

The crop frame is defined by two anchor points -- the eye centre and the
lip centre -- which fix the in-plane rotation, and by the landmark extent,
which fixes the size. ``scale`` enlarges the box, ``vy_ratio`` shifts it
along the eye->lip axis (negative = include more forehead).

Returned affines are 3x3 in pixel coordinates:
``m_o2c`` maps original -> crop, ``m_c2o`` is its inverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, cos, sin

import cv2
import numpy as np

# MediaPipe Face Mesh indices: eye outer corner, inner corner, upper lid, lower lid.
MP_EYE_A = (33, 133, 159, 145)
MP_EYE_B = (263, 362, 386, 374)
MP_LIP_TOP = 0  # outer upper-lip centre
MP_LIP_BOTTOM = 17  # outer lower-lip centre

# LivePortrait 203-point layout.
LP203_EYE_A = (0, 6, 12, 18)
LP203_EYE_B = (24, 30, 36, 42)
LP203_LIP_TOP = 48
LP203_LIP_BOTTOM = 66


@dataclass(slots=True, frozen=True)
class CropResult:
    img_crop: np.ndarray  # (dsize, dsize, C) same dtype as input
    m_o2c: np.ndarray  # (3, 3) original -> crop
    m_c2o: np.ndarray  # (3, 3) crop -> original


def transform_pts(pts: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Apply a 2x3 / 3x3 affine to ``(N, 2)`` points."""
    return pts @ m[:2, :2].T + m[:2, 2]


def eye_lip_axis(pts: np.ndarray) -> np.ndarray:
    """``(2, 2)`` array: row 0 = eye centre, row 1 = lip centre.

    Dispatches on the landmark count (478 MediaPipe, 203 LivePortrait).
    """
    n = pts.shape[0]
    if n == 478:
        eye_a, eye_b, lip_t, lip_b = MP_EYE_A, MP_EYE_B, MP_LIP_TOP, MP_LIP_BOTTOM
    elif n == 203:
        eye_a, eye_b, lip_t, lip_b = LP203_EYE_A, LP203_EYE_B, LP203_LIP_TOP, LP203_LIP_BOTTOM
    else:
        raise ValueError(f"unsupported landmark count {n}; expected 478 or 203")
    eye = (pts[list(eye_a)].mean(axis=0) + pts[list(eye_b)].mean(axis=0)) / 2.0
    lip = (pts[lip_t] + pts[lip_b]) / 2.0
    return np.stack([eye, lip], axis=0).astype(np.float32)


def aligned_rect(
    pts: np.ndarray, *, scale: float, vx_ratio: float, vy_ratio: float
) -> tuple[np.ndarray, float, float]:
    """Centre, side length and rotation angle of the face-aligned square.

    Returns ``(center (2,), size, angle_rad)``. ``angle`` is the rotation
    of the crop's x-axis in image coordinates (clockwise positive).
    """
    pts = np.asarray(pts, dtype=np.float32)[:, :2]
    axis = eye_lip_axis(pts)
    uy = axis[1] - axis[0]
    norm = float(np.linalg.norm(uy))
    uy = np.array([0.0, 1.0], dtype=np.float32) if norm <= 1e-3 else uy / norm
    ux = np.array([uy[1], -uy[0]], dtype=np.float32)

    angle = acos(float(np.clip(ux[0], -1.0, 1.0)))
    if ux[1] < 0:
        angle = -angle

    rot = np.stack([ux, uy], axis=0)  # rows = new basis
    center0 = pts.mean(axis=0)
    local = (pts - center0) @ rot.T
    lt = local.min(axis=0)
    rb = local.max(axis=0)
    center1 = (lt + rb) / 2.0
    size = float(max(rb - lt)) * scale

    center = center0 + ux * center1[0] + uy * center1[1]
    center = center + ux * (vx_ratio * size) + uy * (vy_ratio * size)
    return center.astype(np.float32), size, angle


def similarity_to_crop(
    pts: np.ndarray,
    *,
    dsize: int,
    scale: float,
    vx_ratio: float,
    vy_ratio: float,
    rotate: bool,
) -> np.ndarray:
    """``(2, 3)`` affine mapping original pixels into a ``dsize`` square."""
    center, size, angle = aligned_rect(pts, scale=scale, vx_ratio=vx_ratio, vy_ratio=vy_ratio)
    s = dsize / size
    tcx = tcy = dsize / 2.0
    cx, cy = float(center[0]), float(center[1])
    if rotate:
        ct, st = cos(angle), sin(angle)
        return np.array(
            [
                [s * ct, s * st, tcx - s * (ct * cx + st * cy)],
                [-s * st, s * ct, tcy - s * (-st * cx + ct * cy)],
            ],
            dtype=np.float32,
        )
    return np.array([[s, 0.0, tcx - s * cx], [0.0, s, tcy - s * cy]], dtype=np.float32)


def crop_image(
    img: np.ndarray,
    pts: np.ndarray,
    *,
    dsize: int,
    scale: float = 1.5,
    vx_ratio: float = 0.0,
    vy_ratio: float = -0.1,
    rotate: bool = True,
    border_value: float | tuple[float, ...] = 0.0,
) -> CropResult:
    """Crop ``img`` (HWC) around ``pts`` into an aligned ``dsize`` square."""
    m = similarity_to_crop(
        pts, dsize=dsize, scale=scale, vx_ratio=vx_ratio, vy_ratio=vy_ratio, rotate=rotate
    )
    img_crop = cv2.warpAffine(
        img, m, (dsize, dsize), flags=cv2.INTER_LINEAR, borderValue=border_value
    )
    m_o2c = np.vstack([m, np.array([0.0, 0.0, 1.0], dtype=np.float32)])
    m_c2o = np.linalg.inv(m_o2c).astype(np.float32)
    return CropResult(img_crop=img_crop, m_o2c=m_o2c, m_c2o=m_c2o)


__all__ = [
    "CropResult",
    "aligned_rect",
    "crop_image",
    "eye_lip_axis",
    "similarity_to_crop",
    "transform_pts",
]
