# SPDX-FileCopyrightText: 2026 lpavatar contributors
# SPDX-License-Identifier: Apache-2.0

"""Portrait -> ``Avatar``: everything the render loop needs per portrait.

Cascade (all one-time, CPU is fine)::

    portrait --composite alpha over solid bg--> frame (H, W, 3) uint8
      --BlazeFace + Face Mesh--> 478 pts
      --crop 224 (scale 1.5, vy -0.1)--> landmark203 --> 203 pts (frame px)
      --crop 512 (scale 2.3, vy -0.125, rotated)--> crop_512, m_c2o
      --resize 256--> appearance extractor -> f_s ; motion extractor -> kp_info
      --transform--> x_s (source implicit keypoints)
      --warp feathered mask by m_c2o--> mask_frame (H, W)

Crop parameters default to LivePortrait's image-mode values, which are
also ditto-talkinghead's (``crop_scale=2.3, crop_vy_ratio=-0.125``, rotated
crop). If a motion model ever expects a different convention, change
``RegistrationConfig`` rather than this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from lpavatar.registration.crop import crop_image
from lpavatar.registration.extractors import extract_appearance, extract_motion, to_bchw256
from lpavatar.registration.face import detect_landmarks478
from lpavatar.registration.landmark203 import LM203_SIZE, landmark203
from lpavatar.registration.mask import feathered_mask
from lpavatar.render.keypoints import KPInfo, source_keypoints
from lpavatar.runtime.onnx_engine import Engine


@dataclass(slots=True, frozen=True)
class RegistrationConfig:
    out_size: tuple[int, int] = (720, 1280)  # (H, W) frame the avatar is rendered into
    seed_scale: float = 1.5
    seed_vy_ratio: float = -0.1
    crop_size: int = 512
    crop_scale: float = 2.3
    crop_vx_ratio: float = 0.0
    crop_vy_ratio: float = -0.125
    crop_rotate: bool = True
    background_rgb: tuple[int, int, int] = (200, 200, 200)  # for RGBA portraits
    mask_ratio: tuple[float, float] = (0.9, 0.9)


@dataclass(slots=True, frozen=True)
class Avatar:
    id: str
    frame: np.ndarray  # (H, W, 3) uint8 RGB -- paste-back target
    has_alpha: bool  # True: portrait was RGBA, matting is needed downstream
    crop_512: np.ndarray  # (512, 512, 3) uint8 RGB
    m_c2o: np.ndarray  # (3, 3) crop -> frame affine
    lmk203: np.ndarray  # (203, 2) in frame px
    kp_info: KPInfo
    x_s: np.ndarray  # (1, 21, 3) transformed source keypoints
    f_s: np.ndarray  # (1, 32, 16, 64, 64) appearance feature volume
    mask_frame: np.ndarray  # (H, W) float32 paste-back mask in frame space

    @property
    def size(self) -> tuple[int, int]:
        return self.frame.shape[0], self.frame.shape[1]


@dataclass(slots=True)
class RegistrationEngines:
    blaze_face: Engine
    face_mesh: Engine
    landmark203: Engine
    appearance_extractor: Engine
    motion_extractor: Engine


def load_portrait(path: str | Path) -> np.ndarray:
    """RGB or RGBA uint8 HWC."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def fit_to_frame(img: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
    """Resize to exactly ``(H, W)``; aspect is the caller's responsibility."""
    h, w = out_size
    if img.shape[0] == h and img.shape[1] == w:
        return img
    interp = cv2.INTER_AREA if img.shape[0] > h else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp)


def composite_over(img: np.ndarray, rgb: tuple[int, int, int]) -> tuple[np.ndarray, bool]:
    """RGBA -> RGB over a solid colour. Returns ``(rgb_uint8, had_alpha)``."""
    if img.shape[2] == 3:
        return np.ascontiguousarray(img), False
    alpha = img[:, :, 3:4].astype(np.float32) / 255.0
    bg = np.array(rgb, dtype=np.float32)[None, None, :]
    out = img[:, :, :3].astype(np.float32) * alpha + bg * (1.0 - alpha)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), True


@dataclass(slots=True)
class AvatarRegistrar:
    engines: RegistrationEngines
    config: RegistrationConfig = field(default_factory=RegistrationConfig)
    mask_template: np.ndarray | None = None  # (512, 512) float32; default feathered

    def _mask(self) -> np.ndarray:
        if self.mask_template is not None:
            return self.mask_template
        rw, rh = self.config.mask_ratio
        return feathered_mask(self.config.crop_size, self.config.crop_size, rw, rh)

    def register_image(self, img: np.ndarray, *, avatar_id: str) -> Avatar:
        """``img``: RGB or RGBA uint8 HWC, any size."""
        cfg = self.config
        frame, has_alpha = composite_over(fit_to_frame(img, cfg.out_size), cfg.background_rgb)

        pts478 = detect_landmarks478(
            frame, det=self.engines.blaze_face, mesh=self.engines.face_mesh
        )
        seed = crop_image(
            frame, pts478, dsize=LM203_SIZE, scale=cfg.seed_scale, vy_ratio=cfg.seed_vy_ratio
        )
        lmk203 = landmark203(seed.img_crop, engine=self.engines.landmark203, m_c2o=seed.m_c2o)
        crop = crop_image(
            frame,
            lmk203,
            dsize=cfg.crop_size,
            scale=cfg.crop_scale,
            vx_ratio=cfg.crop_vx_ratio,
            vy_ratio=cfg.crop_vy_ratio,
            rotate=cfg.crop_rotate,
        )

        rgb256 = to_bchw256(crop.img_crop)
        f_s = extract_appearance(rgb256, engine=self.engines.appearance_extractor)
        kp_info = extract_motion(rgb256, engine=self.engines.motion_extractor)
        x_s = source_keypoints(kp_info)

        h, w = frame.shape[:2]
        mask_frame = cv2.warpAffine(
            self._mask(), crop.m_c2o[:2, :], (w, h), flags=cv2.INTER_LINEAR
        ).clip(0.0, 1.0)

        return Avatar(
            id=avatar_id,
            frame=frame,
            has_alpha=has_alpha,
            crop_512=crop.img_crop,
            m_c2o=crop.m_c2o,
            lmk203=lmk203.astype(np.float32),
            kp_info=kp_info,
            x_s=x_s,
            f_s=f_s,
            mask_frame=mask_frame.astype(np.float32),
        )

    def register(self, portrait_path: str | Path, *, avatar_id: str | None = None) -> Avatar:
        path = Path(portrait_path)
        return self.register_image(load_portrait(path), avatar_id=avatar_id or path.stem)


__all__ = [
    "Avatar",
    "AvatarRegistrar",
    "RegistrationConfig",
    "RegistrationEngines",
    "composite_over",
    "fit_to_frame",
    "load_portrait",
]
