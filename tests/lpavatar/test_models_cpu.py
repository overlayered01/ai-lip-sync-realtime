# SPDX-FileCopyrightText: 2026 lpavatar contributors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end CPU checks with the real public ONNX graphs.

Skipped unless ``LPAVATAR_MODELS`` points at a directory holding the graphs
listed in ``lpavatar.artifacts.PUBLIC_MODELS`` and ``LPAVATAR_TEST_PORTRAIT``
points at an RGB portrait. Run ``python -m lpavatar.artifacts`` style
download via ``ensure_models()`` first.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from lpavatar.artifacts import PUBLIC_MODELS

_ROOT = os.environ.get("LPAVATAR_MODELS")
_PORTRAIT = os.environ.get("LPAVATAR_TEST_PORTRAIT")


def _have_models() -> bool:
    return bool(_ROOT) and all((Path(_ROOT) / f"{n}.onnx").is_file() for n in PUBLIC_MODELS)


pytestmark = pytest.mark.skipif(
    not (_have_models() and _PORTRAIT and Path(_PORTRAIT).is_file()),
    reason="set LPAVATAR_MODELS (with all public graphs) and LPAVATAR_TEST_PORTRAIT",
)


@pytest.fixture(scope="module")
def engines():
    from lpavatar.registration.avatar import RegistrationEngines
    from lpavatar.render.stage import RenderEngines
    from lpavatar.runtime import OnnxEngine

    root = Path(_ROOT)  # type: ignore[arg-type]
    e = lambda n: OnnxEngine(root / f"{n}.onnx", device="cpu")  # noqa: E731
    reg = RegistrationEngines(
        blaze_face=e("blaze_face"),
        face_mesh=e("face_mesh"),
        landmark203=e("landmark203"),
        appearance_extractor=e("appearance_extractor"),
        motion_extractor=e("motion_extractor"),
    )
    ren = RenderEngines(
        stitch=e("stitch_network"), warp=e("warp_network_ori"), decoder=e("decoder")
    )
    return reg, ren


@pytest.fixture(scope="module")
def avatar(engines):
    from lpavatar.registration.avatar import AvatarRegistrar, RegistrationConfig

    reg, _ = engines
    registrar = AvatarRegistrar(reg, RegistrationConfig(out_size=(720, 1280)))
    return registrar.register(_PORTRAIT, avatar_id="test")  # type: ignore[arg-type]


def test_registration_shapes_and_pose(avatar):
    assert avatar.frame.shape == (720, 1280, 3) and avatar.frame.dtype == np.uint8
    assert avatar.crop_512.shape == (512, 512, 3)
    assert avatar.lmk203.shape == (203, 2)
    assert avatar.f_s.shape == (1, 32, 16, 64, 64)
    assert avatar.x_s.shape == (1, 21, 3)
    assert avatar.mask_frame.shape == (720, 1280)
    k = avatar.kp_info
    assert k.kp.shape == (1, 21, 3) and k.exp.shape == (1, 21, 3)
    assert abs(float(k.pitch[0])) < 45 and abs(float(k.yaw[0])) < 45 and abs(float(k.roll[0])) < 45
    assert np.allclose(k.rot[0] @ k.rot[0].T, np.eye(3), atol=1e-4)
    # 203 landmarks must lie inside the 512 crop region of the frame
    from lpavatar.registration.crop import transform_pts

    in_crop = transform_pts(avatar.lmk203, np.linalg.inv(avatar.m_c2o))
    assert in_crop.min() > 0 and in_crop.max() < 512


def test_self_reconstruction_matches_source_crop(avatar, engines):
    """Driving = source keypoints must reproduce the source crop (LivePortrait identity)."""
    from lpavatar.render.stage import render_face, stitch

    _, ren = engines
    x_d_st = stitch(avatar.x_s, avatar.x_s, engine=ren.stitch)
    assert np.abs(x_d_st - avatar.x_s).max() < 0.05  # stitching delta ~0 for identity
    face = render_face(avatar.f_s, avatar.x_s, avatar.x_s, engines=ren)
    assert face.shape == (512, 512, 3)
    src = avatar.crop_512.astype(np.float32)
    mse = float(np.mean((face - src) ** 2))
    psnr = 10 * np.log10(255.0**2 / max(mse, 1e-6))
    assert psnr > 20.0, f"self-reconstruction PSNR {psnr:.2f} dB too low"


def test_pasteback_into_frame(avatar, engines):
    from lpavatar.compose.pasteback import pasteback
    from lpavatar.render.stage import render_face

    _, ren = engines
    face = render_face(avatar.f_s, avatar.x_s, avatar.x_s, engines=ren)
    out = pasteback(avatar.frame, face, avatar.m_c2o, avatar.mask_frame)
    assert out.shape == avatar.frame.shape and out.dtype == np.uint8
    # outside the mask the frame is untouched
    untouched = avatar.mask_frame == 0
    assert np.array_equal(out[untouched], avatar.frame[untouched])
    # inside, the result should be close to the original (identity motion)
    inside = avatar.mask_frame > 0.99
    diff = np.abs(out[inside].astype(np.float32) - avatar.frame[inside].astype(np.float32)).mean()
    assert diff < 25.0
