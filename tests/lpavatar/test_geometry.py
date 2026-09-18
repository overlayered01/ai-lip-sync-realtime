# SPDX-FileCopyrightText: 2026 lpavatar contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for lpavatar geometry: crop, keypoint transform, mask, compose."""

from __future__ import annotations

import numpy as np
import pytest

from lpavatar.compose.pasteback import composite_alpha, pasteback
from lpavatar.compose.pixel_format import i420_to_rgb, rgb_to_bgr, rgb_to_i420
from lpavatar.registration.crop import (
    LP203_EYE_A,
    LP203_EYE_B,
    LP203_LIP_BOTTOM,
    LP203_LIP_TOP,
    MP_EYE_A,
    MP_EYE_B,
    MP_LIP_BOTTOM,
    MP_LIP_TOP,
    aligned_rect,
    crop_image,
    eye_lip_axis,
    transform_pts,
)
from lpavatar.registration.mask import feathered_mask
from lpavatar.render.keypoints import (
    KPInfo,
    bin66_to_degree,
    driving_keypoints,
    rotation_matrix,
    source_keypoints,
    transform_keypoints,
)


def _face_pts(n: int, eye_a, eye_b, lip_t, lip_b, *, cx=300.0, cy=200.0, roll_deg=0.0, w=100.0):
    """Synthetic landmark set: eyes at +-w/2 on the x axis, lips w below."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(-w, w, size=(n, 2)).astype(np.float32)
    for i in eye_a:
        pts[i] = (-w / 2, 0)
    for i in eye_b:
        pts[i] = (w / 2, 0)
    pts[lip_t] = (0, w - 5)
    pts[lip_b] = (0, w + 5)
    th = np.deg2rad(roll_deg)
    r = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float32)
    return (pts @ r.T + (cx, cy)).astype(np.float32)


def test_eye_lip_axis_478_and_203():
    p = _face_pts(478, MP_EYE_A, MP_EYE_B, MP_LIP_TOP, MP_LIP_BOTTOM)
    axis = eye_lip_axis(p)
    assert np.allclose(axis[0], (300, 200), atol=1e-4)
    assert np.allclose(axis[1], (300, 300), atol=1e-4)
    p = _face_pts(203, LP203_EYE_A, LP203_EYE_B, LP203_LIP_TOP, LP203_LIP_BOTTOM)
    assert np.allclose(eye_lip_axis(p)[1], (300, 300), atol=1e-4)
    with pytest.raises(ValueError):
        eye_lip_axis(np.zeros((106, 2)))


def test_aligned_rect_recovers_roll():
    for roll in (0.0, 15.0, -30.0):
        p = _face_pts(478, MP_EYE_A, MP_EYE_B, MP_LIP_TOP, MP_LIP_BOTTOM, roll_deg=roll)
        _, size, angle = aligned_rect(p, scale=1.0, vx_ratio=0.0, vy_ratio=0.0)
        assert np.degrees(angle) == pytest.approx(roll, abs=1e-3)
        assert size > 0


def test_crop_image_roundtrip_affine_and_alignment():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    p = _face_pts(478, MP_EYE_A, MP_EYE_B, MP_LIP_TOP, MP_LIP_BOTTOM, roll_deg=20.0)
    res = crop_image(img, p, dsize=256, scale=1.5, vy_ratio=0.0)
    assert res.img_crop.shape == (256, 256, 3)
    assert np.allclose(res.m_o2c @ res.m_c2o, np.eye(3), atol=1e-4)
    # in crop space the eye line must be horizontal and centred
    axis_c = transform_pts(eye_lip_axis(p), res.m_o2c)
    assert axis_c[0, 0] == pytest.approx(axis_c[1, 0], abs=1e-2)  # same x
    assert axis_c[1, 1] > axis_c[0, 1]  # lips below eyes
    # a landmark maps back to itself through the inverse
    back = transform_pts(transform_pts(p, res.m_o2c), res.m_c2o)
    assert np.allclose(back, p, atol=1e-2)


def test_vy_ratio_shifts_along_face_axis():
    p = _face_pts(478, MP_EYE_A, MP_EYE_B, MP_LIP_TOP, MP_LIP_BOTTOM)
    c0, size, _ = aligned_rect(p, scale=1.0, vx_ratio=0.0, vy_ratio=0.0)
    c1, _, _ = aligned_rect(p, scale=1.0, vx_ratio=0.0, vy_ratio=-0.1)
    assert c1[1] == pytest.approx(c0[1] - 0.1 * size, abs=1e-3)
    assert c1[0] == pytest.approx(c0[0], abs=1e-3)


def test_bin66_soft_argmax():
    logits = np.full((1, 66), -10.0, dtype=np.float32)
    logits[0, 40] = 20.0
    assert bin66_to_degree(logits)[0] == pytest.approx(40 * 3 - 97.5, abs=1e-3)
    # already-degrees passthrough
    assert bin66_to_degree(np.array([[12.5]], dtype=np.float32))[0] == pytest.approx(12.5)


def test_rotation_matrix_is_orthonormal_and_identity_at_zero():
    r0 = rotation_matrix(np.zeros(1), np.zeros(1), np.zeros(1))
    assert np.allclose(r0[0], np.eye(3), atol=1e-6)
    r = rotation_matrix(np.array([10.0]), np.array([-25.0]), np.array([7.0]))
    assert np.allclose(r[0] @ r[0].T, np.eye(3), atol=1e-5)
    assert np.linalg.det(r[0]) == pytest.approx(1.0, abs=1e-5)
    # pure roll rotates the x axis in the image plane (row-vector convention)
    rz = rotation_matrix(np.zeros(1), np.zeros(1), np.array([90.0]))[0]
    v = np.array([1.0, 0.0, 0.0]) @ rz
    assert np.allclose(v, [0.0, 1.0, 0.0], atol=1e-6)


def test_transform_keypoints_and_broadcast():
    kp = np.arange(63, dtype=np.float32).reshape(1, 21, 3) / 63.0
    rot = np.eye(3, dtype=np.float32)[None]
    exp = np.zeros((1, 21, 3), dtype=np.float32)
    scale = np.array([[2.0]], dtype=np.float32)
    t = np.array([[1.0, -1.0, 5.0]], dtype=np.float32)
    x = transform_keypoints(kp, rot, exp, scale, t)
    assert x.shape == (1, 21, 3)
    assert np.allclose(x[0, :, 0], 2 * kp[0, :, 0] + 1.0)
    assert np.allclose(x[0, :, 1], 2 * kp[0, :, 1] - 1.0)
    assert np.allclose(x[0, :, 2], 2 * kp[0, :, 2])  # z ignores t
    info = KPInfo(
        kp=kp,
        exp=exp,
        scale=scale,
        t=t,
        pitch=np.zeros(1),
        yaw=np.zeros(1),
        roll=np.zeros(1),
        rot=rot,
    )
    assert np.allclose(source_keypoints(info), x)
    rot_d = np.repeat(rot, 5, axis=0)
    exp_d = np.zeros((5, 63), dtype=np.float32)
    xd = driving_keypoints(info, rot_d, exp_d)
    assert xd.shape == (5, 21, 3)
    assert np.allclose(xd, np.repeat(x, 5, axis=0))


def test_feathered_mask_profile():
    m = feathered_mask(512, 512, 0.9, 0.9)
    assert m.shape == (512, 512) and m.dtype == np.float32
    assert m.min() >= 0.0 and m.max() <= 1.0
    assert np.all(m[26:486, 26:486] == 1.0)  # inner 90% box
    assert m[0, 256] == 0.0 and m[256, 0] == 0.0 and m[0, 0] == 0.0
    assert m[13, 256] == pytest.approx(0.5, abs=0.05)


def test_pasteback_identity_transform_blends_with_mask():
    frame = np.full((512, 512, 3), 10, dtype=np.uint8)
    face = np.full((512, 512, 3), 250.0, dtype=np.float32)
    mask = np.zeros((512, 512), dtype=np.float32)
    mask[100:400, 100:400] = 1.0
    out = pasteback(frame, face, np.eye(3, dtype=np.float32), mask)
    assert out.dtype == np.uint8
    assert np.all(out[200, 200] == 250) and np.all(out[10, 10] == 10)


def test_composite_alpha_and_pixel_formats():
    fg = np.full((4, 6, 3), 200, dtype=np.uint8)
    bg = np.zeros((4, 6, 3), dtype=np.uint8)
    a = np.full((4, 6), 0.25, dtype=np.float32)
    out = composite_alpha(fg, a, bg)
    assert np.all(out == 50)
    # smooth image: 4:2:0 chroma subsampling is near-lossless on gradients
    yy, xx = np.mgrid[0:64, 0:96]
    rgb = np.stack([xx * 2, yy * 3, 255 - xx * 2], axis=-1).clip(0, 255).astype(np.uint8)
    assert np.array_equal(rgb_to_bgr(rgb)[..., 0], rgb[..., 2])
    yuv = rgb_to_i420(rgb)
    assert yuv.shape == (96, 96)  # H*3/2 x W
    back = i420_to_rgb(yuv, 96)
    assert np.abs(back.astype(int) - rgb.astype(int)).mean() < 4
    with pytest.raises(ValueError):
        rgb_to_i420(rgb[:63])
