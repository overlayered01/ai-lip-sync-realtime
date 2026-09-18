# SPDX-FileCopyrightText: 2026 ai-lip-sync contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for lpavatar.motion with fake engines (no model files needed).

The fake LMDM copies condition channel 0 into motion dim 0, and the fake
HuBERT returns the mean sample value per 50 Hz frame. Feeding audio whose
sample value is its 25 fps frame index therefore lets every test check
*which* frame ends up where -- the timing claims of the handoff §11.
"""

from __future__ import annotations

import numpy as np
import pytest

from lpavatar.motion.condition import COND_DIM, ConditionBuilder, ConditionSource, eye_features
from lpavatar.motion.ditto_motion import (
    Audio2MotionOnline,
    DittoMotionStream,
    FrameControl,
    MotionRetarget,
    MotionStreamConfig,
)
from lpavatar.motion.hubert import (
    FRAME_SAMPLES,
    HOP_SAMPLES,
    WINDOW_SAMPLES,
    HubertFeatures,
    iter_windows,
)
from lpavatar.motion.lmdm import LMDMSampler, ddim_time_pairs, make_beta
from lpavatar.motion.vector import (
    MOTION_DIM,
    MotionInfo,
    motion_to_vector,
    vector_to_motion,
)
from lpavatar.render.keypoints import rotation_matrix, transform_keypoints


class FakeEngine:
    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    @property
    def inputs(self):
        return {}

    @property
    def outputs(self):
        return {}

    def __call__(self, **kw):
        self.calls += 1
        return self.fn(**kw)


def fake_hubert_engine() -> FakeEngine:
    def fn(input_values):
        a = np.asarray(input_values).reshape(-1)
        n = a.shape[0] // 320
        enc = np.zeros((n, 1024), np.float32)
        enc[:, 0] = a[: n * 320].reshape(n, 320).mean(1)
        return {"encoding_out": enc}

    return FakeEngine(fn)


def fake_lmdm_engine() -> FakeEngine:
    def fn(x, cond_frame, cond, time_cond):
        x_start = np.zeros_like(x)
        x_start[..., 0] = cond[..., 0]
        return {"pred_noise": np.zeros_like(x), "x_start": x_start}

    return FakeEngine(fn)


def frame_indexed_audio(n_frames: int) -> np.ndarray:
    """Sample value == its 25 fps frame index."""
    return np.repeat(np.arange(n_frames, dtype=np.float32), FRAME_SAMPLES)


def random_motion(rng) -> MotionInfo:
    return MotionInfo(
        scale=rng.uniform(1.2, 1.8, (1, 1)).astype(np.float32),
        pitch=rng.standard_normal((1, 66)).astype(np.float32),
        yaw=rng.standard_normal((1, 66)).astype(np.float32),
        roll=rng.standard_normal((1, 66)).astype(np.float32),
        t=rng.standard_normal((1, 3)).astype(np.float32),
        exp=(rng.standard_normal((1, 63)) * 0.02).astype(np.float32),
    )


def zero_condition() -> ConditionBuilder:
    return ConditionBuilder(
        ConditionSource(
            sc=np.zeros(63, np.float32),
            eye_open=np.zeros(2, np.float32),
            eye_ball=np.zeros(6, np.float32),
            emo=np.zeros((1, 8), np.float32),
        )
    )


# --- vector -----------------------------------------------------------------------------------


def test_vector_roundtrip():
    info = random_motion(np.random.default_rng(0))
    v = motion_to_vector(info)
    assert v.shape == (MOTION_DIM,)
    assert v[0] == pytest.approx(info.scale.item() - 1.0)
    back = vector_to_motion(v)
    for k in ("scale", "pitch", "yaw", "roll", "t", "exp"):
        assert np.allclose(getattr(back, k), getattr(info, k))


# --- lmdm -------------------------------------------------------------------------------------


def test_ddim_time_pairs():
    pairs = ddim_time_pairs(1000, 10)
    assert pairs[0] == (999, 899)
    assert pairs[-1] == (99, -1)
    assert len(pairs) == 10
    assert ddim_time_pairs(1000, 50)[0] == (999, 979)


def test_make_beta_cosine():
    b = make_beta(1000)
    assert b.shape == (1000,)
    assert b[0] > 0 and b.max() <= 0.999
    ac = np.cumprod(1 - b)
    assert ac[0] > ac[500] > ac[-1] > 0


def test_sampler_last_step_is_x_start():
    eng = fake_lmdm_engine()
    s = LMDMSampler(eng, seed=1)
    kp = np.zeros((1, MOTION_DIM), np.float32)
    cond = np.zeros((1, 80, COND_DIM), np.float32)
    cond[..., 0] = np.arange(80)
    out = s(kp, cond, 10)
    assert out.shape == (1, 80, MOTION_DIM)
    assert eng.calls == 10
    assert np.allclose(out[0, :, 0], np.arange(80))
    assert np.allclose(out[0, :, 1:], 0)


# --- hubert -----------------------------------------------------------------------------------


def test_hubert_window_slices_current_frames():
    hf = HubertFeatures(fake_hubert_engine())
    win = frame_indexed_audio(11)[:WINDOW_SAMPLES]  # 10 frames + 80 samples
    feat = hf(win)
    assert feat.shape == (5, 1024)
    # past 3 frames dropped, future 2 dropped -> frames 3..7 of the window
    assert np.allclose(feat[:, 0], np.arange(3, 8), atol=0.05)


def test_hubert_offline_alignment():
    hf = HubertFeatures(fake_hubert_engine())
    feat = hf.offline(frame_indexed_audio(37))
    assert feat.shape == (37, 1024)
    # the fake has no receptive field, so ditto's 80-sample shift shows up as 0.125
    assert np.allclose(feat[:, 0], np.arange(37), atol=0.15)
    assert hf.silence(75).shape == (75, 1024)


def test_iter_windows_overlap_and_alignment():
    audio = frame_indexed_audio(20)
    wins = list(iter_windows(audio))
    assert all(w.shape == (WINDOW_SAMPLES,) for w in wins)
    assert len(wins) == int(np.ceil((20 * FRAME_SAMPLES + 3 * FRAME_SAMPLES) / HOP_SAMPLES))
    hf = HubertFeatures(fake_hubert_engine())
    f0 = hf(wins[0])[:, 0]
    f1 = hf(wins[1])[:, 0]
    assert np.allclose(f0, np.arange(0, 5), atol=0.05)  # first window = frames 0..4
    assert np.allclose(f1, np.arange(5, 10), atol=0.05)  # hop is 5 frames, windows overlap


# --- condition --------------------------------------------------------------------------------


def test_condition_builder_shapes_and_emo_cycle():
    src = ConditionSource(
        sc=np.ones(63, np.float32),
        eye_open=np.full(2, 2.0, np.float32),
        eye_ball=np.full(6, 3.0, np.float32),
        emo=np.eye(8, dtype=np.float32)[:3],
    )
    cb = ConditionBuilder(src)
    aud = np.zeros((5, 1024), np.float32)
    c = cb(aud, idx=-2)
    assert c.shape == (5, COND_DIM)
    # emo cycles over 3 rows by frame index, negative indices clamp to 0
    assert np.allclose(c[:, 1024:1032].argmax(1), [0, 0, 0, 1, 2])
    assert np.allclose(c[0, 1032:1034], 2.0) and np.allclose(c[0, 1034:1040], 3.0)
    assert np.allclose(c[0, 1040:], 1.0)


def test_eye_features_shapes():
    lmk = np.random.default_rng(0).uniform(0, 512, (478, 3)).astype(np.float32)
    eye_open, eye_ball = eye_features(lmk)
    assert eye_open.shape == (2,) and eye_ball.shape == (6,)
    assert np.isfinite(eye_open).all() and np.isfinite(eye_ball).all()
    eye_open2, _ = eye_features(lmk[:, :2])
    assert eye_open2.shape == (2,)


# --- online state machine ---------------------------------------------------------------------


def make_a2m(overlap: int, seed: int = 0) -> tuple[Audio2MotionOnline, MotionStreamConfig]:
    cfg = MotionStreamConfig(overlap_v2=overlap, sampling_timesteps=10, seed=seed)
    sampler = LMDMSampler(fake_lmdm_engine(), seed=seed)
    silence = np.zeros((overlap, 1024), np.float32)
    silence[:, 0] = np.arange(-overlap, 0)  # warm-up frames carry their (negative) index
    a2m = Audio2MotionOnline(
        sampler, zero_condition(), random_motion(np.random.default_rng(seed)), cfg, silence
    )
    return a2m, cfg


def feats(frame_start: int, n: int = 5) -> np.ndarray:
    f = np.zeros((n, 1024), np.float32)
    f[:, 0] = np.arange(frame_start, frame_start + n)
    return f


@pytest.mark.parametrize(
    "overlap, first_output_after_window, lookahead",
    [(75, 1, 12), (70, 3, 22)],
)
def test_online_output_timing(overlap, first_output_after_window, lookahead):
    a2m, cfg = make_a2m(overlap)
    assert cfg.audio_lookahead_frames == lookahead
    emitted = []
    first = None
    for w in range(12):
        out = a2m.push(feats(5 * w))
        if out.shape[0] and first is None:
            first = w
        emitted.extend(out[:, 0].tolist())
    assert first == first_output_after_window
    # frames come out contiguous from 0 and carry their own index through fusion/smoothing
    # (the first block shows the 3-frame smoothing of the warm-up boundary, < 0.05)
    assert np.allclose(emitted, np.arange(len(emitted)), atol=0.05)
    assert a2m.emitted_frames == len(emitted)
    # the last ``fuse_length`` frames of received audio are still un-fused; the first frame
    # of a block therefore needs ``valid + fuse`` frames of audio after it (+ HuBERT future)
    assert len(emitted) == 5 * 12 - cfg.fuse_length


def test_online_flush_pads_tail():
    a2m, cfg = make_a2m(75)
    total = 0
    for w in range(6):
        total += a2m.push(feats(5 * w)).shape[0]
    tail = a2m.flush()
    assert total + tail.shape[0] == 30
    # the padded tail repeats the last feature, so smoothing pulls the last frame back a bit
    assert np.allclose(tail[:, 0], np.arange(total, 30), atol=0.4)


def test_rewind_drops_pending_and_realigns():
    a2m, cfg = make_a2m(75)
    for w in range(8):
        a2m.push(feats(5 * w))
    e = a2m.emitted_frames
    assert e == 35  # 40 frames received, fuse_length = 5 still pending
    # None: keep everything emitted, drop the un-emitted tail; the next window is the
    # one after the last fully consumed one
    assert a2m.rewind(None) == 35
    out = a2m.push(feats(40))  # window 8 = frames 40..44 -> emits 35..39
    assert np.allclose(out[:, 0], np.arange(35, 40))
    # rewind into already emitted frames: 17 -> quantised to 15
    assert a2m.rewind(17) == 15
    assert a2m.emitted_frames == 15
    out = a2m.push(feats(20))  # window 4 again -> re-emits 15..19
    assert np.allclose(out[:, 0], np.arange(15, 20))
    out = a2m.push(feats(25))
    assert np.allclose(out[:, 0], np.arange(20, 25))


def test_kp_cond_reset_and_fix_dims():
    a2m, cfg = make_a2m(75)
    for w in range(3):
        a2m.push(feats(5 * w))
    # fix_kp_cond=1 with dims [0,202): pose dims follow the last prediction, exp is source
    assert np.allclose(a2m.kp_cond[0, 202:], a2m.s_kp_cond[0, 202:])
    assert a2m.kp_cond[0, 0] != a2m.s_kp_cond[0, 0]
    a2m.reset_kp_cond()
    assert np.allclose(a2m.kp_cond, a2m.s_kp_cond)


# --- retarget ---------------------------------------------------------------------------------


def test_retarget_identity_and_controls():
    rng = np.random.default_rng(3)
    src = random_motion(rng)
    kp = rng.standard_normal((1, 21, 3)).astype(np.float32)
    cfg = MotionStreamConfig(drive_eye=False, fix_gaze=False, delta_eye_arr=None)
    rt = MotionRetarget(src, kp, cfg)
    v = motion_to_vector(src)
    x_d, m = rt(v, 0)  # first frame becomes d0 -> identity
    rot_s = rotation_matrix(m.pitch, m.yaw, m.roll)
    expect = transform_keypoints(kp, rot_s, src.exp, src.scale, src.t)
    assert np.allclose(x_d, expect, atol=1e-5)
    # a driving frame with a different exp: only lips are driven, rest stays source
    d = src.copy()
    d.exp = d.exp + 0.01
    x_d2, m2 = rt(motion_to_vector(d), 1)
    exp2 = m2.exp.reshape(21, 3)
    lips = [6, 12, 14, 17, 19, 20]
    assert np.allclose(exp2[lips], src.exp.reshape(21, 3)[lips] + 0.01, atol=1e-6)
    others = [i for i in range(21) if i not in lips]
    assert np.allclose(exp2[others], src.exp.reshape(21, 3)[others], atol=1e-6)
    # vad_alpha = 0 holds the lips at the source pose
    _, m3 = rt(motion_to_vector(d), 2, FrameControl(vad_alpha=0.0))
    assert np.allclose(m3.exp, src.exp, atol=1e-6)
    # pitch offset rotates the head
    _, m4 = rt(v, 3, FrameControl(delta_pitch=10.0))
    assert m4.pitch.item() == pytest.approx(m.pitch.item() + 10.0, abs=1e-4)


def test_stream_facade_controls_and_rewind():
    cfg = MotionStreamConfig(overlap_v2=75, drive_eye=False, fix_gaze=False, seed=0)
    rng = np.random.default_rng(0)
    src = random_motion(rng)
    hf = HubertFeatures(fake_hubert_engine())
    silence = hf.silence(75)
    silence[:, 0] = np.arange(-75, 0)
    a2m = Audio2MotionOnline(
        LMDMSampler(fake_lmdm_engine(), seed=0), zero_condition(), src, cfg, silence
    )
    rt = MotionRetarget(src, rng.standard_normal((1, 21, 3)).astype(np.float32), cfg)
    stream = DittoMotionStream(hubert=hf, a2m=a2m, retarget=rt, cfg=cfg)
    wins = list(iter_windows(frame_indexed_audio(40), pad_end=False))
    outs = []
    for i, w in enumerate(wins[:6]):
        outs += stream.push_window(w, FrameControl(vad_alpha=0.0 if i == 2 else 1.0))
    assert [o.frame_idx for o in outs] == list(range(25))
    assert np.allclose([o.vector[0] for o in outs], np.arange(25), atol=0.05)
    assert stream.windows_pushed == 6
    assert stream.rewind(12) == 10
    assert stream.windows_pushed == 3
    outs2 = stream.push_window(wins[3])
    assert [o.frame_idx for o in outs2] == list(range(10, 15))
    assert np.allclose([o.vector[0] for o in outs2], np.arange(10, 15), atol=0.05)
