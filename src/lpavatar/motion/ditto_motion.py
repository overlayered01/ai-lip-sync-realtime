# SPDX-FileCopyrightText: 2024 Ant Group Co., Ltd. (ditto-talkinghead)
# SPDX-FileCopyrightText: 2026 ai-lip-sync contributors
# SPDX-License-Identifier: Apache-2.0
#
# Ports ditto ``core/atomic_components/audio2motion.py`` (Audio2Motion),
# ``stream_pipeline_online.py`` (``_audio2motion_worker``) and
# ``core/atomic_components/motion_stitch.py`` (MotionStitch minus the stitch
# network, which lives in ``lpavatar.render.stage``). Changes for live use:
# sliding-window input, ``rewind`` for barge-in, per-frame ``FrameControl``,
# ``overlap_v2=75`` default. See docs/DITTO_AVATAR_HANDOFF.md §11.

"""Online audio -> driving-keypoint state machine.

    stream = DittoMotionStream.from_ditto_cfg(cfg_pkl, avatar, hubert=..., lmdm=...)
    for window in windows_of_6480_samples_every_3200:      # see hubert.iter_windows
        for out in stream.push_window(window, control):    # 0 or 5 MotionOutput
            face = render_face(avatar.f_s, avatar.x_s, out.x_d, engines=...)

Frame indices are absolute (frame 0 = first frame of the first window).
``stream.emitted_frames`` tells how far output has reached; ``rewind(frame)``
throws away everything after ``frame`` (barge-in), keeping the LMDM context.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lpavatar.motion.condition import ConditionBuilder, ConditionSource
from lpavatar.motion.hubert import CHUNK_FUTURE, HubertFeatures
from lpavatar.motion.lmdm import LMDMSampler
from lpavatar.motion.vector import (
    MOTION_DIM,
    POSE_DIMS,
    MotionInfo,
    motion_from_dict,
    motion_from_kpinfo,
    motion_to_vector,
    vector_to_motion,
)
from lpavatar.registration.avatar import Avatar
from lpavatar.render.keypoints import bin66_to_degree, rotation_matrix, transform_keypoints
from lpavatar.runtime.onnx_engine import Engine

LIP_KP = (6, 12, 14, 17, 19, 20)
EYE_KP = (11, 13, 15, 16, 18)
BLINK_OPEN_MIN = 60
BLINK_OPEN_MAX = 100


def load_ditto_cfg(path: str | Path) -> dict:
    """``default_kwargs`` of a ditto cfg pickle (plain dict / numpy, no torch needed)."""
    with open(path, "rb") as f:
        cfg = pickle.load(f)
    return cfg["default_kwargs"] if "default_kwargs" in cfg else cfg


# --------------------------------------------------------------------------------------------
# config


@dataclass(slots=True)
class MotionStreamConfig:
    seq_frames: int = 80
    overlap_v2: int = 75  # ditto ships 70; 75 halves the output lag (§11.1)
    sampling_timesteps: int = 10
    fix_kp_cond: int = 1
    fix_kp_cond_dim: tuple[int, int] | None = (0, POSE_DIMS)
    smo_k_d: int = 3
    v_min_max_for_clip: np.ndarray | None = None  # (>=2, 265): rows 0 / 1 = min / max
    relative_d: bool = True
    use_d_keys: tuple[str, ...] = ("exp", "pitch", "yaw", "roll", "t")
    drive_eye: bool = True
    delta_eye_arr: np.ndarray | None = None  # (15, 63) blink sequence
    delta_eye_open_n: int = 0  # 0: random 60..100 frames between blinks, <0: never, >0: fixed
    fix_gaze: bool = True
    delta_pitch: float = 0.0  # ditto overall_ctrl_info (shipped: 2)
    delta_yaw: float = 0.0
    delta_roll: float = 0.0
    seed: int | None = None

    @property
    def valid_clip_len(self) -> int:
        return self.seq_frames - self.overlap_v2

    @property
    def fuse_length(self) -> int:
        return min(self.overlap_v2, self.valid_clip_len)

    @property
    def audio_lookahead_frames(self) -> int:
        """Audio frames that must exist beyond frame ``k`` before ``k`` (first of a
        block) can be emitted: ``valid + fuse`` for the LMDM plus HuBERT's future."""
        return self.valid_clip_len + self.fuse_length + CHUNK_FUTURE

    @classmethod
    def from_ditto_cfg(cls, cfg: dict, **overrides) -> MotionStreamConfig:
        kw = dict(
            seq_frames=80,
            overlap_v2=75,
            sampling_timesteps=int(cfg.get("sampling_timesteps", 10)),
            fix_kp_cond=int(cfg.get("fix_kp_cond", 1)),
            fix_kp_cond_dim=tuple(cfg["fix_kp_cond_dim"]) if cfg.get("fix_kp_cond_dim") else None,
            smo_k_d=int(cfg.get("smo_k_d", 3)),
            v_min_max_for_clip=cfg.get("v_min_max_for_clip"),
            relative_d=bool(cfg.get("relative_d", True)),
            delta_eye_arr=cfg.get("delta_eye_arr"),
            delta_eye_open_n=int(cfg.get("delta_eye_open_n", 0)),
            delta_pitch=float((cfg.get("overall_ctrl_info") or {}).get("delta_pitch", 0.0)),
            delta_yaw=float((cfg.get("overall_ctrl_info") or {}).get("delta_yaw", 0.0)),
            delta_roll=float((cfg.get("overall_ctrl_info") or {}).get("delta_roll", 0.0)),
        )
        if cfg.get("use_d_keys"):
            kw["use_d_keys"] = tuple(cfg["use_d_keys"])
        kw.update(overrides)
        return cls(**kw)


# --------------------------------------------------------------------------------------------
# Audio2Motion online (LMDM window management, fusion, smoothing, kp_cond)


class Audio2MotionOnline:
    """Feeds 80-frame condition windows to the LMDM and returns *valid* motion vectors.

    ``push(feat)``: ``feat (n, 1024)`` new audio features (any n, normally 5) ->
    ``(m, 265)`` valid vectors, ``m`` a multiple of ``valid_clip_len`` (often 0).
    ``flush()``: end of stream, pads the last window. ``rewind(frame)``: barge-in.
    """

    def __init__(
        self,
        sampler: LMDMSampler,
        condition: ConditionBuilder,
        source_motion: MotionInfo,
        cfg: MotionStreamConfig,
        hubert_silence: np.ndarray,
    ) -> None:
        self.cfg = cfg
        self.sampler = sampler
        self.condition = condition
        self.seq = cfg.seq_frames
        self.valid = cfg.valid_clip_len
        self.fuse = cfg.fuse_length
        self.overlap = cfg.overlap_v2
        if self.valid <= 0:
            raise ValueError("overlap_v2 must be < seq_frames")
        self.fuse_alpha = (np.arange(self.fuse, dtype=np.float32) / self.fuse).reshape(1, -1, 1)
        self.s_kp_cond = motion_to_vector(source_motion).reshape(1, MOTION_DIM)
        self.kp_cond = self.s_kp_cond.copy()
        self.sampler.setup(cfg.sampling_timesteps)
        if hubert_silence.shape[0] != self.overlap:
            raise ValueError(f"need {self.overlap} silence features, got {hubert_silence.shape[0]}")
        # feature index F <-> motion frame F - overlap (warm-up silence is negative time)
        self.audio_feat = np.asarray(hubert_silence, np.float32).copy()
        self.feat0 = 0  # absolute feature index of audio_feat[0]
        self.item_buffer = np.zeros((0, hubert_silence.shape[1]), np.float32)
        self.local_idx = 0  # next window start (relative to audio_feat)
        self.res: np.ndarray | None = None  # (1, L, 265)
        self.res_frame0 = -self.overlap  # motion frame of res[:, 0]
        self.valid_start: int | None = None  # relative to res
        self.clip_idx = 0
        self.d0: MotionInfo | None = None
        self.emitted_frames = 0

    # -- helpers -----------------------------------------------------------------------------

    def _fuse(self, res: np.ndarray, pred: np.ndarray) -> np.ndarray:
        r1_s, r1_e = res.shape[1] - self.fuse, res.shape[1]
        r2_s = self.seq - self.valid - self.fuse
        r2_e = self.seq - self.valid
        r1 = res[:, r1_s:r1_e]
        r2 = pred[:, r2_s:r2_e]
        res[:, r1_s:r1_e] = r1 * (1 - self.fuse_alpha) + r2 * self.fuse_alpha
        return np.concatenate([res, pred[:, r2_e:]], 1)

    def _smo(self, res: np.ndarray, s: int, e: int) -> np.ndarray:
        k = self.cfg.smo_k_d
        if k <= 1:
            return res
        src = res.copy()
        n = res.shape[1]
        half = k // 2
        for i in range(max(s, 0), min(e, n)):
            ss, ee = max(0, i - half), min(n, i + half + 1)
            res[:, i, :POSE_DIMS] = src[:, ss:ee, :POSE_DIMS].mean(axis=1)
        return res

    def _update_kp_cond(self, res: np.ndarray, idx: int) -> None:
        cfg = self.cfg
        if cfg.fix_kp_cond == 0:
            self.kp_cond = res[:, idx - 1].copy()
        elif self.clip_idx % cfg.fix_kp_cond == 0:
            self.kp_cond = self.s_kp_cond.copy()
            if cfg.fix_kp_cond_dim is not None:
                ds, de = cfg.fix_kp_cond_dim
                self.kp_cond[:, ds:de] = res[:, idx - 1, ds:de]
        else:
            self.kp_cond = res[:, idx - 1].copy()

    def _run_clip(self, aud_cond: np.ndarray) -> None:
        pred = self.sampler(self.kp_cond, aud_cond, self.cfg.sampling_timesteps)
        if self.res is None:
            res = self._smo(pred, 0, pred.shape[1])
        else:
            res = self._fuse(self.res, pred)
            L = res.shape[1]
            res = self._smo(res, L - self.valid - self.fuse, L - self.valid + 1)
        self.clip_idx += 1
        self._update_kp_cond(res, res.shape[1] - self.overlap)
        self.res = res

    def reset_kp_cond(self) -> None:
        """Full reset to the source pose (idle-drift guard, §11.4)."""
        self.kp_cond = self.s_kp_cond.copy()

    def clip_vector(self, v: np.ndarray) -> np.ndarray:
        mm = self.cfg.v_min_max_for_clip
        if mm is None:
            return v
        return np.clip(v, mm[0][None], mm[1][None])

    # -- streaming ---------------------------------------------------------------------------

    def push(self, feat: np.ndarray | None, *, is_end: bool = False) -> np.ndarray:
        """New features in, valid motion vectors ``(m, 265)`` out (``m`` may be 0)."""
        if feat is not None:
            self.item_buffer = np.concatenate([self.item_buffer, np.asarray(feat, np.float32)], 0)
        if not is_end and self.item_buffer.shape[0] < self.valid:
            return np.zeros((0, MOTION_DIM), np.float32)
        self.audio_feat = np.concatenate([self.audio_feat, self.item_buffer], 0)
        self.item_buffer = self.item_buffer[:0]
        total_real = self.feat0 + self.audio_feat.shape[0] - self.overlap  # frames with audio

        out: list[np.ndarray] = []
        while True:
            aud = self.audio_feat[self.local_idx : self.local_idx + self.seq]
            real_valid = self.valid
            if aud.shape[0] == 0:
                break
            if aud.shape[0] < self.seq:
                if not is_end:
                    break
                # end of stream: pad with the last feature, emit only frames that had audio
                remaining = total_real - self.emitted_frames
                if remaining <= 0 or self.valid_start is None:
                    break
                real_valid = min(self.valid, remaining)
                pad = np.repeat(aud[-1:], self.seq - aud.shape[0], 0)
                aud = np.concatenate([aud, pad], 0)
            frame_idx = self.feat0 + self.local_idx - self.overlap  # frame of aud[0]
            self._run_clip(self.condition(aud, frame_idx)[None])
            assert self.res is not None
            if self.valid_start is None:
                # first clip: warm-up only, keep d0 (relative-motion anchor)
                self.valid_start = self.res.shape[1] - self.fuse
                self.d0 = vector_to_motion(self.clip_vector(self.res[0, 0]))
            else:
                out.append(self.res[0, self.valid_start : self.valid_start + real_valid])
                self.valid_start += real_valid
                self.emitted_frames += real_valid
            self.local_idx += real_valid
            self._trim()
            if self.local_idx >= self.audio_feat.shape[0]:
                break
        if not out:
            return np.zeros((0, MOTION_DIM), np.float32)
        return self.clip_vector(np.concatenate(out, 0))

    def flush(self) -> np.ndarray:
        """End of stream: run once more with the last feature repeated (offline use)."""
        return self.push(None, is_end=True)

    def _trim(self) -> None:
        assert self.res is not None and self.valid_start is not None
        L = self.res.shape[1]
        if L > self.seq * 2:
            cut = L - self.seq * 2
            self.res = self.res[:, cut:]
            self.valid_start -= cut
            self.res_frame0 += cut
        L = self.audio_feat.shape[0]
        if L > self.seq * 2:
            cut = L - self.seq * 2
            self.audio_feat = self.audio_feat[cut:]
            self.local_idx -= cut
            self.feat0 += cut

    def rewind(self, frame: int | None = None) -> int:
        """Barge-in: forget every prediction and feature after motion frame ``frame``.

        ``None`` (or a frame beyond what was emitted) keeps all emitted frames and
        only drops the not-yet-emitted tail + unconsumed features. A smaller
        ``frame`` (the audio position actually played) also re-opens the
        already-emitted frames ``[frame, emitted)`` so they are re-predicted and
        fused with the silence that follows. Returns the frame actually rewound to.
        """
        if self.res is None or self.valid_start is None:
            self.item_buffer = self.item_buffer[:0]
            return self.emitted_frames
        if frame is None or frame > self.emitted_frames:
            frame = self.emitted_frames
        # stay on the clip grid so window numbering (5 frames per window) still lines up
        frame = (frame // self.valid) * self.valid
        lowest = max(self.res_frame0 + self.valid, self.feat0 - self.overlap + self.valid)
        frame = max(frame, lowest)
        self.res = self.res[:, : frame + self.fuse - self.res_frame0]
        self.valid_start = frame - self.res_frame0
        self.local_idx = frame + self.fuse - self.feat0
        self.audio_feat = self.audio_feat[: frame + self.fuse + self.overlap - self.feat0]
        self.item_buffer = self.item_buffer[:0]
        self.emitted_frames = frame
        return frame


# --------------------------------------------------------------------------------------------
# retargeting: 265-d vector -> driving keypoints for one avatar


@dataclass(slots=True)
class FrameControl:
    """Per-frame knobs (ditto ``ctrl_kwargs``). Applied at emit time."""

    vad_alpha: float = 1.0  # 1: model lips; 0: source lips (silence)
    delta_pitch: float = 0.0  # degrees, added on top of the config offsets
    delta_yaw: float = 0.0
    delta_roll: float = 0.0
    delta_exp: np.ndarray | None = None  # (63,)


def _blink_schedule(n: int, blink_n: int, open_n: int, rng: np.random.Generator) -> list[int]:
    """ditto ``_set_eye_blink_idx``: index into ``delta_eye_arr`` per frame (0 = open)."""
    idx = [0] * n
    if open_n < 0:
        return idx
    cur = open_n if open_n > 0 else int(rng.integers(BLINK_OPEN_MIN, BLINK_OPEN_MAX + 1))
    end_n = open_n if open_n > 0 else int(rng.integers(BLINK_OPEN_MIN, BLINK_OPEN_MAX + 1))
    max_i = n - max(end_n, blink_n)
    while cur < max_i:
        idx[cur : cur + blink_n] = list(range(blink_n))
        gap = open_n if open_n > 0 else int(rng.integers(BLINK_OPEN_MIN, BLINK_OPEN_MAX + 1))
        cur = cur + blink_n + gap
    return idx


class MotionRetarget:
    """ditto ``MotionStitch`` without the stitch network: relative motion onto the
    avatar, lip/eye expression masking, blinks, gaze fix, pose offsets."""

    BLINK_SCHEDULE_LEN = 3000

    def __init__(
        self,
        avatar_motion: MotionInfo,
        kp_source: np.ndarray,
        cfg: MotionStreamConfig,
        *,
        reference_motion: MotionInfo | None = None,
        seed: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.x_s = avatar_motion
        self.kp = np.asarray(kp_source, np.float32).reshape(1, 21, 3)
        self.d0: MotionInfo | None = None
        rng = np.random.default_rng(seed)
        # scale ratio between the conditioning character and this avatar (ditto ch_info)
        ratio = 1.0
        if reference_motion is not None:
            ratio = float(reference_motion.scale.item() / avatar_motion.scale.item())
        self.gain = {
            k: (ratio if k in {"exp", "pitch", "yaw", "roll"} else 1.0) for k in cfg.use_d_keys
        }
        # expression mixing masks (_fix_exp_for_x_d_info_v2)
        a1 = np.zeros((21, 3), np.float32)
        a1[list(LIP_KP)] = 1
        a2 = np.zeros((21, 3), np.float32)
        self.use_blink = cfg.drive_eye and cfg.delta_eye_arr is not None
        if cfg.drive_eye:
            if cfg.delta_eye_arr is None:
                a1[list(EYE_KP)] = 1
            else:
                a2[list(EYE_KP)] = 1
        a1, a2 = a1.reshape(1, -1), a2.reshape(1, -1)
        self.fix_a1 = a1 * (1 - a2)
        self.fix_a2 = (1 - a1) + a1 * a2
        self.fix_a3 = a2
        self.lip_mask = np.zeros((21, 3), np.float32)
        self.lip_mask[list(LIP_KP)] = 1
        self.lip_mask = self.lip_mask.reshape(1, -1)
        self.blink_idx: list[int] = []
        if self.use_blink:
            self.blink_idx = _blink_schedule(
                self.BLINK_SCHEDULE_LEN, len(cfg.delta_eye_arr), cfg.delta_eye_open_n, rng
            )
        self.pose_s = (
            float(bin66_to_degree(avatar_motion.yaw)[0]),
            float(bin66_to_degree(avatar_motion.pitch)[0]),
        )

    def _mix(self, d: MotionInfo) -> MotionInfo:
        """ditto ``_mix_s_d_info``: ``x_s + (x_d - d0) * gain`` for driven keys."""
        x = self.x_s.copy()
        if self.cfg.relative_d:
            if self.d0 is None:
                self.d0 = d.copy()
            for k, g in self.gain.items():
                setattr(x, k, getattr(self.x_s, k) + (getattr(d, k) - getattr(self.d0, k)) * g)
        else:
            for k, g in self.gain.items():
                setattr(x, k, getattr(d, k) * g)
        return x

    def _fix_gaze(self, x: MotionInfo, yaw_d: float, pitch_d: float) -> None:
        dx = (yaw_d - self.pose_s[0]) * 0.26
        dy = (pitch_d - self.pose_s[1]) * 0.28
        exp = x.exp
        if dx > 0:
            exp[0, 33] += dx * 0.0007
            exp[0, 45] += dx * 0.001
        else:
            exp[0, 33] += dx * 0.001
            exp[0, 45] += dx * 0.0007
        exp[0, 34] += dy * -0.001
        exp[0, 46] += dy * -0.001

    def __call__(
        self, vec: np.ndarray, frame_idx: int, ctrl: FrameControl | None = None
    ) -> tuple[np.ndarray, MotionInfo]:
        """``vec (265,)`` (already clipped) -> ``x_d (1, 21, 3)`` driving keypoints + the
        retargeted ``MotionInfo`` (pose fields in degrees ``(1,)``)."""
        ctrl = ctrl or FrameControl()
        cfg = self.cfg
        x = self._mix(vector_to_motion(vec))

        delta_eye: np.ndarray | float = 0.0
        if self.use_blink:
            assert cfg.delta_eye_arr is not None
            delta_eye = cfg.delta_eye_arr[self.blink_idx[frame_idx % len(self.blink_idx)]][None]
        x.exp = x.exp * self.fix_a1 + self.x_s.exp * self.fix_a2 + delta_eye * self.fix_a3

        if ctrl.vad_alpha < 1.0:
            # ditto ctrl_vad blends the whole exp; we blend the lip keypoints only so
            # blinks keep running while the mouth is held at the source pose.
            a = float(np.clip(ctrl.vad_alpha, 0.0, 1.0))
            m = self.lip_mask
            x.exp = x.exp * (1 - m) + (x.exp * a + self.x_s.exp * (1 - a)) * m
        if ctrl.delta_exp is not None:
            x.exp = x.exp + np.asarray(ctrl.delta_exp, np.float32).reshape(1, 63)

        pitch = float(bin66_to_degree(x.pitch)[0]) + cfg.delta_pitch + ctrl.delta_pitch
        yaw = float(bin66_to_degree(x.yaw)[0]) + cfg.delta_yaw + ctrl.delta_yaw
        roll = float(bin66_to_degree(x.roll)[0]) + cfg.delta_roll + ctrl.delta_roll
        if cfg.drive_eye and cfg.fix_gaze:
            self._fix_gaze(x, yaw, pitch)

        rot = rotation_matrix(np.array([pitch]), np.array([yaw]), np.array([roll]))
        x_d = transform_keypoints(self.kp, rot, x.exp, self.x_s.scale, x.t)
        x.pitch, x.yaw, x.roll = (
            np.array([pitch], np.float32),
            np.array([yaw], np.float32),
            np.array([roll], np.float32),
        )
        return x_d.astype(np.float32), x


# --------------------------------------------------------------------------------------------
# facade


@dataclass(slots=True)
class MotionOutput:
    frame_idx: int
    x_d: np.ndarray  # (1, 21, 3) driving keypoints (pre-stitch)
    motion: MotionInfo  # retargeted, pose in degrees
    vector: np.ndarray  # (265,) raw LMDM output (clipped)


@dataclass(slots=True)
class DittoMotionStream:
    hubert: HubertFeatures
    a2m: Audio2MotionOnline
    retarget: MotionRetarget
    cfg: MotionStreamConfig
    controls: dict[int, FrameControl] = field(default_factory=dict)
    default_control: FrameControl = field(default_factory=FrameControl)
    windows_pushed: int = 0

    @classmethod
    def from_ditto_cfg(
        cls,
        cfg_pkl: str | Path | dict,
        avatar: Avatar,
        *,
        hubert: Engine,
        lmdm: Engine,
        condition_on: str = "reference",
        emo: int | list | np.ndarray | None = None,
        lmk478_crop: np.ndarray | None = None,
        **cfg_overrides,
    ) -> DittoMotionStream:
        """Build everything from a ditto cfg pickle. ``condition_on="reference"`` uses
        the pickle's reference character for the LMDM condition and kp_cond (what ditto
        ships); ``"avatar"`` conditions on this avatar (needs ``lmk478_crop`` from
        ``detect_landmarks478(avatar.crop_512, ..., keep_z=True)``)."""
        raw = load_ditto_cfg(cfg_pkl) if not isinstance(cfg_pkl, dict) else cfg_pkl
        cfg = MotionStreamConfig.from_ditto_cfg(raw, **cfg_overrides)
        avatar_motion = motion_from_kpinfo(avatar.kp_info)
        if condition_on == "reference":
            ref = raw["ch_info"]
            source = ConditionSource.from_ditto_cfg(raw, emo=emo)
            cond_motion = motion_from_dict(ref["x_s_info_lst"][0])
            ref_motion: MotionInfo | None = cond_motion
        elif condition_on == "avatar":
            if lmk478_crop is None:
                raise ValueError("condition_on='avatar' needs lmk478_crop")
            source = ConditionSource.from_avatar(
                avatar.kp_info.kp, lmk478_crop, emo if emo is not None else 4
            )
            cond_motion = avatar_motion
            ref_motion = None
        else:
            raise ValueError(condition_on)
        hf = HubertFeatures(hubert)
        sampler = LMDMSampler(lmdm, seq_frames=cfg.seq_frames, seed=cfg.seed)
        a2m = Audio2MotionOnline(
            sampler, ConditionBuilder(source), cond_motion, cfg, hf.silence(cfg.overlap_v2)
        )
        rt = MotionRetarget(
            avatar_motion, avatar.kp_info.kp, cfg, reference_motion=ref_motion, seed=cfg.seed
        )
        return cls(hubert=hf, a2m=a2m, retarget=rt, cfg=cfg)

    # -- properties --------------------------------------------------------------------------

    @property
    def emitted_frames(self) -> int:
        return self.a2m.emitted_frames

    @property
    def frames_per_window(self) -> int:
        return self.hubert.chunksize[1]

    @property
    def audio_lookahead_frames(self) -> int:
        return self.cfg.audio_lookahead_frames

    # -- streaming ---------------------------------------------------------------------------

    def set_control(self, frame_idx: int, n: int, ctrl: FrameControl) -> None:
        for i in range(frame_idx, frame_idx + n):
            self.controls[i] = ctrl

    def _emit(self, vectors: np.ndarray) -> list[MotionOutput]:
        if self.retarget.d0 is None and self.a2m.d0 is not None:
            self.retarget.d0 = self.a2m.d0.copy()
        outs = []
        start = self.a2m.emitted_frames - vectors.shape[0]
        for i, v in enumerate(vectors):
            fi = start + i
            ctrl = self.controls.pop(fi, self.default_control)
            x_d, motion = self.retarget(v, fi, ctrl)
            outs.append(MotionOutput(frame_idx=fi, x_d=x_d, motion=motion, vector=v))
        return outs

    def push_window(
        self, audio_window: np.ndarray, ctrl: FrameControl | None = None
    ) -> list[MotionOutput]:
        """One 6,480-sample window (hop 3,200) -> 0 or ``valid_clip_len``·k frames.
        ``ctrl`` applies to the 5 frames this window's *current* part covers."""
        if ctrl is not None:
            self.set_control(
                self.windows_pushed * self.frames_per_window, self.frames_per_window, ctrl
            )
        self.windows_pushed += 1
        return self._emit(self.a2m.push(self.hubert(audio_window)))

    def push_features(self, feat: np.ndarray) -> list[MotionOutput]:
        """Already-extracted ``(n, 1024)`` features (offline / tests)."""
        return self._emit(self.a2m.push(feat))

    def flush(self) -> list[MotionOutput]:
        return self._emit(self.a2m.flush())

    def rewind(self, frame: int | None = None) -> int:
        """Barge-in at (played) motion frame ``frame``; see ``Audio2MotionOnline.rewind``."""
        f = self.a2m.rewind(frame)
        self.windows_pushed = (f + self.cfg.fuse_length) // self.frames_per_window
        for k in [k for k in self.controls if k >= f]:
            del self.controls[k]
        return f

    def reset_pose(self) -> None:
        self.a2m.reset_kp_cond()


__all__ = [
    "EYE_KP",
    "LIP_KP",
    "Audio2MotionOnline",
    "DittoMotionStream",
    "FrameControl",
    "MotionOutput",
    "MotionRetarget",
    "MotionStreamConfig",
    "load_ditto_cfg",
]
