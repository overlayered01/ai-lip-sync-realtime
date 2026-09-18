# SPDX-FileCopyrightText: 2024 Ant Group Co., Ltd. (ditto-talkinghead)
# SPDX-FileCopyrightText: 2026 ai-lip-sync contributors
# SPDX-License-Identifier: Apache-2.0
#
# Follows ditto ``core/atomic_components/condition_handler.py`` and
# ``core/utils/eye_info.py`` (Apache-2.0).

"""LMDM condition: per frame ``[hubert 1024 | emo 8 | eye_open 2 | eye_ball 6 | sc 63]`` = 1103.

``ConditionSource`` holds the per-avatar (or per-reference-character) parts.
ditto's shipped config drives the LMDM with a fixed *reference* character's
``sc`` / eye / pose (``ch_info`` in the cfg pickle) and retargets the motion
to the actual avatar afterwards; ``ConditionSource.from_ditto_cfg`` reproduces
that, ``ConditionSource.from_avatar`` conditions on the avatar itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import softmax

EMO_LABELS = ("Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise", "Contempt")
EMO_NEUTRAL = 4
COND_DIM = 1024 + 8 + 2 + 6 + 63  # 1103


def emo_from_label(label: int | list[int] | tuple[int, ...]) -> np.ndarray:
    """One-hot-ish emotion vector: chosen labels get logit 8, rest 0, softmax -> ``(8,)``."""
    logits = np.zeros(8, dtype=np.float32)
    if isinstance(label, (list, tuple)):
        for i in label:
            logits[int(i)] = 8.0
    else:
        logits[int(label)] = 8.0
    return softmax(logits).astype(np.float32)


def parse_emo(emo: int | list | tuple | np.ndarray) -> np.ndarray:
    """ditto ``_parse_emo_seq``: -> ``(M, 8)``; cycled by frame index when M > 1."""
    if isinstance(emo, np.ndarray) and emo.ndim == 2 and emo.shape[1] == 8:
        return emo.astype(np.float32)
    if isinstance(emo, (int, np.integer)):
        return emo_from_label(int(emo)).reshape(1, 8)
    if isinstance(emo, (list, tuple)) and emo and isinstance(emo[0], (int, np.integer)):
        return emo_from_label(list(emo)).reshape(1, 8)
    if isinstance(emo, list) and emo and isinstance(emo[0], (list, tuple)):
        return np.stack([emo_from_label(list(i)) for i in emo], 0)
    raise ValueError(f"unsupported emo spec: {type(emo)}")


# --- eye features from MediaPipe 478 landmarks (ditto EyeAttrUtilsByMP) ----------------------

_L_W = (33, 133)
_L_H = ((145, 159), (144, 160), (153, 158))
_R_W = (263, 362)
_R_H = ((374, 386), (373, 387), (380, 385))
_L_BALL = 468
_R_BALL = 473


def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.sqrt(((p1 - p2) ** 2).sum()))


def _direc(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return (p1 - p2) / (_dist(p1, p2) + 1e-8)


def _one_eye(lmk: np.ndarray, w: tuple[int, int], hs, ball: int) -> tuple[float, np.ndarray]:
    width = _dist(lmk[w[0]], lmk[w[1]])
    opening = sum(_dist(lmk[a], lmk[b]) for a, b in hs) / (width + 1e-8)
    center = (lmk[w[0]] + lmk[w[1]]) * 0.5
    eye_direc = _direc(lmk[w[0]], lmk[w[1]])  # inner -> outer
    ball = lmk[ball]
    move_dist = _dist(ball, center)
    move_direc = _direc(ball, center) - eye_direc
    return opening, move_direc * move_dist


def eye_features(lmk478: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(478, 3)`` or ``(478, 2)`` crop-pixel landmarks -> ``eye_open (2,)``, ``eye_ball (6,)``.

    ditto computes these on the 512 crop with 3-D Face Mesh points; pass
    ``detect_landmarks478(..., keep_z=True)`` for the same result. With 2-D
    points the z component of ``eye_ball`` is zero.
    """
    lmk = np.asarray(lmk478, dtype=np.float32)
    if lmk.shape[1] == 2:
        lmk = np.concatenate([lmk, np.zeros((lmk.shape[0], 1), np.float32)], 1)
    l_open, l_move = _one_eye(lmk, _L_W, _L_H, _L_BALL)
    r_open, r_move = _one_eye(lmk, _R_W, _R_H, _R_BALL)
    eye_open = np.array([l_open, r_open], dtype=np.float32)
    # ditto: np.stack([L_ball_move, R_ball_move], -1) -> (3, 2) -> reshape(6)
    eye_ball = np.stack([l_move, r_move], -1).reshape(6).astype(np.float32)
    return eye_open, eye_ball


# --- condition source + builder -------------------------------------------------------------


@dataclass(slots=True)
class ConditionSource:
    sc: np.ndarray  # (63,) canonical keypoints of the conditioning character
    eye_open: np.ndarray  # (2,)
    eye_ball: np.ndarray  # (6,)
    emo: np.ndarray  # (M, 8)

    @classmethod
    def from_ditto_cfg(
        cls, cfg: dict, emo: int | list | np.ndarray | None = None
    ) -> ConditionSource:
        """From the ``default_kwargs`` of a ditto cfg pickle (reference character)."""
        ch = cfg["ch_info"]
        emo_spec = cfg["emo"] if emo is None else emo
        return cls(
            sc=np.asarray(ch["sc"], np.float32).reshape(63),
            eye_open=np.asarray(ch["eye_open_lst"][0], np.float32).reshape(2),
            eye_ball=np.asarray(ch["eye_ball_lst"][0], np.float32).reshape(6),
            emo=parse_emo(emo_spec),
        )

    @classmethod
    def from_avatar(
        cls,
        kp: np.ndarray,
        lmk478_crop: np.ndarray,
        emo: int | list | np.ndarray = EMO_NEUTRAL,
    ) -> ConditionSource:
        """Condition on the avatar itself: ``kp`` (1,21,3) from its ``KPInfo`` and the
        478 landmarks detected on its 512 crop."""
        eye_open, eye_ball = eye_features(lmk478_crop)
        return cls(
            sc=np.asarray(kp, np.float32).reshape(63),
            eye_open=eye_open,
            eye_ball=eye_ball,
            emo=parse_emo(emo),
        )


class ConditionBuilder:
    """``(n, 1024)`` audio features + absolute frame index -> ``(n, 1103)``."""

    def __init__(self, source: ConditionSource) -> None:
        self.source = source
        self.emo = np.asarray(source.emo, np.float32).reshape(-1, 8)
        self._static = np.concatenate(
            [source.eye_open.reshape(2), source.eye_ball.reshape(6), source.sc.reshape(63)]
        ).astype(np.float32)

    @property
    def dim(self) -> int:
        return COND_DIM

    def __call__(self, aud_feat: np.ndarray, idx: int, emo: np.ndarray | None = None) -> np.ndarray:
        """``idx`` is the absolute motion-frame index of ``aud_feat[0]`` (negative during
        the silence warm-up; clamped to 0 like ditto). ``emo`` overrides per call: ``(8,)``
        or ``(n, 8)``."""
        a = np.asarray(aud_feat, np.float32)
        n = a.shape[0]
        if emo is None:
            ids = [max(i, 0) % self.emo.shape[0] for i in range(idx, idx + n)]
            emo_seq = self.emo[ids]
        else:
            e = np.asarray(emo, np.float32).reshape(-1, 8)
            emo_seq = np.broadcast_to(e, (n, 8)) if e.shape[0] == 1 else e
        static = np.broadcast_to(self._static, (n, self._static.shape[0]))
        return np.concatenate([a, emo_seq, static], -1).astype(np.float32)


__all__ = [
    "COND_DIM",
    "EMO_LABELS",
    "EMO_NEUTRAL",
    "ConditionBuilder",
    "ConditionSource",
    "emo_from_label",
    "eye_features",
    "parse_emo",
]
