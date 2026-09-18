# SPDX-FileCopyrightText: 2024 Ant Group Co., Ltd. (ditto-talkinghead)
# SPDX-FileCopyrightText: 2026 ai-lip-sync contributors
# SPDX-License-Identifier: Apache-2.0
#
# Follows ditto-talkinghead ``core/atomic_components/wav2feat.py`` (Apache-2.0).

"""HuBERT streaming features: one 6,480-sample window -> 5 motion frames at 25 Hz.

Window layout at 16 kHz (``chunksize = (past 3, current 5, future 2)`` frames
of 640 samples, plus an 80-sample shift)::

    |<-- 1920 past -->|<------ 3200 current ------>|<- 1280 future ->|80|
                       ^ the 5 frames this window produces

Consecutive windows advance by ``HOP_SAMPLES = 3200`` and therefore overlap
with their neighbours; see docs/DITTO_AVATAR_HANDOFF.md §11.2. The HuBERT
graph emits 50 Hz frames; the last ``2 * (5 + 2)`` are taken, the future
``2 * 2`` dropped, and adjacent pairs averaged to 25 Hz.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np

from lpavatar.runtime.onnx_engine import Engine

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 640  # one 25 fps motion frame
SHIFT_SAMPLES = 80
CHUNK_PAST = 3
CHUNK_CURRENT = 5
CHUNK_FUTURE = 2
CHUNKSIZE: tuple[int, int, int] = (CHUNK_PAST, CHUNK_CURRENT, CHUNK_FUTURE)
WINDOW_SAMPLES = sum(CHUNKSIZE) * FRAME_SAMPLES + SHIFT_SAMPLES  # 6480
HOP_SAMPLES = CHUNK_CURRENT * FRAME_SAMPLES  # 3200
FEAT_DIM = 1024


class HubertFeatures:
    """``hubert_streaming_fix_kv.onnx``: ``input_values (1, N)`` -> ``encoding_out (T, 1024)``."""

    def __init__(self, engine: Engine, chunksize: tuple[int, int, int] = CHUNKSIZE) -> None:
        self.engine = engine
        self.chunksize = chunksize
        self.window_samples = sum(chunksize) * FRAME_SAMPLES + SHIFT_SAMPLES
        self.hop_samples = chunksize[1] * FRAME_SAMPLES
        self.feat_dim = FEAT_DIM

    def __call__(self, audio_window: np.ndarray) -> np.ndarray:
        """``(window_samples,)`` float32 -> ``(chunksize[1], 1024)`` float32."""
        a = np.asarray(audio_window, dtype=np.float32).reshape(-1)
        if a.shape[0] != self.window_samples:
            raise ValueError(
                f"HuBERT window must be {self.window_samples} samples, got {a.shape[0]}"
            )
        enc = self.engine(input_values=a.reshape(1, -1))["encoding_out"]
        enc = np.asarray(enc, dtype=np.float32).reshape(-1, FEAT_DIM)
        _, cur, fut = self.chunksize
        start = -(cur + fut) * 2
        end = -fut * 2 if fut > 0 else None
        valid = enc[start:end]
        if valid.shape[0] != cur * 2:
            raise RuntimeError(
                f"HuBERT returned {enc.shape[0]} frames; expected at least {(cur + fut) * 2}"
            )
        return valid.reshape(cur, 2, FEAT_DIM).mean(axis=1)

    def offline(self, audio_16k: np.ndarray) -> np.ndarray:
        """Whole clip -> ``(ceil(len / 640), 1024)``; ditto's ``wav2feat`` padding."""
        a = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
        num_f = math.ceil(a.shape[0] / SAMPLE_RATE * 25)
        past, cur, fut = self.chunksize
        lead = self.window_samples - (cur + fut) * FRAME_SAMPLES
        padded = np.concatenate(
            [np.zeros(lead, np.float32), a, np.zeros(self.window_samples, np.float32)]
        )
        out = []
        i = 0
        while i < num_f:
            s = i * FRAME_SAMPLES
            out.append(self(padded[s : s + self.window_samples]))
            i += cur
        return np.concatenate(out, 0)[:num_f]

    def silence(self, n_frames: int) -> np.ndarray:
        """Features for ``n_frames`` of digital silence (used to warm the LMDM window)."""
        return self.offline(np.zeros(n_frames * FRAME_SAMPLES, np.float32))


def iter_windows(
    audio_16k: np.ndarray, *, chunksize: tuple[int, int, int] = CHUNKSIZE, pad_end: bool = True
) -> Iterator[np.ndarray]:
    """Slice a whole clip the way the live player does: ``past`` frames of
    silence prepended, then ``window_samples`` windows every ``hop`` samples.
    Mirrors ditto ``inference.py`` (online branch)."""
    past, cur, _ = chunksize
    window = sum(chunksize) * FRAME_SAMPLES + SHIFT_SAMPLES
    hop = cur * FRAME_SAMPLES
    a = np.concatenate(
        [np.zeros(past * FRAME_SAMPLES, np.float32), np.asarray(audio_16k, np.float32)]
    )
    for i in range(0, a.shape[0], hop):
        w = a[i : i + window]
        if w.shape[0] < window:
            if not pad_end:
                return
            w = np.pad(w, (0, window - w.shape[0]))
        yield w


__all__ = [
    "CHUNKSIZE",
    "CHUNK_CURRENT",
    "CHUNK_FUTURE",
    "CHUNK_PAST",
    "FEAT_DIM",
    "FRAME_SAMPLES",
    "HOP_SAMPLES",
    "SAMPLE_RATE",
    "SHIFT_SAMPLES",
    "WINDOW_SAMPLES",
    "HubertFeatures",
    "iter_windows",
]
