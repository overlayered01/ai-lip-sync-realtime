# SPDX-FileCopyrightText: 2024 Ant Group Co., Ltd. (ditto-talkinghead)
# SPDX-FileCopyrightText: 2026 ai-lip-sync contributors
# SPDX-License-Identifier: Apache-2.0
#
# Follows ditto ``core/models/lmdm.py`` (``make_beta``, ``_setup_np``, ``_call_np``).

"""DDIM sampler around the LMDM graph ``lmdm_v0.4_hubert.onnx``.

Graph I/O: ``x (1,80,265)``, ``cond_frame (1,265)``, ``cond (1,80,1103)``,
``time_cond (1,) int64`` -> ``pred_noise (1,80,265)``, ``x_start (1,80,265)``.

Schedule: cosine betas (``n=1000, s=8e-3``), DDIM with ``eta=1``. The
per-step noise is drawn once in ``setup`` and reused for every clip (ditto
does the same); the initial ``x`` is fresh per call.
"""

from __future__ import annotations

import numpy as np

from lpavatar.runtime.onnx_engine import Engine


def make_beta(n_timestep: int = 1000, cosine_s: float = 8e-3) -> np.ndarray:
    steps = np.arange(n_timestep + 1, dtype=np.float64) / n_timestep + cosine_s
    alphas = np.cos(steps / (1 + cosine_s) * np.pi / 2) ** 2
    alphas = alphas / alphas[0]
    betas = 1 - alphas[1:] / alphas[:-1]
    return np.clip(betas, 0, 0.999)


def ddim_time_pairs(n_timestep: int, sampling_timesteps: int) -> list[tuple[int, int]]:
    """``linspace(-1, T-1, steps+1)`` as ints, reversed, paired: ``[(T-1, .), ..., (., -1)]``."""
    times = np.linspace(-1, n_timestep - 1, sampling_timesteps + 1)
    times = list(reversed([int(t) for t in times]))  # torch .int() truncates toward zero
    return list(zip(times[:-1], times[1:], strict=False))


class LMDMSampler:
    def __init__(
        self,
        engine: Engine,
        *,
        seq_frames: int = 80,
        motion_dim: int = 265,
        n_timestep: int = 1000,
        cosine_s: float = 8e-3,
        eta: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.engine = engine
        self.seq_frames = seq_frames
        self.motion_dim = motion_dim
        self.n_timestep = n_timestep
        self.eta = eta
        self.rng = np.random.default_rng(seed)
        self.alphas_cumprod = np.cumprod(1.0 - make_beta(n_timestep, cosine_s))
        self.sampling_timesteps: int | None = None
        self._time_pairs: list[tuple[int, int]] = []
        self._time_cond: list[np.ndarray] = []
        self._alpha_next_sqrt: list[float] = []
        self._sigma: list[float] = []
        self._c: list[float] = []
        self._noise: list[np.ndarray] = []

    def setup(self, sampling_timesteps: int) -> None:
        if self.sampling_timesteps == sampling_timesteps:
            return
        self.sampling_timesteps = sampling_timesteps
        shape = (1, self.seq_frames, self.motion_dim)
        self._time_pairs = ddim_time_pairs(self.n_timestep, sampling_timesteps)
        self._time_cond, self._alpha_next_sqrt, self._sigma, self._c, self._noise = (
            [],
            [],
            [],
            [],
            [],
        )
        for time, time_next in self._time_pairs:
            self._time_cond.append(np.full((1,), time, dtype=np.int64))
            if time_next < 0:
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = self.eta * np.sqrt((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha))
            self._alpha_next_sqrt.append(float(np.sqrt(alpha_next)))
            self._sigma.append(float(sigma))
            self._c.append(float(np.sqrt(1 - alpha_next - sigma**2)))
            self._noise.append(self.rng.standard_normal(shape).astype(np.float32))

    def _step(
        self, x: np.ndarray, cond_frame: np.ndarray, cond: np.ndarray, time_cond: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        out = self.engine(x=x, cond_frame=cond_frame, cond=cond, time_cond=time_cond)
        return np.asarray(out["pred_noise"], np.float32), np.asarray(out["x_start"], np.float32)

    def __call__(
        self, kp_cond: np.ndarray, aud_cond: np.ndarray, sampling_timesteps: int | None = None
    ) -> np.ndarray:
        """``kp_cond (1,265)``, ``aud_cond (1,80,1103)`` -> ``(1,80,265)`` motion vectors."""
        self.setup(sampling_timesteps or self.sampling_timesteps or 10)
        cond_frame = np.ascontiguousarray(kp_cond, dtype=np.float32).reshape(1, self.motion_dim)
        cond = np.ascontiguousarray(aud_cond, dtype=np.float32)
        if cond.shape[:2] != (1, self.seq_frames):
            raise ValueError(f"aud_cond must be (1, {self.seq_frames}, D), got {cond.shape}")
        x = self.rng.standard_normal((1, self.seq_frames, self.motion_dim)).astype(np.float32)
        j = 0
        for i, (_, time_next) in enumerate(self._time_pairs):
            pred_noise, x_start = self._step(x, cond_frame, cond, self._time_cond[i])
            if time_next < 0:
                x = x_start
                continue
            x = (
                x_start * self._alpha_next_sqrt[j]
                + self._c[j] * pred_noise
                + self._sigma[j] * self._noise[j]
            )
            j += 1
        return x


__all__ = ["LMDMSampler", "ddim_time_pairs", "make_beta"]
