# SPDX-FileCopyrightText: 2026 lpavatar contributors
# SPDX-License-Identifier: Apache-2.0

"""Minimal engine abstraction: named numpy arrays in, named numpy arrays out.

``OnnxEngine`` wraps an ``onnxruntime.InferenceSession`` and validates
input names / dtypes. It runs on the CPU or CUDA execution provider; the
choice is a constructor argument so the same registration code can be
exercised on a CPU-only development machine and on the deployment GPU.

A TensorRT implementation of the same ``Engine`` protocol is the natural
next step for the real-time path (see ``docs/REIMPLEMENTATION_PLAN.md``
R5/R6); it is not part of this module because it cannot be verified
without a GPU.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

_ORT_TO_NUMPY: dict[str, type] = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(bool)": np.bool_,
}


@dataclass(slots=True, frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int | str | None, ...]
    dtype: type


class Engine(Protocol):
    """Callable model: ``engine(**inputs) -> {output_name: array}``."""

    @property
    def inputs(self) -> dict[str, TensorSpec]: ...

    @property
    def outputs(self) -> dict[str, TensorSpec]: ...

    def __call__(self, **inputs: np.ndarray) -> dict[str, np.ndarray]: ...


class OnnxEngine:
    """ONNX Runtime backed ``Engine``.

    Args:
        path: ``.onnx`` file.
        device: ``"cpu"`` or ``"cuda"``. CUDA falls back to CPU for ops the
            CUDA EP does not support (ORT default behaviour).
        device_id: CUDA device index.
        intra_op_threads: CPU thread count (``None`` = ORT default).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        device: str = "cpu",
        device_id: int = 0,
        intra_op_threads: int | None = None,
    ) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if intra_op_threads is not None:
            opts.intra_op_num_threads = intra_op_threads
        if device == "cuda":
            providers: list[Any] = [
                ("CUDAExecutionProvider", {"device_id": device_id}),
                "CPUExecutionProvider",
            ]
        elif device == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")

        self.path = Path(path)
        self.device = device
        self.session = ort.InferenceSession(str(self.path), sess_options=opts, providers=providers)
        self._inputs = {
            i.name: TensorSpec(i.name, tuple(i.shape), _ORT_TO_NUMPY[i.type])
            for i in self.session.get_inputs()
        }
        self._outputs = {
            o.name: TensorSpec(o.name, tuple(o.shape), _ORT_TO_NUMPY[o.type])
            for o in self.session.get_outputs()
        }
        self._output_names = list(self._outputs)

    @property
    def inputs(self) -> dict[str, TensorSpec]:
        return self._inputs

    @property
    def outputs(self) -> dict[str, TensorSpec]:
        return self._outputs

    def __call__(self, **inputs: np.ndarray) -> dict[str, np.ndarray]:
        feed = self._prepare(inputs)
        results = self.session.run(self._output_names, feed)
        return dict(zip(self._output_names, results, strict=True))

    def _prepare(self, inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        missing = set(self._inputs) - set(inputs)
        extra = set(inputs) - set(self._inputs)
        if missing or extra:
            raise ValueError(
                f"{self.path.name}: input mismatch; missing={sorted(missing)} extra={sorted(extra)}"
            )
        feed: dict[str, np.ndarray] = {}
        for name, arr in inputs.items():
            spec = self._inputs[name]
            a = np.ascontiguousarray(arr, dtype=spec.dtype)
            for dim, want in zip(a.shape, spec.shape, strict=False):
                if isinstance(want, int) and want != dim:
                    raise ValueError(
                        f"{self.path.name}: input {name!r} shape {a.shape} "
                        f"does not match {spec.shape}"
                    )
            if len(a.shape) != len(spec.shape):
                raise ValueError(
                    f"{self.path.name}: input {name!r} rank {a.ndim} does not match {spec.shape}"
                )
            feed[name] = a
        return feed

    def __repr__(self) -> str:
        return f"OnnxEngine({self.path.name}, device={self.device})"


__all__ = ["Engine", "OnnxEngine", "TensorSpec"]
