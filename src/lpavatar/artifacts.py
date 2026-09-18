# SPDX-FileCopyrightText: 2026 lpavatar contributors
# SPDX-License-Identifier: Apache-2.0

"""Download the public ONNX graphs this package uses.

All files come from the Apache-2.0 Hugging Face repo
``digital-avatar/ditto-talkinghead`` (folder ``ditto_onnx/``), which is
public and needs no login:

    blaze_face, face_mesh                     MediaPipe (Apache-2.0)
    landmark203, appearance_extractor,
    motion_extractor, stitch_network,
    warp_network_ori, decoder                 LivePortrait (MIT)

MODNet is not included there; point ``MODNET_ONNX`` at your own export of
the official model (Apache-2.0) if matting is needed.

Storage root: ``$LPAVATAR_MODELS`` or ``<repo>/artifacts/lpavatar``.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

HF_REPO = "digital-avatar/ditto-talkinghead"
HF_FOLDER = "ditto_onnx"
PUBLIC_MODELS: tuple[str, ...] = (
    "blaze_face",
    "face_mesh",
    "landmark203",
    "appearance_extractor",
    "motion_extractor",
    "stitch_network",
    "warp_network_ori",
    "decoder",
)


def models_root() -> Path:
    env = os.environ.get("LPAVATAR_MODELS")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "artifacts" / "lpavatar"


def model_path(name: str, root: Path | None = None) -> Path:
    return (root or models_root()) / f"{name}.onnx"


def hf_url(name: str) -> str:
    return f"https://huggingface.co/{HF_REPO}/resolve/main/{HF_FOLDER}/{name}.onnx"


def ensure_models(
    names: tuple[str, ...] = PUBLIC_MODELS, *, root: Path | None = None
) -> dict[str, Path]:
    """Download any missing graphs; return ``{name: path}``."""
    root = root or models_root()
    root.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for name in names:
        if name not in PUBLIC_MODELS:
            raise KeyError(f"unknown model {name!r}; known: {PUBLIC_MODELS}")
        p = model_path(name, root)
        if not p.is_file():
            tmp = p.with_suffix(".onnx.part")
            urllib.request.urlretrieve(hf_url(name), tmp)
            tmp.replace(p)
        out[name] = p
    return out


__all__ = ["PUBLIC_MODELS", "ensure_models", "hf_url", "model_path", "models_root"]


if __name__ == "__main__":  # python -m lpavatar.artifacts  -> download everything
    for name, path in ensure_models().items():
        print(f"{name:22s} {path}")
