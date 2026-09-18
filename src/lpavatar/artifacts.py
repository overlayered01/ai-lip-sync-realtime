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


CUDA_WARP_MODEL = "warp_network_ori_op20"


def ensure_cuda_warp(root: Path | None = None) -> Path:
    """``warp_network_ori.onnx`` re-exported at opset 20 for the CUDA provider.

    The graph ships at opset 17, whose ``GridSample`` the CUDA execution
    provider only implements for 4-D tensors; the warp network samples a 5-D
    volume and fails at run time (``Opset 16-19 versions of this op only
    supports 4-D input tensors``). Opset 20 ``GridSample`` supports 5-D on
    CUDA (measured 18 ms vs 560 ms on CPU, max abs diff vs the original ~1e-6).
    Conversion is done once with ``onnx.version_converter``.
    """
    root = root or models_root()
    dst = model_path(CUDA_WARP_MODEL, root)
    if dst.is_file():
        return dst
    import onnx
    from onnx import version_converter

    src = ensure_models(("warp_network_ori",), root=root)["warp_network_ori"]
    converted = version_converter.convert_version(onnx.load(str(src)), 20)
    tmp = dst.with_suffix(".onnx.part")
    onnx.save(converted, str(tmp))
    tmp.replace(dst)
    return dst


def ensure_fp16(name: str, root: Path | None = None, *, block_ops: tuple[str, ...] = ()) -> Path:
    """``<name>.onnx`` -> ``<name>_fp16.onnx`` (weights and activations fp16, I/O kept
    fp32) via ``onnxconverter-common``. Done once; ``block_ops`` stay fp32.

    Measured on an RTX 5080 (ORT CUDA EP, cuda graph): decoder 27.7 -> 15.7 ms
    (PSNR 78 dB vs fp32), warp 18.8 -> 9.9 ms with ``GridSample`` blocked
    (decoded PSNR 59 dB).
    """
    root = root or models_root()
    dst = model_path(f"{name}_fp16", root)
    if dst.is_file():
        return dst
    import onnx
    from onnxconverter_common import float16

    src = model_path(name, root)
    if not src.is_file():
        raise FileNotFoundError(src)
    m16 = float16.convert_float_to_float16(
        onnx.load(str(src)),
        keep_io_types=True,
        op_block_list=list(float16.DEFAULT_OP_BLOCK_LIST) + list(block_ops),
    )
    tmp = dst.with_suffix(".onnx.part")
    onnx.save(m16, str(tmp))
    tmp.replace(dst)
    return dst


def render_model_paths(
    device: str, root: Path | None = None, *, fp16: bool = True
) -> dict[str, Path]:
    """Graphs for the stitch / warp / decoder stages on ``device``.

    CPU: the original fp32 graphs. CUDA: opset-20 warp (GridSample 5-D) and,
    with ``fp16`` (default), fp16 copies of warp and decoder. Stitch stays fp32
    (0.2 ms). Conversions run once and are cached next to the originals.
    """
    root = root or models_root()
    base = ensure_models(("stitch_network", "warp_network_ori", "decoder"), root=root)
    paths = {
        "stitch": base["stitch_network"],
        "warp": base["warp_network_ori"],
        "decoder": base["decoder"],
    }
    if device == "cuda":
        paths["warp"] = ensure_cuda_warp(root)
        if fp16:
            paths["warp"] = ensure_fp16(CUDA_WARP_MODEL, root, block_ops=("GridSample",))
            paths["decoder"] = ensure_fp16("decoder", root)
    return paths


def warp_model_for(device: str, root: Path | None = None) -> Path:
    """Warp graph to load for ``device``: opset-20 copy on CUDA, original on CPU."""
    if device == "cuda":
        return ensure_cuda_warp(root)
    return ensure_models(("warp_network_ori",), root=root)["warp_network_ori"]


__all__ = [
    "CUDA_WARP_MODEL",
    "PUBLIC_MODELS",
    "ensure_cuda_warp",
    "ensure_fp16",
    "ensure_models",
    "hf_url",
    "model_path",
    "models_root",
    "render_model_paths",
    "warp_model_for",
]


if __name__ == "__main__":  # python -m lpavatar.artifacts  -> download everything
    for name, path in ensure_models().items():
        print(f"{name:22s} {path}")
