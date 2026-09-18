# SPDX-FileCopyrightText: 2026 ai-lip-sync contributors
# SPDX-License-Identifier: Apache-2.0

"""Download every model this project needs (all Apache-2.0 / MIT, no login).

    python scripts/download_models.py            # renderer graphs + motion models + cfg
    python scripts/download_models.py --render   # only the 8 LivePortrait/MediaPipe graphs
    python scripts/download_models.py --motion   # only HuBERT + LMDM + online cfg

Files land under ``$LPAVATAR_MODELS`` (default ``<repo>/artifacts/lpavatar``).
Source: https://huggingface.co/digital-avatar/ditto-talkinghead (Apache-2.0).

Not downloaded on purpose: ``ditto_pytorch/aux_models/2d106det.onnx`` and
``det_10g.onnx`` (InsightFace, non-commercial). See docs/DITTO_AVATAR_HANDOFF.md §2.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lpavatar.artifacts import HF_REPO, PUBLIC_MODELS, ensure_models, models_root  # noqa: E402

# name -> path inside the HF repo
MOTION_FILES: dict[str, str] = {
    "hubert_streaming_fix_kv.onnx": "ditto_pytorch/aux_models/hubert_streaming_fix_kv.onnx",
    "lmdm_v0.4_hubert.onnx": "ditto_onnx/lmdm_v0.4_hubert.onnx",
    "v0.4_hubert_cfg_trt_online.pkl": "ditto_cfg/v0.4_hubert_cfg_trt_online.pkl",
    "v0.4_hubert_cfg_trt.pkl": "ditto_cfg/v0.4_hubert_cfg_trt.pkl",
}


def _hf_url(repo_path: str) -> str:
    return f"https://huggingface.co/{HF_REPO}/resolve/main/{repo_path}"


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")

    def hook(blocks: int, block_size: int, total: int) -> None:
        if total > 0:
            done = min(blocks * block_size, total)
            sys.stdout.write(f"\r  {dst.name}: {done / 1e6:8.1f} / {total / 1e6:.1f} MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    sys.stdout.write("\n")
    tmp.replace(dst)


def ensure_motion_models(root: Path | None = None) -> dict[str, Path]:
    root = root or models_root()
    out: dict[str, Path] = {}
    for name, repo_path in MOTION_FILES.items():
        dst = root / name
        if not dst.is_file():
            _download(_hf_url(repo_path), dst)
        out[name] = dst
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--render", action="store_true", help="only the 8 renderer/registration graphs")
    ap.add_argument(
        "--motion", action="store_true", help="only HuBERT, LMDM and the ditto cfg pickles"
    )
    ap.add_argument("--root", type=Path, default=None, help="override $LPAVATAR_MODELS")
    args = ap.parse_args()

    root = args.root or models_root()
    print(f"models root: {root}")
    do_render = args.render or not args.motion
    do_motion = args.motion or not args.render

    if do_render:
        print(f"renderer graphs ({len(PUBLIC_MODELS)}):")
        for name, path in ensure_models(root=root).items():
            print(f"  {name:24s} {path.stat().st_size / 1e6:8.1f} MB")
    if do_motion:
        print("motion models:")
        for name, path in ensure_motion_models(root=root).items():
            print(f"  {name:36s} {path.stat().st_size / 1e6:8.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
