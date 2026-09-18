# SPDX-FileCopyrightText: 2026 ai-lip-sync contributors
# SPDX-License-Identifier: Apache-2.0

"""Register a portrait once and cache the ``Avatar`` as an ``.npz``.

    python scripts/register_avatar.py example/image.png --out artifacts/avatars/image.npz
    python scripts/register_avatar.py portrait.png --size 720 1280 --device cuda

The cache holds every array the render loop needs (frame, crop_512, m_c2o,
lmk203, kp_info fields, x_s, f_s, mask_frame) so the player can start without
running the registration models. Load it back with ``load_avatar_npz``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lpavatar.artifacts import ensure_models  # noqa: E402
from lpavatar.registration.avatar import (  # noqa: E402
    Avatar,
    AvatarRegistrar,
    RegistrationConfig,
    RegistrationEngines,
)
from lpavatar.render.keypoints import KPInfo  # noqa: E402
from lpavatar.runtime import OnnxEngine  # noqa: E402

_KP_FIELDS = ("kp", "exp", "scale", "t", "pitch", "yaw", "roll", "rot")
_KP_OPTIONAL = ("pitch_bins", "yaw_bins", "roll_bins")


def save_avatar_npz(avatar: Avatar, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "id": np.array(avatar.id),
        "frame": avatar.frame,
        "has_alpha": np.array(avatar.has_alpha),
        "crop_512": avatar.crop_512,
        "m_c2o": avatar.m_c2o,
        "lmk203": avatar.lmk203,
        "x_s": avatar.x_s,
        "f_s": avatar.f_s,
        "mask_frame": avatar.mask_frame,
    }
    for name in _KP_FIELDS:
        arrays[f"kp_{name}"] = np.asarray(getattr(avatar.kp_info, name))
    for name in _KP_OPTIONAL:
        v = getattr(avatar.kp_info, name)
        if v is not None:
            arrays[f"kp_{name}"] = np.asarray(v)
    np.savez_compressed(path, **arrays)


def load_avatar_npz(path: str | Path) -> Avatar:
    z = np.load(path)
    fields = {name: z[f"kp_{name}"] for name in _KP_FIELDS}
    fields.update({name: z[f"kp_{name}"] for name in _KP_OPTIONAL if f"kp_{name}" in z})
    kp_info = KPInfo(**fields)
    return Avatar(
        id=str(z["id"]),
        frame=z["frame"],
        has_alpha=bool(z["has_alpha"]),
        crop_512=z["crop_512"],
        m_c2o=z["m_c2o"],
        lmk203=z["lmk203"],
        kp_info=kp_info,
        x_s=z["x_s"],
        f_s=z["f_s"],
        mask_frame=z["mask_frame"],
    )


def build_registrar(device: str, config: RegistrationConfig) -> AvatarRegistrar:
    paths = ensure_models()
    eng = RegistrationEngines(
        blaze_face=OnnxEngine(paths["blaze_face"], device=device),
        face_mesh=OnnxEngine(paths["face_mesh"], device=device),
        landmark203=OnnxEngine(paths["landmark203"], device=device),
        appearance_extractor=OnnxEngine(paths["appearance_extractor"], device=device),
        motion_extractor=OnnxEngine(paths["motion_extractor"], device=device),
    )
    return AvatarRegistrar(eng, config)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("portrait", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default artifacts/avatars/<stem>.npz")
    ap.add_argument("--id", default=None, help="avatar id (default: file stem)")
    ap.add_argument("--size", type=int, nargs=2, metavar=("H", "W"), default=(720, 1280))
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = ap.parse_args()

    out = args.out or Path("artifacts") / "avatars" / f"{args.portrait.stem}.npz"
    config = RegistrationConfig(out_size=(args.size[0], args.size[1]))

    t0 = time.perf_counter()
    registrar = build_registrar(args.device, config)
    t1 = time.perf_counter()
    avatar = registrar.register(args.portrait, avatar_id=args.id)
    t2 = time.perf_counter()
    save_avatar_npz(avatar, out)

    print(f"avatar '{avatar.id}' frame {avatar.size} has_alpha={avatar.has_alpha}")
    print(f"  models {t1 - t0:.2f} s, register {t2 - t1:.2f} s")
    kp = avatar.kp_info
    print(
        f"  pitch/yaw/roll deg: {kp.pitch.item():.1f} / {kp.yaw.item():.1f} / {kp.roll.item():.1f}"
    )
    print(f"  -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
