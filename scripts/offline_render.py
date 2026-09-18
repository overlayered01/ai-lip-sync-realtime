# SPDX-FileCopyrightText: 2026 ai-lip-sync contributors
# SPDX-License-Identifier: Apache-2.0

"""wav + portrait -> mp4 through the *online* motion state machine (CPU ok).

    python scripts/offline_render.py --audio example/audio.wav --portrait example/image.png \
        --out out/example.mp4 [--max-seconds 4] [--overlap 75] [--steps 10] [--device cuda]

Runs exactly the live path: 6,480-sample windows every 3,200 samples pushed
one by one, frames rendered as they are released. Prints per-stage timing
and a lip-sync sanity metric (correlation between per-frame audio RMS and
lip displacement). Audio is muxed with ffmpeg when it is on PATH.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lpavatar.artifacts import models_root  # noqa: E402
from lpavatar.compose.pasteback import pasteback  # noqa: E402
from lpavatar.motion.ditto_motion import LIP_KP, DittoMotionStream, FrameControl  # noqa: E402
from lpavatar.motion.hubert import FRAME_SAMPLES, SAMPLE_RATE, iter_windows  # noqa: E402
from lpavatar.registration.avatar import RegistrationConfig  # noqa: E402
from lpavatar.render.stage import RenderEngines, render_face  # noqa: E402
from lpavatar.runtime import OnnxEngine  # noqa: E402
from register_avatar import build_registrar, load_avatar_npz  # noqa: E402


def load_audio_16k(path: Path) -> np.ndarray:
    import soundfile as sf
    import soxr

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        mono = soxr.resample(mono, sr, SAMPLE_RATE, quality="HQ")
    return np.ascontiguousarray(mono, dtype=np.float32)


def lipsync_metric(audio: np.ndarray, lip_disp: np.ndarray) -> float:
    n = min(len(lip_disp), audio.shape[0] // FRAME_SAMPLES)
    if n < 10:
        return float("nan")
    rms = np.sqrt((audio[: n * FRAME_SAMPLES].reshape(n, FRAME_SAMPLES) ** 2).mean(1))
    return float(np.corrcoef(rms, lip_disp[:n])[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--portrait", type=Path, help="portrait image (registered on the fly)")
    ap.add_argument("--avatar", type=Path, help="cached avatar .npz from register_avatar.py")
    ap.add_argument("--out", type=Path, default=Path("out/render.mp4"))
    ap.add_argument("--cfg", type=Path, default=None, help="ditto cfg pickle (default: online)")
    ap.add_argument("--overlap", type=int, default=75, help="overlap_v2: 75 (480 ms) or 70")
    ap.add_argument("--steps", type=int, default=10, help="DDIM sampling steps")
    ap.add_argument("--emo", type=int, default=None, help="fixed emotion label 0..7")
    ap.add_argument("--condition", choices=("reference", "avatar"), default="reference")
    ap.add_argument(
        "--vad-silence",
        action="store_true",
        help="hold lips at the source pose on windows with near-zero RMS",
    )
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--size", type=int, nargs=2, metavar=("H", "W"), default=(720, 1280))
    ap.add_argument(
        "--crop-only", action="store_true", help="write the 512 face instead of paste-back"
    )
    ap.add_argument("--snapshots", type=int, default=4, help="PNG frames written next to --out")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = models_root()
    cfg_path = args.cfg or root / "v0.4_hubert_cfg_trt_online.pkl"
    for name in ("hubert_streaming_fix_kv.onnx", "lmdm_v0.4_hubert.onnx"):
        if not (root / name).is_file():
            sys.exit(f"missing {root / name}; run scripts/download_models.py --motion")

    t0 = time.perf_counter()
    if args.avatar:
        avatar = load_avatar_npz(args.avatar)
    elif args.portrait:
        registrar = build_registrar(args.device, RegistrationConfig(out_size=tuple(args.size)))
        avatar = registrar.register(args.portrait)
    else:
        sys.exit("give --portrait or --avatar")
    if avatar.kp_info.pitch_bins is None:
        sys.exit("avatar cache predates pose logits; re-run register_avatar.py")
    t_reg = time.perf_counter() - t0

    t0 = time.perf_counter()
    stream = DittoMotionStream.from_ditto_cfg(
        cfg_path,
        avatar,
        hubert=OnnxEngine(root / "hubert_streaming_fix_kv.onnx", device=args.device),
        lmdm=OnnxEngine(root / "lmdm_v0.4_hubert.onnx", device=args.device),
        condition_on=args.condition,
        emo=args.emo,
        overlap_v2=args.overlap,
        sampling_timesteps=args.steps,
        seed=args.seed,
    )
    engines = RenderEngines(
        stitch=OnnxEngine(root / "stitch_network.onnx", device=args.device),
        warp=OnnxEngine(root / "warp_network_ori.onnx", device=args.device),
        decoder=OnnxEngine(root / "decoder.onnx", device=args.device),
    )
    t_setup = time.perf_counter() - t0
    print(f"setup: register {t_reg:.1f} s, models + LMDM warm-up {t_setup:.1f} s")
    print(
        f"overlap_v2={stream.cfg.overlap_v2} valid={stream.cfg.valid_clip_len} "
        f"lookahead={stream.audio_lookahead_frames} frames "
        f"({stream.audio_lookahead_frames * 40} ms)"
    )

    audio = load_audio_16k(args.audio)
    if args.max_seconds:
        audio = audio[: int(args.max_seconds * SAMPLE_RATE)]
    n_frames_expected = int(np.ceil(audio.shape[0] / FRAME_SAMPLES))
    print(f"audio {audio.shape[0] / SAMPLE_RATE:.2f} s -> {n_frames_expected} frames")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.crop_only:
        h, w = 512, 512
    else:
        h, w = avatar.frame.shape[:2]
    tmp_video = args.out.with_suffix(".video.mp4")
    writer = cv2.VideoWriter(str(tmp_video), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h))
    if not writer.isOpened():
        sys.exit("cv2.VideoWriter failed to open")

    snap_every = max(1, n_frames_expected // max(args.snapshots, 1))
    lip_src = avatar.kp_info.exp.reshape(21, 3)[list(LIP_KP)]
    lip_disp: list[float] = []
    t_motion = t_render = 0.0
    frames_written = 0
    last_report = time.perf_counter()

    def render_outputs(outs) -> None:
        nonlocal t_render, frames_written
        for o in outs:
            t1 = time.perf_counter()
            face = render_face(avatar.f_s, avatar.x_s, o.x_d, engines=engines)
            if args.crop_only:
                img = np.clip(face, 0, 255).astype(np.uint8)
            else:
                img = pasteback(avatar.frame, face, avatar.m_c2o, avatar.mask_frame)
            t_render += time.perf_counter() - t1
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if o.frame_idx % snap_every == 0:
                cv2.imwrite(
                    str(args.out.with_name(f"{args.out.stem}_f{o.frame_idx:04d}.png")),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                )
            lip = o.motion.exp.reshape(21, 3)[list(LIP_KP)]
            lip_disp.append(float(np.linalg.norm(lip - lip_src)))
            frames_written += 1

    for i, win in enumerate(iter_windows(audio)):
        ctrl = None
        if args.vad_silence:
            cur = win[3 * FRAME_SAMPLES : 8 * FRAME_SAMPLES]
            ctrl = FrameControl(vad_alpha=0.0 if np.sqrt((cur**2).mean()) < 1e-3 else 1.0)
        t1 = time.perf_counter()
        outs = stream.push_window(win, ctrl)
        t_motion += time.perf_counter() - t1
        render_outputs(outs)
        if time.perf_counter() - last_report > 10:
            last_report = time.perf_counter()
            print(
                f"  window {i + 1}: {frames_written} frames written "
                f"(motion {t_motion:.1f} s, render {t_render:.1f} s)"
            )
    t1 = time.perf_counter()
    outs = stream.flush()
    t_motion += time.perf_counter() - t1
    render_outputs(outs)
    writer.release()

    n_win = stream.windows_pushed
    print(f"frames written: {frames_written} (expected about {n_frames_expected})")
    print(
        f"motion: {t_motion:.1f} s total, {1000 * t_motion / max(n_win, 1):.0f} ms per window "
        f"(HuBERT + LMDM {stream.cfg.sampling_timesteps} steps "
        f"every {stream.cfg.valid_clip_len} frames)"
    )
    print(
        f"render: {t_render:.1f} s total, "
        f"{1000 * t_render / max(frames_written, 1):.0f} ms per frame"
    )
    corr = lipsync_metric(audio, np.asarray(lip_disp))
    print(
        f"lip-sync sanity: corr(audio RMS, lip displacement) = {corr:.3f} "
        f"(> 0.3 expected for speech; ~0 means no lip-sync)"
    )

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg,
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(tmp_video),
            "-i",
            str(args.audio),
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(args.out),
        ]
        if args.max_seconds:
            cmd[cmd.index("-shortest") : cmd.index("-shortest")] = [
                "-t",
                f"{frames_written / 25:.3f}",
            ]
        subprocess.run(cmd, check=True)
        tmp_video.unlink(missing_ok=True)
        print(f"-> {args.out}")
    else:
        tmp_video.replace(args.out)
        print(f"-> {args.out} (no ffmpeg on PATH: video only, no audio track)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
