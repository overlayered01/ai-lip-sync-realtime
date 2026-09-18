# SPDX-FileCopyrightText: 2026 ai-lip-sync contributors
# SPDX-License-Identifier: Apache-2.0

"""Real-time lip-sync test: play a wav through the speakers while the avatar talks.

    python scripts/realtime_demo.py --audio example/audio.wav --avatar artifacts/avatars/image.npz
    python scripts/realtime_demo.py --audio tts.wav --portrait me.png --live --crop-only

A slimmed-down M2 player (docs/DITTO_AVATAR_HANDOFF.md §6.2 / §11):

- audio 16 kHz source -> 6,480-sample windows every 3,200 samples -> motion thread
  (HuBERT + LMDM) -> bounded queue -> N render workers (stitch/warp/decoder/paste-back,
  each with its own engines) -> ordered frame heap -> display loop
- the audio output device is the master clock: frame ``i`` is shown when the
  playback position reaches ``i * 40 ms``; frames that are already late are
  skipped *before* rendering so the pipeline never falls behind for good
- ``--live`` releases each window only when its audio "would have arrived" from a
  streaming TTS (real-time pacing), so the printed start delay is the true
  TTS-sample-to-display latency; without it all audio is available up front
- keys: ``q`` quit, ``b`` barge-in (cut audio to silence now, ``rewind()`` the
  motion state, keep idling), ``v`` toggle the vad_alpha ramp

CUDA graph capture in ORT is per thread and must not overlap other GPU work, so
the workers capture their engines one at a time at start-up (render workers,
then motion) before anything runs concurrently.

Prints a summary: start delay, achieved fps, per-stage ms, drops.
"""

from __future__ import annotations

import argparse
import heapq
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lpavatar.artifacts import models_root  # noqa: E402
from lpavatar.compose.pasteback import PasteBack  # noqa: E402
from lpavatar.motion.ditto_motion import (  # noqa: E402
    DittoMotionStream,
    FrameControl,
    MotionOutput,
    load_motion_engines,
)
from lpavatar.motion.hubert import (  # noqa: E402
    CHUNK_PAST,
    FRAME_SAMPLES,
    HOP_SAMPLES,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
)
from lpavatar.registration.avatar import RegistrationConfig  # noqa: E402
from lpavatar.render.stage import load_render_engines, render_face  # noqa: E402
from offline_render import load_audio_16k  # noqa: E402
from register_avatar import build_registrar, load_avatar_npz  # noqa: E402

FRAME_SEC = FRAME_SAMPLES / SAMPLE_RATE  # 0.04
PLAY_RATE = 48_000


# --------------------------------------------------------------------------------------------
# shared audio source (model side 16 kHz, playback side 48 kHz), supports barge-in cut


class AudioSource:
    def __init__(self, pcm16k: np.ndarray, tail_windows: int) -> None:
        import soxr

        self.lock = threading.Lock()
        self.pcm16k = np.asarray(pcm16k, np.float32)
        self.pcm48k = soxr.resample(self.pcm16k, SAMPLE_RATE, PLAY_RATE, quality="HQ").astype(
            np.float32
        )
        self.n_windows = int(np.ceil(self.pcm16k.shape[0] / HOP_SAMPLES)) + tail_windows
        self.n_frames = self.n_windows * 5
        self.cut_sample16: int | None = None  # audio after this sample is silence (barge-in)
        self.play_pos48 = 0  # samples handed to the device
        self.started = threading.Event()

    def window(self, w: int) -> np.ndarray:
        """Model window ``w``: 3 past frames + 5 current + 2 future + 80, source-aligned."""
        start = w * HOP_SAMPLES - CHUNK_PAST * FRAME_SAMPLES
        out = np.zeros(WINDOW_SAMPLES, np.float32)
        with self.lock:
            src = self.pcm16k
            lim = (
                src.shape[0] if self.cut_sample16 is None else min(src.shape[0], self.cut_sample16)
            )
        s, e = max(start, 0), min(start + WINDOW_SAMPLES, lim)
        if e > s:
            out[s - start : e - start] = src[s:e]
        return out

    def window_is_silent(self, w: int) -> bool:
        cur = self.window(w)[CHUNK_PAST * FRAME_SAMPLES : (CHUNK_PAST + 5) * FRAME_SAMPLES]
        return float(np.sqrt((cur**2).mean())) < 1e-3

    def playback_chunk(self, n: int) -> np.ndarray:
        with self.lock:
            s = self.play_pos48
            lim = self.pcm48k.shape[0]
            if self.cut_sample16 is not None:
                lim = min(lim, self.cut_sample16 * PLAY_RATE // SAMPLE_RATE)
            e = min(s + n, lim)
            out = np.zeros(n, np.float32)
            if e > s:
                out[: e - s] = self.pcm48k[s:e]
            self.play_pos48 = s + n
        return out

    @property
    def played_seconds(self) -> float:
        with self.lock:
            return self.play_pos48 / PLAY_RATE

    def cut_now(self) -> int:
        """Barge-in: silence from the current playback position. Returns the motion frame."""
        with self.lock:
            pos16 = self.play_pos48 * SAMPLE_RATE // PLAY_RATE
            self.cut_sample16 = pos16
        return int(pos16 // FRAME_SAMPLES)


# --------------------------------------------------------------------------------------------


@dataclass
class Stats:
    motion_ms: list[float] = field(default_factory=list)
    render_ms: list[float] = field(default_factory=list)
    shown: int = 0
    skipped: int = 0  # dropped before rendering (already late)
    dropped: int = 0  # rendered but late at display time
    av_offset_ms: list[float] = field(default_factory=list)
    start_delay_s: float | None = None


class FrameHeap:
    """Rendered frames ordered by index; bounded; barge-in clears it and bumps ``gen``."""

    def __init__(self, maxsize: int) -> None:
        self.lock = threading.Condition()
        self.heap: list[tuple[int, int, np.ndarray]] = []
        self.maxsize = maxsize
        self.gen = 0

    def put(self, gen: int, idx: int, bgr: np.ndarray, stop: threading.Event) -> None:
        with self.lock:
            while len(self.heap) >= self.maxsize and not stop.is_set():
                self.lock.wait(0.05)
            if gen == self.gen:
                heapq.heappush(self.heap, (idx, gen, bgr))

    def peek(self) -> tuple[int, np.ndarray] | None:
        with self.lock:
            while self.heap and self.heap[0][1] != self.gen:
                heapq.heappop(self.heap)
            if not self.heap:
                return None
            idx, _, bgr = self.heap[0]
            return idx, bgr

    def pop(self) -> None:
        with self.lock:
            heapq.heappop(self.heap)
            self.lock.notify_all()

    def clear(self) -> int:
        with self.lock:
            self.heap.clear()
            self.gen += 1
            self.lock.notify_all()
            return self.gen


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--portrait", type=Path)
    ap.add_argument("--avatar", type=Path)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--overlap", type=int, default=75)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--render-workers", type=int, default=2, help="parallel render threads")
    ap.add_argument("--live", action="store_true", help="pace audio arrival like streaming TTS")
    ap.add_argument("--crop-only", action="store_true", help="show the 512 face, skip paste-back")
    ap.add_argument("--no-vad", action="store_true", help="disable the vad_alpha ramp on silence")
    ap.add_argument("--size", type=int, nargs=2, metavar=("H", "W"), default=(720, 1280))
    ap.add_argument("--tail", type=float, default=1.0, help="seconds of silence after the audio")
    ap.add_argument("--scale", type=float, default=1.0, help="display scale")
    ap.add_argument("--output-device", type=int, default=None, help="sounddevice output index")
    ap.add_argument("--fp32", action="store_true", help="CUDA: keep warp/decoder in fp32")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import sounddevice as sd

    root = models_root()
    t0 = time.perf_counter()
    if args.avatar:
        avatar = load_avatar_npz(args.avatar)
    elif args.portrait:
        avatar = build_registrar(
            args.device, RegistrationConfig(out_size=tuple(args.size))
        ).register(args.portrait)
    else:
        sys.exit("give --portrait or --avatar")
    hubert, lmdm = load_motion_engines(args.device, root)
    stream = DittoMotionStream.from_ditto_cfg(
        root / "v0.4_hubert_cfg_trt_online.pkl",
        avatar,
        hubert=hubert,
        lmdm=lmdm,
        overlap_v2=args.overlap,
        sampling_timesteps=args.steps,
        seed=args.seed,
    )
    n_render = max(1, args.render_workers)
    engine_sets = [
        load_render_engines(args.device, root, fp16=not args.fp32) for _ in range(n_render)
    ]
    paste = PasteBack(avatar.frame, avatar.m_c2o, avatar.mask_frame)
    print(
        f"setup {time.perf_counter() - t0:.1f} s; lookahead {stream.audio_lookahead_frames} "
        f"frames; {n_render} render worker(s)"
    )

    audio = load_audio_16k(args.audio)
    source = AudioSource(audio, tail_windows=int(np.ceil(args.tail / 0.2)))
    stats = Stats()
    stop = threading.Event()
    barge = threading.Event()
    vad_enabled = not args.no_vad
    motion_q: queue.Queue[tuple[int, MotionOutput] | None] = queue.Queue(maxsize=10)
    frames = FrameHeap(maxsize=15)
    wall = {"start": time.perf_counter()}
    capture_lock = threading.Lock()  # one CUDA graph capture at a time
    renders_captured = threading.Barrier(n_render + 1)  # render workers + motion
    motion_ready = threading.Event()
    render_done = threading.Semaphore(0)
    errors: list[str] = []

    # -- motion thread ---------------------------------------------------------------------
    def motion_worker() -> None:
        renders_captured.wait()  # every render worker has captured and is idle
        stream.warmup()  # LMDM graph capture in this thread
        motion_ready.set()
        wall["start"] = time.perf_counter()
        w = 0
        alpha = 1.0
        while not stop.is_set() and w < source.n_windows:
            if args.live:
                # this window's audio (incl. HuBERT future + shift) has arrived at:
                arrive = (
                    wall["start"]
                    + (w * HOP_SAMPLES + WINDOW_SAMPLES - CHUNK_PAST * FRAME_SAMPLES) / SAMPLE_RATE
                )
                while time.perf_counter() < arrive and not stop.is_set():
                    time.sleep(0.002)
            if barge.is_set():
                barge.clear()
                cut_frame = source.cut_now()
                f = stream.rewind(cut_frame)
                w = stream.windows_pushed
                while True:  # drop motion not yet rendered
                    try:
                        motion_q.get_nowait()
                    except queue.Empty:
                        break
                gen = frames.clear()  # drop rendered-but-unshown frames, invalidate in-flight
                print(f"[barge-in] played frame {cut_frame} -> rewound to {f}, gen {gen}")
                continue
            target = 0.0 if (vad_enabled and source.window_is_silent(w)) else 1.0
            alpha = float(np.clip(alpha + np.sign(target - alpha) * 0.34, 0.0, 1.0))
            t1 = time.perf_counter()
            outs = stream.push_window(source.window(w), FrameControl(vad_alpha=alpha))
            stats.motion_ms.append(1000 * (time.perf_counter() - t1))
            gen = frames.gen
            for o in outs:
                while not stop.is_set():
                    try:
                        motion_q.put((gen, o), timeout=0.1)
                        break
                    except queue.Full:
                        continue
            w += 1
        for _ in range(n_render):
            motion_q.put(None)

    # -- render workers ----------------------------------------------------------------------
    def render_worker(engines) -> None:  # noqa: ANN001
        with capture_lock:  # stitch / warp / decoder graph capture in this thread
            for _ in range(2):
                render_face(avatar.f_s, avatar.x_s, avatar.x_s, engines=engines)
        renders_captured.wait()
        motion_ready.wait()
        try:
            while not stop.is_set():
                try:
                    item = motion_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    break
                gen, o = item
                if gen != frames.gen:
                    continue
                if source.started.is_set():
                    late = source.played_seconds - o.frame_idx * FRAME_SEC
                    if late > FRAME_SEC:  # would be shown late anyway: skip the GPU work
                        stats.skipped += 1
                        continue
                t1 = time.perf_counter()
                face = render_face(avatar.f_s, avatar.x_s, o.x_d, engines=engines)
                if args.crop_only:
                    img = np.clip(face, 0, 255).astype(np.uint8)
                else:
                    img = paste(face)
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                if args.scale != 1.0:
                    bgr = cv2.resize(bgr, None, fx=args.scale, fy=args.scale)
                stats.render_ms.append(1000 * (time.perf_counter() - t1))
                frames.put(gen, o.frame_idx, bgr, stop)
        finally:
            render_done.release()

    def guarded(fn, *a):  # noqa: ANN001
        def run() -> None:
            try:
                fn(*a)
            except Exception:  # noqa: BLE001
                errors.append(traceback.format_exc())
                stop.set()

        return run

    for i, eng in enumerate(engine_sets):
        threading.Thread(target=guarded(render_worker, eng), daemon=True, name=f"render{i}").start()
    threading.Thread(target=guarded(motion_worker), daemon=True, name="motion").start()

    # -- audio device = master clock ---------------------------------------------------------
    def audio_cb(outdata, nframes, time_info, status):  # noqa: ANN001
        outdata[:, 0] = source.playback_chunk(nframes)

    out_stream = sd.OutputStream(
        samplerate=PLAY_RATE,
        channels=1,
        dtype="float32",
        blocksize=480,
        device=args.output_device,
        callback=audio_cb,
    )

    # -- display loop --------------------------------------------------------------------------
    win = "ai-lip-sync realtime"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    last: np.ndarray | None = None
    fps_t, fps_n = time.perf_counter(), 0
    finished_workers = 0
    try:
        while not stop.is_set():
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("b"):
                barge.set()
            if key == ord("v"):
                vad_enabled = not vad_enabled
                print(f"[vad] {'on' if vad_enabled else 'off'}")

            while render_done.acquire(blocking=False):
                finished_workers += 1
            head = frames.peek()
            if head is None:
                if finished_workers >= n_render and (
                    not source.started.is_set()
                    or source.played_seconds > source.n_frames * FRAME_SEC + 0.3
                ):
                    break
                if last is not None:
                    cv2.imshow(win, last)
                time.sleep(0.002)
                continue
            idx, bgr = head
            if not source.started.is_set():
                # first frame ready -> start the clock (audio) now
                stats.start_delay_s = time.perf_counter() - wall["start"]
                out_stream.start()
                source.started.set()
                print(f"audio started {stats.start_delay_s:.2f} s after the first sample arrived")
            now = source.played_seconds
            due = idx * FRAME_SEC
            if now + 0.002 < due:
                if last is not None:
                    cv2.imshow(win, last)
                continue
            frames.pop()
            if now - due > FRAME_SEC:
                stats.dropped += 1
                continue
            stats.av_offset_ms.append(1000 * (now - due))
            cv2.imshow(win, bgr)
            last = bgr
            stats.shown += 1
            fps_n += 1
            if time.perf_counter() - fps_t >= 1.0:
                fps_val = fps_n / (time.perf_counter() - fps_t)
                fps_t, fps_n = time.perf_counter(), 0
                mm = np.mean(stats.motion_ms[-10:]) if stats.motion_ms else 0
                rm = np.mean(stats.render_ms[-25:]) if stats.render_ms else 0
                print(
                    f"t={now:6.2f}s shown={stats.shown:4d} fps={fps_val:4.1f} "
                    f"motion={mm:5.0f} ms/win render={rm:5.1f} ms/frame "
                    f"q_motion={motion_q.qsize():2d} q_frame={len(frames.heap):2d} "
                    f"skip={stats.skipped} drop={stats.dropped} "
                    f"av={np.mean(stats.av_offset_ms[-25:]):+5.1f} ms"
                )
    finally:
        stop.set()
        try:
            out_stream.stop()
            out_stream.close()
        except Exception:  # noqa: BLE001
            pass
        cv2.destroyAllWindows()

    for err in errors:
        print("worker error:\n" + err)
    n_src_frames = int(np.ceil(audio.shape[0] / FRAME_SAMPLES))
    print("\n=== summary ===")
    print(
        f"audio {audio.shape[0] / SAMPLE_RATE:.2f} s ({n_src_frames} frames), "
        f"live pacing: {args.live}, render workers: {n_render}"
    )
    if stats.start_delay_s is not None:
        print(f"start delay (first TTS sample -> first frame shown): {stats.start_delay_s:.2f} s")
    if stats.motion_ms:
        print(f"motion: median {np.median(stats.motion_ms):.0f} ms / window (budget 200 ms)")
    if stats.render_ms:
        print(f"render: median {np.median(stats.render_ms):.1f} ms / frame per worker")
    print(
        f"frames shown {stats.shown}, skipped before render {stats.skipped}, "
        f"dropped at display {stats.dropped}"
    )
    if stats.av_offset_ms:
        print(
            f"A/V offset: mean {np.mean(stats.av_offset_ms):+.1f} ms, "
            f"max {np.max(stats.av_offset_ms):+.1f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
