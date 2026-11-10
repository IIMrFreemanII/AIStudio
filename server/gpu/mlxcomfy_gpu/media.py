"""Encoding results and previews: h264 MP4 through ffmpeg (imageio-ffmpeg's bundled binary), PNG and JPEG
through Pillow, WAV through the standard library. Frames are uint8 RGB numpy arrays; audio is float [channels, n]."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
from PIL import Image


def to_uint8(frames) -> np.ndarray:
    """float [0, 1] or uint8 → uint8, any leading shape; last axis RGB."""
    a = np.asarray(frames)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return a


def write_mp4(frames, path: Path, fps: float = 16.0, crf: int = 16) -> None:
    """[T, H, W, 3] → H.264 MP4 (yuv420p, faststart) that AVFoundation and browsers play."""
    import imageio_ffmpeg

    frames = to_uint8(frames)
    t, h, w, _ = frames.shape
    if w % 2 or h % 2:  # yuv420p needs even sizes
        frames = frames[:, : h - h % 2, : w - w % 2]
        h, w = frames.shape[1:3]
    writer = imageio_ffmpeg.write_frames(
        str(path), (w, h), fps=fps, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p", quality=None,
        output_params=["-crf", str(crf), "-preset", "medium", "-movflags", "+faststart"], macro_block_size=1)
    writer.send(None)
    for f in frames:
        writer.send(np.ascontiguousarray(f))
    writer.close()


def write_png(image, path: Path) -> None:
    Image.fromarray(to_uint8(image)).save(path, format="PNG", compress_level=4)


def jpeg_preview(image, max_side: int = 512, quality: int = 70) -> bytes:
    """A small JPEG for live previews (latest-wins, so cheap beats pretty)."""
    img = image if isinstance(image, Image.Image) else Image.fromarray(to_uint8(image))
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.BILINEAR)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def write_wav(samples, sample_rate: int, path: Path) -> None:
    """float audio in -1…1, [channels, n] (or [n] for mono) → 16-bit PCM WAV."""
    a = np.asarray(samples, dtype=np.float32)
    if a.ndim == 1:
        a = a[None]
    if a.ndim != 2 or a.shape[0] > 8:
        raise ValueError(f"expected audio as [channels, samples], got shape {a.shape}")
    pcm = (np.clip(np.nan_to_num(a), -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(pcm.shape[0])
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(np.ascontiguousarray(pcm.T).tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """A 16-bit PCM WAV (what the app uploads) → float [channels, n] and the sample rate."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("expected 16-bit PCM WAV")
        n, ch, rate = w.getnframes(), w.getnchannels(), w.getframerate()
        data = np.frombuffer(w.readframes(n), dtype="<i2").reshape(-1, ch).T
    return data.astype(np.float32) / 32768.0, rate


def tone(seconds: float, sample_rate: int, channels: int, seed: int) -> np.ndarray:
    """A gentle rising sweep (the test tasks' audio), [channels, n]."""
    n = max(1, int(round(seconds * sample_rate)))
    t = np.arange(n, dtype=np.float64) / sample_rate
    f0 = 220.0 + (seed % 97) * 2.0
    phase = 2 * np.pi * (f0 * t + 0.5 * f0 * t * t / max(seconds, 1e-3))  # f0 → 2·f0
    mono = 0.3 * np.sin(phase) * np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.01 + 1e-9)
    return np.repeat(mono[None].astype(np.float32), channels, axis=0)
