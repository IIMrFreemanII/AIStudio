"""Encoding results and previews: h264 MP4 through ffmpeg (imageio-ffmpeg's bundled binary), PNG and JPEG
through Pillow, WAV through the standard library. Frames are uint8 RGB numpy arrays; audio is float [channels, n]."""

from __future__ import annotations

import io
import subprocess
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


def write_mp4(frames, path: Path, fps: float = 16.0, crf: int = 16, audio: Path | None = None) -> None:
    """[T, H, W, 3] → H.264 MP4 (yuv420p, faststart) that AVFoundation and browsers play. With ``audio`` (a WAV or
    any file ffmpeg reads) the sound is muxed in, cut to the video's length (``mux_audio``)."""
    import imageio_ffmpeg

    frames = to_uint8(frames)
    t, h, w, _ = frames.shape
    if w % 2 or h % 2:  # yuv420p needs even sizes
        frames = frames[:, : h - h % 2, : w - w % 2]
        h, w = frames.shape[1:3]
    target = path.with_name(path.stem + ".silent" + path.suffix) if audio is not None else path
    writer = imageio_ffmpeg.write_frames(
        str(target), (w, h), fps=fps, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p", quality=None,
        output_params=["-crf", str(crf), "-preset", "medium", "-movflags", "+faststart"], macro_block_size=1)
    writer.send(None)
    for f in frames:
        writer.send(np.ascontiguousarray(f))
    writer.close()
    if audio is not None:
        try:
            mux_audio(target, audio, path, seconds=t / fps)
        finally:
            target.unlink(missing_ok=True)


def mux_audio(video: Path, audio: Path, out: Path, seconds: float | None = None) -> None:
    """Copies the H.264 video stream of ``video`` and adds ``audio`` as AAC 192k, cut to ``seconds`` when given
    (the audio can be longer than the video; a shorter one just ends early)."""
    import imageio_ffmpeg

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-i", str(video), "-i", str(audio),
           "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
    if seconds is not None:
        cmd += ["-t", f"{seconds:.3f}"]
    cmd += ["-movflags", "+faststart", str(out)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"ffmpeg could not add the audio: {done.stderr.strip()[-600:]}")


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


def resample(samples: np.ndarray, rate: int, target: int) -> np.ndarray:
    """Band-limited resampling of float [channels, n] by FFT (zero-padding up, truncating down): enough for speech
    references, with no scipy or torchaudio in the image."""
    if rate == target:
        return samples
    n = samples.shape[-1]
    m = max(1, int(round(n * target / rate)))
    spec = np.fft.rfft(samples, axis=-1)
    out = np.zeros(samples.shape[:-1] + (m // 2 + 1,), dtype=spec.dtype)
    keep = min(out.shape[-1], spec.shape[-1])
    out[..., :keep] = spec[..., :keep]
    return (np.fft.irfft(out, n=m, axis=-1) * (m / n)).astype(np.float32)

def tone(seconds: float, sample_rate: int, channels: int, seed: int) -> np.ndarray:
    """A gentle rising sweep (the test tasks' audio), [channels, n]."""
    n = max(1, int(round(seconds * sample_rate)))
    t = np.arange(n, dtype=np.float64) / sample_rate
    f0 = 220.0 + (seed % 97) * 2.0
    phase = 2 * np.pi * (f0 * t + 0.5 * f0 * t * t / max(seconds, 1e-3))  # f0 → 2·f0
    mono = 0.3 * np.sin(phase) * np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.01 + 1e-9)
    return np.repeat(mono[None].astype(np.float32), channels, axis=0)
