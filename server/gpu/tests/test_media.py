"""Encoding helpers: the MP4 with sound."""

import subprocess

import imageio_ffmpeg
import numpy as np

from aistudio_gpu.media import mux_audio, tone, write_mp4, write_wav


def _streams(path) -> str:
    return subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(path)], capture_output=True, text=True).stderr


def test_write_mp4_with_audio_has_both_streams_and_the_videos_length(tmp_path):
    frames = np.random.default_rng(0).random((16, 32, 48, 3)).astype(np.float32)
    wav = tmp_path / "speech.wav"
    write_wav(tone(2.0, 16000, 1, 0), 16000, wav)  # longer than the 1 s of video: cut to it
    out = tmp_path / "clip.mp4"
    write_mp4(frames, out, fps=16, audio=wav)
    info = _streams(out)
    assert "Video: h264" in info and "Audio: aac" in info
    n, secs = imageio_ffmpeg.count_frames_and_secs(str(out))
    assert n == 16 and abs(secs - 1.0) < 0.1
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".mp4"] == ["clip.mp4"]  # the silent temp file is gone


def test_mux_audio_copies_the_video_and_cuts_to_seconds(tmp_path):
    silent, wav, out = tmp_path / "silent.mp4", tmp_path / "a.wav", tmp_path / "out.mp4"
    write_mp4(np.zeros((16, 32, 32, 3), np.float32), silent, fps=16)
    write_wav(tone(3.0, 16000, 1, 0), 16000, wav)
    mux_audio(silent, wav, out, seconds=1.0)
    assert "Audio: aac" in _streams(out)
    assert abs(imageio_ffmpeg.count_frames_and_secs(str(out))[1] - 1.0) < 0.1


def test_mux_audio_reports_ffmpegs_error(tmp_path):
    import pytest

    with pytest.raises(RuntimeError, match="ffmpeg could not add the audio"):
        mux_audio(tmp_path / "missing.mp4", tmp_path / "missing.wav", tmp_path / "out.mp4")


def test_s2v_renders_at_multiples_of_64_so_nothing_is_padded():
    from aistudio_gpu.tasks.wan_s2v import model_size

    assert model_size(832, 480) == (832, 448)  # the movie's 16:9 clips: cropped, not pillarboxed
    assert model_size(1280, 720) == (1280, 704) and model_size(480, 832) == (448, 832) and model_size(50, 50) == (64, 64)
