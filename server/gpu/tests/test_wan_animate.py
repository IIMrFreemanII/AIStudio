"""Wan-Animate-2's helpers and the fake worker: the clip cut to start … start + seconds, 24 fps, the picture first,
the clip's sound kept when asked; the two-part caption; the colour match."""

import subprocess

import numpy as np
import pytest


class _QuietSink:
    def status(self, *a): pass
    def progress(self, *a): pass
    def preview(self, *a): pass
    def check(self): pass


def _frames(path):
    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    w, h = meta["size"]
    return meta, [np.frombuffer(f, np.uint8).reshape(h, w, 3) for f in reader]


def _has_audio(path) -> bool:
    import imageio_ffmpeg

    return "Audio:" in subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(path)], capture_output=True,
                                      text=True).stderr


@pytest.fixture()
def clip(tmp_path):
    """2 s at 25 fps with a tone: red for the first second, green for the second."""
    from aistudio_gpu.media import tone, write_mp4, write_wav

    frames = np.zeros((50, 48, 64, 3), np.float32)
    frames[:25, :, :, 0] = 1.0
    frames[25:, :, :, 1] = 1.0
    wav = tmp_path / "sound.wav"
    write_wav(tone(2.0, 16000, 1, 3), 16000, wav)
    path = tmp_path / "dance.mp4"
    write_mp4(frames, path, fps=25, audio=wav)
    return path


def test_fake_animate_cuts_the_clip_and_keeps_its_sound(tmp_path, clip, monkeypatch):
    from PIL import Image

    from aistudio_gpu.tasks import TASKS, make_worker

    monkeypatch.setenv("AISTUDIO_FAKE_MODELS", "1")
    spec = TASKS["wan-animate2"]
    pic = tmp_path / "her.png"
    Image.new("RGB", (40, 80), (0, 0, 255)).save(pic)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out = make_worker(spec.family).run(spec, spec.normalize({"start": 1.0, "seconds": 1.0}),
                                       {"image": pic, "video": clip}, _QuietSink(), out_dir)
    meta, frames = _frames(out.path)
    assert meta["fps"] == 24 and len(frames) == 24 and meta["size"] == (64, 48)  # 1 s of the clip at 24 fps
    assert frames[0][20, 20, 2] > 200  # opens on the (blue) picture
    assert frames[-1][20, 20, 1] > 200 and frames[-1][20, 20, 0] < 60  # from 1 s on: the green half
    assert _has_audio(out.path) and out.extra["audio"]

    quiet = make_worker(spec.family).run(spec, spec.normalize({"keep_audio": False}), {"image": pic, "video": clip},
                                         _QuietSink(), out_dir)
    assert not _has_audio(quiet.path) and len(_frames(quiet.path)[1]) == 48  # the whole 2 s


def test_a_start_past_the_end_is_refused(tmp_path, clip, monkeypatch):
    from PIL import Image

    from aistudio_gpu.tasks import TASKS, make_worker

    monkeypatch.setenv("AISTUDIO_FAKE_MODELS", "1")
    spec = TASKS["wan-animate2"]
    pic = tmp_path / "her.png"
    Image.new("RGB", (40, 80)).save(pic)
    with pytest.raises(ValueError, match="no frames"):
        make_worker(spec.family).run(spec, spec.normalize({"start": 30.0}), {"image": pic, "video": clip},
                                     _QuietSink(), tmp_path)


def test_caption_is_the_models_two_part_form():
    from aistudio_gpu.tasks.wan_animate import LOOK, SCENE, caption

    assert caption("a woman in a red dress", "a sunny beach") == f"{LOOK}a woman in a red dress {SCENE}a sunny beach"
    assert caption("").startswith(LOOK + "the person") and caption("").endswith("as in the reference picture")
    own = f"{LOOK}穿着红色连衣裙 {SCENE}海滩"
    assert caption(own, "ignored") == own


def test_color_match_moves_each_frame_onto_the_pictures_colours():
    from aistudio_gpu.tasks.wan_animate import color_match

    rng = np.random.default_rng(0)
    ref = rng.uniform(0.2, 0.6, (32, 32, 3))
    ref[..., 0] += 0.3  # a warm picture
    frames = np.stack([np.clip(ref[..., ::-1] * 0.7 + 0.1, 0, 1)] * 3)  # cool, darker frames
    matched = color_match(frames, ref)
    assert np.allclose(matched.reshape(-1, 3).mean(0), ref.reshape(-1, 3).mean(0), atol=0.02)
    assert matched.shape == frames.shape and matched.min() >= 0 and matched.max() <= 1


def test_download_entry_and_group():
    from aistudio_gpu.download import MODELS, expand

    assert expand(["wan-animate2"]) == ["wan-animate2"]
    entry = MODELS["wan-animate2"]
    assert entry["repo"] == "Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers" and len(entry["revision"]) == 40
