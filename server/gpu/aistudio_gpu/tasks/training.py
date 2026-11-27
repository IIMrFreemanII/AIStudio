"""What the LoRA trainers share: the dataset (one zip of pictures and captions), aspect buckets and the step count.

A dataset is a zip of up to ``MAX_PICTURES`` pictures (``.png``/``.jpg``/``.jpeg``/``.webp``), each with an optional
caption in a ``.txt`` of the same name (``01.png`` + ``01.txt``). The app writes the captions so they describe what
changes between pictures (clothes, place, pose, light) and not the person, whose look the LoRA then learns under the
trigger word that starts every caption.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

MAX_PICTURES = 60
MIN_PICTURES = 4
MAX_ZIP_BYTES = 600 * 2**20
MAX_PICTURE_BYTES = 40 * 2**20
PICTURE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


@dataclass
class Sample:
    name: str
    image: object  # PIL.Image, RGB
    caption: str


def read_dataset(path: Path) -> list[Sample]:
    """The pictures and captions in the zip at ``path``, sorted by name. Refuses a broken or oversized zip, paths that
    leave the archive, and fewer than ``MIN_PICTURES`` or more than ``MAX_PICTURES`` pictures."""
    from PIL import Image

    if path.stat().st_size > MAX_ZIP_BYTES:
        raise ValueError(f"the training pictures are over {MAX_ZIP_BYTES >> 20} MB")
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as e:
        raise ValueError("the training pictures aren't a zip file") from e
    with archive:
        infos = {}
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if name.startswith(("/", "\\")) or ".." in Path(name).parts:
                raise ValueError(f"the zip has an unsafe path: {name}")
            infos[name] = info
        pictures = sorted(n for n in infos if Path(n).suffix.lower() in PICTURE_SUFFIXES
                          and not Path(n).name.startswith("."))
        if len(pictures) < MIN_PICTURES:
            raise ValueError(f"train with at least {MIN_PICTURES} pictures (got {len(pictures)})")
        if len(pictures) > MAX_PICTURES:
            raise ValueError(f"train with at most {MAX_PICTURES} pictures (got {len(pictures)})")
        samples = []
        for name in pictures:
            if infos[name].file_size > MAX_PICTURE_BYTES:
                raise ValueError(f"{name} is over {MAX_PICTURE_BYTES >> 20} MB")
            try:
                image = Image.open(io.BytesIO(archive.read(name))).convert("RGB")
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"{name} isn't a picture ({e})") from e
            text = str(Path(name).with_suffix(".txt"))
            caption = archive.read(text).decode("utf-8", "replace").strip() if text in infos else ""
            samples.append(Sample(name, image, " ".join(caption.split())))
        return samples


def bucket(width: int, height: int, area: int = 1024 * 1024, multiple: int = 64) -> tuple[int, int]:
    """A training size near ``area`` pixels with the picture's aspect ratio, in multiples of ``multiple``."""
    aspect = width / height
    h = (area / aspect) ** 0.5
    w = h * aspect
    return max(multiple, round(w / multiple) * multiple), max(multiple, round(h / multiple) * multiple)


def fit(image, size: tuple[int, int]):
    """``image`` scaled to cover ``size`` and centre-cropped to it."""
    from PIL import Image

    w, h = size
    scale = max(w / image.width, h / image.height)
    resized = image.resize((max(w, round(image.width * scale)), max(h, round(image.height * scale))),
                           Image.Resampling.LANCZOS)
    left, top = (resized.width - w) // 2, (resized.height - h) // 2
    return resized.crop((left, top, left + w, top + h))


def total_steps(pictures: int, asked: int, per_picture: int, low: int, high: int) -> int:
    """``asked`` when set (> 0), else ``per_picture`` × pictures within [low, high]."""
    return asked if asked > 0 else min(max(per_picture * pictures, low), high)
