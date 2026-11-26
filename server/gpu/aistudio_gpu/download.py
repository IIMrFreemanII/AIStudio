"""Downloading models into the models folder (a Modal Volume, or a folder on a GPU box), once.

Wan 2.2's official diffusers repos store float32 weights (57 GB per transformer). The two transformers come from
bfloat16 copies instead (safetensors only; configs identical to the official ones, sizes exactly 2 bytes per
parameter — checked 2026-11-10); everything else comes from the official repo, with the 11 GB text encoder converted
to bfloat16 here. A cold start then reads ≈ 63 GB. Every repo is pinned to a commit. A folder counts as installed
once its ``.complete`` marker exists (written last).

The audio models: ACE-Step 1.5 XL in ACE-Step's own checkpoint layout (≈ 38 GB), Qwen3-TTS's three 1.7B checkpoints
(≈ 14 GB), and Stable Audio 3 medium, gated, converted here to the diffusers layout with diffusers' own conversion
script (fetched at the diffusers release the server pins, checked by its hash).

The image models of the Stable Diffusion family: SD 3.5 Large (gated, ≈ 27.6 GB: bf16 transformer, three text
encoders, VAE), its Large Turbo transformer (gated, ≈ 16.3 GB; everything else is shared with Large) and SDXL 1.0
(≈ 7 GB, the fp16 files only), and Chroma1-HD with its Flash transformer and a face adapter (≈ 48 GB). Their repos also hold fp16 duplicates, ComfyUI single files, flax, onnx and openvino
copies; the allow lists leave those out.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

TRANSFORMERS = ["transformer/*", "transformer_2/*"]
MODELS: dict[str, dict] = {
    "wan22-t2v": {"repo": "Wan-AI/Wan2.2-T2V-A14B-Diffusers", "revision": "5be7df9619b54f4e2667b2755bc6a756675b5cd7",
                  "bf16": ["text_encoder"],
                  "mirror": ("ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16", "fe9c6303ddfdefcae7f06ec7fc7c244bada46cc3")},
    "wan22-i2v": {"repo": "Wan-AI/Wan2.2-I2V-A14B-Diffusers", "revision": "596658fd9ca6b7b71d5057529bbf319ecbc61d74",
                  "bf16": ["text_encoder"],
                  "mirror": ("cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers", "1f424b0caf6617cfb63822bb0298331bcf44e0e1")},
    # The Lightning 4-step LoRAs (rank 64), laid out as <t2v|i2v>/{high,low}_noise_model.safetensors.
    "wan22-lightning": {"repo": "lightx2v/Wan2.2-Lightning", "revision": "18bccf8884ec0a078eed79785eb4ef13ea16ce1e",
                        "files": {
        "t2v/high_noise_model.safetensors": "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors",
        "t2v/low_noise_model.safetensors": "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors",
        "i2v/high_noise_model.safetensors": "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
        "i2v/low_noise_model.safetensors": "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
    }},
    # Wan 2.2 S2V-14B (Apache 2.0), the official repo's layout (it is not in diffusers): the 14B transformer in bf16
    # (4 shards), the UMT5-XXL encoder, the Wan 2.1 VAE, the tokenizer and the wav2vec2 audio encoder (safetensors
    # only; the repo also holds its flax and pytorch copies, a language model and eval logs). ≈ 45.7 GB (42.6 GiB),
    # summed from the repo's file listing at this revision, 2026-11-15.
    "wan22-s2v": {"repo": "Wan-AI/Wan2.2-S2V-14B", "revision": "dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b", "allow": [
        "config.json", "configuration.json", "diffusion_pytorch_model-*.safetensors",
        "diffusion_pytorch_model.safetensors.index.json", "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth",
        "google/umt5-xxl/*",
        "wav2vec2-large-xlsr-53-english/config.json", "wav2vec2-large-xlsr-53-english/preprocessor_config.json",
        "wav2vec2-large-xlsr-53-english/special_tokens_map.json", "wav2vec2-large-xlsr-53-english/vocab.json",
        "wav2vec2-large-xlsr-53-english/alphabet.json", "wav2vec2-large-xlsr-53-english/model.safetensors"]},
    # LongCat-Video-Avatar 1.6 (MIT): its bf16 avatar DiT, the two distillation LoRAs, the scheduler and
    # Whisper-large-v3 (safetensors only; not the int8 DiT, the vocal separator or Whisper's flax/pt copies), plus
    # LongCat-Video's UMT5 text encoder, tokenizer and VAE (not its own 54 GB DiT). ≈ 63.6 GB, summed from the repos'
    # file listings at these revisions, 2026-11-17.
    "longcat-avatar": {"repo": "meituan-longcat/LongCat-Video-Avatar-1.6", "parts": [
        ("meituan-longcat/LongCat-Video-Avatar-1.6", "acfc628244201fc53fb6fe8a8903c5bac9bdf68f", "avatar", [
            "config.json", "model_index.json", "scheduler/*", "base_model/*", "loraA/*", "loraB/*",
            "whisper-large-v3/model.safetensors", "whisper-large-v3/config.json",
            "whisper-large-v3/generation_config.json", "whisper-large-v3/preprocessor_config.json",
            "whisper-large-v3/tokenizer.json", "whisper-large-v3/tokenizer_config.json", "whisper-large-v3/vocab.json",
            "whisper-large-v3/merges.txt", "whisper-large-v3/added_tokens.json", "whisper-large-v3/normalizer.json",
            "whisper-large-v3/special_tokens_map.json"]),
        ("meituan-longcat/LongCat-Video", "03b55529b1d1d4045f5fbe14d65c8c6e8116b278", "base", [
            "text_encoder/*", "tokenizer/*", "vae/*"]),
    ]},
    # InfiniteTalk (Apache 2.0) on Wan 2.1 I2V 14B 480p (its official layout: the fp32 transformer, which the repo
    # casts to bf16 as it loads, the UMT5 and CLIP encoders, the VAE and tokenizers), InfiniteTalk's single-speaker
    # weights (not its 143 GB of quantised copies), the Chinese wav2vec2 base (its safetensors file is on PR 1) and
    # Kijai's lightx2v 4-step LoRA. ≈ 94 GB, summed from the listings, 2026-11-17.
    "infinitetalk": {"repo": "MeiGen-AI/InfiniteTalk", "parts": [
        ("Wan-AI/Wan2.1-I2V-14B-480P", "6b73f84e66371cdfe870c72acd6826e1d61cf279", "wan21-i2v", [
            "config.json", "diffusion_pytorch_model-*.safetensors", "diffusion_pytorch_model.safetensors.index.json",
            "models_t5_umt5-xxl-enc-bf16.pth", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            "Wan2.1_VAE.pth", "google/umt5-xxl/*", "xlm-roberta-large/*"]),
        ("MeiGen-AI/InfiniteTalk", "d59847ebdacf19245bfca3fb23311c0cada8378a", "infinitetalk", ["single/*"]),
        ("TencentGameMate/chinese-wav2vec2-base", "5f7a6cfdfc5440ec78748f79a9dd66f077c2b463", "wav2vec2", [
            "config.json", "preprocessor_config.json", "model.safetensors"]),
        ("Kijai/WanVideo_comfy", "8260d429d19fd7a72304cad059160b95d843913f", "lightx2v", {
            "lightx2v.safetensors": "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors"}),
    ]},
    # Gated: needs a Hugging Face token whose account accepted the licence on the model page.
    "flux1-dev": {"repo": "black-forest-labs/FLUX.1-dev", "revision": "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
                  "gated": True, "allow": [
        "model_index.json", "scheduler/*", "text_encoder/*", "text_encoder_2/*", "tokenizer/*", "tokenizer_2/*",
        "transformer/*", "vae/*"]},
    # Qwen-Image (the original 20B, Apache 2.0): bf16 throughout (transformer ≈ 41 GB, Qwen2.5-VL text encoder ≈ 17 GB).
    "qwen-image": {"repo": "Qwen/Qwen-Image", "revision": "75e0b4be04f60ec59a75f475837eced720f823b6", "allow": [
        "model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*", "transformer/*", "vae/*", "LICENSE"]},
    # Its Lightning 8-step LoRA (V2.0, bf16): `fast` mode, ≈ 6× fewer GPU seconds than 50 steps with CFG.
    "qwen-image-lightning": {"repo": "lightx2v/Qwen-Image-Lightning",
                             "revision": "d468c534b360e0754c6bf9e38ed49e1e09cd2e2a", "files": {
        "8steps.safetensors": "Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors"}},
    # Qwen-Image-Edit 2511 (Apache 2.0): bf16 throughout (transformer ≈ 41 GB, Qwen2.5-VL text encoder ≈ 17 GB),
    # plus the processor that reads the input pictures.
    "qwen-image-edit": {"repo": "Qwen/Qwen-Image-Edit-2511", "revision": "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9",
                        "allow": ["model_index.json", "processor/*", "scheduler/*", "text_encoder/*", "tokenizer/*",
                                  "transformer/*", "vae/*"]},
    # Its Lightning 8-step LoRA (V1.0, bf16): `fast` mode.
    "qwen-image-edit-lightning": {"repo": "lightx2v/Qwen-Image-Edit-2511-Lightning",
                                  "revision": "107d238d68b8387e32843d0b4a2e24ceb0d3ad86", "files": {
        "8steps.safetensors": "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"}},
    # ACE-Step looks for its main repo's four components in checkpoints/ before it loads the XL DiT and the 4B planner.
    "ace-step-xl": {"repo": "ACE-Step/Ace-Step1.5", "parts": [
        ("ACE-Step/Ace-Step1.5", "19671f406d603126926c1b7e2adc169acbcade22", "checkpoints"),
        ("ACE-Step/acestep-v15-xl-turbo", "d4a0b288b83ebb7e25a8c0b32c573c22e134e8ee", "checkpoints/acestep-v15-xl-turbo"),
        ("ACE-Step/acestep-5Hz-lm-4B", "0a3ec94b557aea7d508da38b31cfe7341f6ff737", "checkpoints/acestep-5Hz-lm-4B"),
    ]},
    "qwen3-tts": {"repo": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "parts": [
        ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "fd4b254389122332181a7c3db7f27e918eec64e3", "base"),
        ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "0c0e3051f131929182e2c023b9537f8b1c68adfe", "custom"),
        ("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "5ecdb67327fd37bb2e042aab12ff7391903235d3", "design"),
    ]},
    # MiniMax H3 Community Licence (not for the US, EU, UK or Korea without MiniMax's licence). Only the root diffusers
    # layout (≈ 210 GB: both transformer partitions and the shared parts), not the original checkpoint's FL2VA/ and
    # Ref2VA/ copies (288 GB); plus LightX2V's Turbo LoRA for each partition.
    "minimax-h3": {"repo": "MiniMaxAI/MiniMax-H3", "parts": [
        ("MiniMaxAI/MiniMax-H3", "0919fb684102ef115126612f3fbb84d1027e7317", ".", [
            "model_index.json", "modular_model_index.json", "LICENSE", "transformer/*", "transformer_ref/*",
            "text_encoder/*", "tokenizer/*", "processor/*", "vae/*", "audio_vae/*", "scheduler/*",
            "audio_scheduler/*"]),
        ("lightx2v/Minimax-h3-Turbo", "c2bf13b837e33bd851b370285fc7db99482d2ab2", "turbo", {
            "fl2v.safetensors": "minimax_h3_fl2v_turbo_8step_v1.2_768p_bf16.safetensors",
            "ref2v.safetensors": "minimax_h3_ref2v_turbo_8step_v1.1_768p_bf16.safetensors"}),
    ]},
    # Research and non-commercial licence; the audio tokenizer (Qwen3-TTS's) ships inside the repo.
    "breeze-tts": {"repo": "BreezeBlue/Breeze-TTS-2", "revision": "3e28c5151381a722f1d8661b4118c298caa77aa4",
                   "allow": ["*.json", "*.safetensors", "audio_tokenizer/*", "LICENSE"]},
    # Gated (Stability AI Community licence); its T5Gemma text encoder ships inside the repo.
    "stable-audio-3": {"repo": "stabilityai/stable-audio-3-medium", "revision": "d4d3579395a8ac0df4ab13b5b1d7138ce7290f81",
                       "gated": True, "sa3": True},
    # SD 3.5 Large: gated, Stability AI Community licence; accept it on the model page.
    "sd35-large": {"repo": "stabilityai/stable-diffusion-3.5-large", "revision": "ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f",
                   "gated": True, "allow": [
        "model_index.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*", "tokenizer_3/*",
        "text_encoder/config.json", "text_encoder/model.safetensors",
        "text_encoder_2/config.json", "text_encoder_2/model.safetensors",
        "text_encoder_3/config.json", "text_encoder_3/model.safetensors.index.json",
        "text_encoder_3/model-0000?-of-00002.safetensors", "transformer/*", "vae/*"]},
    # Large Turbo (4 steps): only its transformer; text encoders, VAE and scheduler are Large's.
    "sd35-large-turbo": {"repo": "stabilityai/stable-diffusion-3.5-large-turbo",
                         "revision": "ec07796fc06b096cc56de9762974a28f4c632eda", "gated": True,
                         "allow": ["transformer/*"]},
    # SDXL 1.0 base (not gated): the fp16 files, run in bf16.
    "sdxl": {"repo": "stabilityai/stable-diffusion-xl-base-1.0", "revision": "462165984030d82259a11f4367a4eed129e94a7b",
             "allow": [
        "model_index.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*",
        "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
        "text_encoder_2/config.json", "text_encoder_2/model.fp16.safetensors",
        "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors"]},
    # Stable Diffusion 1.5 (not gated): the fp16 files only (no safety checker), run in bf16.
    "sd15": {"repo": "stable-diffusion-v1-5/stable-diffusion-v1-5", "revision": "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
             "allow": [
        "model_index.json", "scheduler/*", "tokenizer/*", "feature_extractor/*",
        "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
        "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors"]},
    # Anima (circlestone-labs, non-commercial licence, not gated): a 2B anime model, a Cosmos DiT with a Qwen3-0.6B text
    # encoder and the Qwen-Image VAE, as a Modular Diffusers repo. Every file is taken.
    "anima": {"repo": "circlestone-labs/Anima-Base-v1.0-Diffusers", "revision": "073c3a9db359c31ad0e8aa268d15775473c2176c",
              "allow": ["modular_model_index.json", "scheduler/*", "t5_tokenizer/*", "text_conditioner/*",
                        "text_encoder/*", "tokenizer/*", "transformer/*", "vae/*"]},
    # LTX-2.3 in diffusers' layout (LTX-2 open licence): ≈ 100 GB — the Gemma 3 text encoder ≈ 49 GB, the bf16
    # transformer ≈ 38 GB. Not `connectors/diffusion_pytorch_model.safetensors`: a 6.3 GB duplicate of the shards.
    "ltx23": {"repo": "diffusers/LTX-2.3-Diffusers", "revision": "8eee8edcf067e838b843f926ec4d4cc9b2be1aaf", "allow": [
        "model_index.json", "scheduler/*", "tokenizer/*", "processor/*", "text_encoder/*", "transformer/*", "vae/*",
        "audio_vae/*", "vocoder/*", "connectors/config.json", "connectors/diffusion_pytorch_model-*.safetensors",
        "connectors/diffusion_pytorch_model.safetensors.index.json"]},
    # Its distilled transformer (≈ 38 GB; `fast` mode: 8 steps, no guidance), swapped into the loaded pipeline in place,
    # like Chroma1-Flash.
    "ltx23-distilled": {"repo": "diffusers/LTX-2.3-Distilled-Diffusers",
                        "revision": "432e0d3c2d1769aaa4d295f9243f7062bf6b47ee", "allow": ["transformer/*"]},
    # IP-Adapter "plus face" for SDXL (Apache-2.0): the ViT-H image encoder (≈ 2.5 GB) and the face adapter (≈ 0.85 GB).
    "sdxl-face": {"repo": "h94/IP-Adapter", "revision": "018e402774aeeddd60609b4ecdb7e298259dc729",
                  "allow": ["models/image_encoder/config.json", "models/image_encoder/model.safetensors",
                            "sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors"]},
    # Chroma1-HD (Apache-2.0, an 8.9B Flux-based model, lodestones): the diffusers layout only (bf16 transformer
    # ≈ 17.8 GB, the T5-XXL text encoder ≈ 9.5 GB, the VAE), not the repo's single-file copy. ≈ 27.5 GB.
    "chroma": {"repo": "lodestones/Chroma1-HD", "revision": "0e0c60ece1e82b17cb7f77342d765ba5024c40c0", "allow": [
        "model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*", "transformer/*", "vae/*"]},
    # Chroma1-Flash (Apache-2.0): Chroma1-HD with CFG baked in (few steps, no guidance pass); only its transformer
    # (≈ 17.8 GB), swapped into the loaded pipeline in place, like SD 3.5 Large Turbo.
    "chroma-flash": {"repo": "lodestones/Chroma1-Flash", "revision": "093c4f63b60507234aa53a073f1f9f565b3f4336",
                     "allow": ["transformer/*"]},
    # A face picture for Chroma: XLabs' FLUX.1 IP-Adapter v2 (Apache-2.0, 16 image tokens, ≈ 1.06 GB) with the CLIP
    # ViT-L/14 image encoder it was trained on (≈ 1.7 GB). Chroma's transformer keeps Flux's 19 double blocks.
    "chroma-face": {"repo": "XLabs-AI/flux-ip-adapter-v2", "parts": [
        ("XLabs-AI/flux-ip-adapter-v2", "d1507fa7314df9f7dbe30a1e9659e5e634a58464", "adapter", {
            "ip_adapter.safetensors": "ip_adapter.safetensors"}),
        ("openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41", "image_encoder", [
            "config.json", "model.safetensors", "preprocessor_config.json"]),
    ]},
}
# What a task name expands to.
GROUPS = {"wan22": ["wan22-t2v", "wan22-i2v", "wan22-lightning"], "wan22-s2v": ["wan22-s2v"],
          "longcat-avatar": ["longcat-avatar"], "infinitetalk": ["infinitetalk"],
          "lipsync": ["wan22-s2v", "longcat-avatar", "infinitetalk"], "flux": ["flux1-dev"],
          "qwen": ["qwen-image", "qwen-image-lightning"],
          "qwen-edit": ["qwen-image-edit", "qwen-image-edit-lightning"], "music": ["ace-step-xl"],
          "sound": ["stable-audio-3"], "speech": ["qwen3-tts"], "audio": ["ace-step-xl", "stable-audio-3", "qwen3-tts"],
          "breeze-tts": ["breeze-tts"], "minimax-h3": ["minimax-h3"],  # not in "audio": it is non-commercial, so only on request
          "sd35": ["sd35-large", "sd35-large-turbo"], "sdxl": ["sdxl", "sdxl-face"], "sd15": ["sd15"], "anima": ["anima"],
          "sdxl-face": ["sdxl-face"], "sd": ["sd35-large", "sd35-large-turbo", "sdxl", "sdxl-face", "sd15"],
          "chroma": ["chroma", "chroma-flash", "chroma-face"], "chroma-face": ["chroma-face"],
          "ltx23": ["ltx23", "ltx23-distilled"]}
SA3_CONVERTER = ("https://raw.githubusercontent.com/huggingface/diffusers/v0.42.1/scripts/"
                 "convert_stable_audio_3_to_diffusers.py",
                 "ee782fa2dfacc637251d00add7cee93f6c3aafcbed6abf111ff209947d98267f")


def expand(names: list[str]) -> list[str]:
    out: list[str] = []
    for n in names:
        for m in GROUPS.get(n, [n]):
            if m not in MODELS:
                raise SystemExit(f"unknown model {m!r}; choose from {', '.join(list(GROUPS) + list(MODELS))}")
            if m not in out:
                out.append(m)
    return out


def download(names: list[str], root: Path, token: str | None = None, on_done=None) -> None:
    token = token or os.environ.get("HF_TOKEN")
    root.mkdir(parents=True, exist_ok=True)
    for name in expand(names):
        dest = root / name
        if (dest / ".complete").exists():
            print(f"✓ {name} already installed ({size_gb(dest):.1f} GB)")
            continue
        spec = MODELS[name]
        if spec.get("gated") and not token:
            raise SystemExit(f"{name} is gated: set HF_TOKEN (a token whose account accepted the licence at "
                             f"https://huggingface.co/{spec['repo']})")
        start = time.monotonic()
        print(f"↓ {name} from {spec['repo']}", flush=True)
        dest.mkdir(parents=True, exist_ok=True)
        if "files" in spec:
            _files(spec, dest, token)
        elif "parts" in spec:
            for repo, revision, into, *rest in spec["parts"]:
                print(f"  {repo} → {name}/{into}", flush=True)
                pick = rest[0] if rest else None  # an allow list, or {target: source} single files
                if isinstance(pick, dict):
                    _files({"repo": repo, "revision": revision, "files": pick}, dest / into, token)
                else:
                    _snapshot(repo, dest / into, token, revision=revision, allow=pick, ignore=[".gitattributes"])
        elif spec.get("sa3"):
            _convert_sa3(spec, dest, token)
        elif "bf16" in spec:
            _convert(spec, dest, token)
        else:
            _snapshot(spec["repo"], dest, token, allow=spec.get("allow"), revision=spec.get("revision"))
        (dest / ".complete").write_text(json.dumps({"repo": spec["repo"], "at": time.time()}))
        print(f"✓ {name}: {size_gb(dest):.1f} GB in {time.monotonic() - start:.0f} s", flush=True)
        if on_done:
            on_done(name)  # e.g. commit the Modal Volume so other containers see it


def _snapshot(repo: str, dest: Path, token: str | None, allow=None, ignore=None, revision=None):
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, local_dir=dest, token=token, allow_patterns=allow, ignore_patterns=ignore,
                             revision=revision)


def _files(spec: dict, dest: Path, token: str | None) -> None:
    from huggingface_hub import hf_hub_download

    for target, source in spec["files"].items():
        out = dest / target
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=dest) as tmp:
            got = hf_hub_download(spec["repo"], source, local_dir=tmp, token=token, revision=spec.get("revision"))
            shutil.move(got, out)


def _convert(spec: dict, dest: Path, token: str | None) -> None:
    """The transformers from the bfloat16 mirror; everything else from the official repo, the ``bf16`` components
    loaded in float32 and saved as bfloat16 (from a temporary cache, deleted right after)."""
    import torch

    repo, big, revision = spec["repo"], spec["bf16"], spec.get("revision")
    skip = [f"{c}/*" for c in big] + [".gitattributes", "*.md", "assets/*", "examples/*"]
    if "mirror" in spec:
        mirror, mirror_rev = spec["mirror"]
        print(f"  transformers (bfloat16) from {mirror}", flush=True)
        _snapshot(mirror, dest, token, allow=TRANSFORMERS, revision=mirror_rev)
        skip += TRANSFORMERS
    _snapshot(repo, dest, token, ignore=skip, revision=revision)
    index = json.loads((dest / "model_index.json").read_text())
    for component in big:
        if (dest / component / "config.json").exists():
            continue
        library, class_name = index[component]
        cls = getattr(importlib.import_module(library), class_name)
        with tempfile.TemporaryDirectory(dir=os.environ.get("AISTUDIO_DOWNLOAD_TMP")) as cache:
            print(f"  converting {component} ({class_name}) to bfloat16", flush=True)
            model = cls.from_pretrained(repo, subfolder=component, dtype=torch.bfloat16, cache_dir=cache,
                                        token=token, revision=revision, low_cpu_mem_usage=True)
            model.save_pretrained(dest / component, max_shard_size="5GB", safe_serialization=True)
            del model
            gc.collect()


def _convert_sa3(spec: dict, dest: Path, token: str | None) -> None:
    """Stability's checkpoint (stable-audio-tools layout) → the diffusers layout ``StableAudio3Pipeline`` loads, with
    diffusers' own conversion script; the original is deleted afterwards."""
    source = dest / "source"
    _snapshot(spec["repo"], source, token, revision=spec["revision"],
              allow=["model.safetensors", "model_config.json", "t5gemma-b-b-ul2/*", "LICENSE*", "NOTICE"])
    url, sha = SA3_CONVERTER
    script = source / "convert_stable_audio_3_to_diffusers.py"
    data = urllib.request.urlopen(url, timeout=60).read()
    if hashlib.sha256(data).hexdigest() != sha:
        raise SystemExit(f"the conversion script at {url} changed (sha256 mismatch); not running it")
    script.write_bytes(data)
    print("  converting Stable Audio 3 to the diffusers layout", flush=True)
    subprocess.run([sys.executable, str(script), "--checkpoint_path", str(source / "model.safetensors"),
                    "--model_config_path", str(source / "model_config.json"), "--output_dir", str(dest),
                    "--text_encoder_repo", str(source / "t5gemma-b-b-ul2"), "--dtype", "float32",
                    "--skip_sanity_check"], check=True)
    for f in ("LICENSE.md", "LICENSE_GEMMA.md", "NOTICE"):
        if (source / f).exists():
            shutil.copy(source / f, dest / f)
    shutil.rmtree(source)


def size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9
