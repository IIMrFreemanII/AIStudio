"""MiniMax H3 Image LoRAs (``tasks/minimax_h3_image.py``): a video-shaped LoRA loads into a tiny image transformer,
its audio-branch layers dropped. Needs diffusers ≥ 0.43 (the image transformer); skipped otherwise."""

import types

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "MiniMaxH3ImageTransformer2DModel"):
    pytest.skip("diffusers without MiniMax H3 Image (needs 0.43)", allow_module_level=True)

from aistudio_gpu.tasks.minimax_h3_image import LoRAPipe  # noqa: E402

TINY = dict(num_attention_heads=2, attention_head_dim=16, hidden_size=32, num_layers=1, num_refiner_layers=1,
            ffn_dim=64, in_channels=4, text_dim=32, freq_dim=32, time_embed_hidden_dim=32, time_embed_dim=16,
            rope_freq_dim=4)


def _video_lora(rank: int = 4) -> dict:
    """A diffusers-format LoRA over every Linear of a tiny *video* transformer (audio projections included)."""
    video = diffusers.MiniMaxH3Transformer3DModel(**TINY, audio_in_channels=4)
    state = {}
    for n, mod in video.named_modules():
        if isinstance(mod, torch.nn.Linear):
            state[f"transformer.{n}.lora_A.weight"] = torch.randn(rank, mod.in_features) * 0.01
            state[f"transformer.{n}.lora_B.weight"] = torch.randn(mod.out_features, rank) * 0.01
    return state


def test_a_video_lora_loads_into_the_image_transformer_without_its_audio_layers(capsys):
    import inspect

    cls = diffusers.MiniMaxH3ImageTransformer2DModel
    keys = inspect.signature(cls.__init__).parameters
    image = cls(**{k: v for k, v in TINY.items() if k in keys})
    pipe = LoRAPipe(types.SimpleNamespace(transformer=image))
    pipe.load_lora_weights(_video_lora(), adapter_name="lora_a")
    assert "dropped" in capsys.readouterr().out
    assert pipe.get_list_adapters() == {"transformer": ["lora_a"]}
    assert any(".lora_A." in n for n, _ in image.named_parameters())
    pipe.set_adapters(["lora_a"], [0.8])
    pipe.disable_lora()
    pipe.enable_lora()
    pipe.delete_adapters("lora_a")
    assert pipe.get_list_adapters() == {}


def test_a_lora_with_no_image_layer_is_refused():
    import inspect

    cls = diffusers.MiniMaxH3ImageTransformer2DModel
    keys = inspect.signature(cls.__init__).parameters
    pipe = LoRAPipe(types.SimpleNamespace(transformer=cls(**{k: v for k, v in TINY.items() if k in keys})))
    with pytest.raises(ValueError, match="none of its layers"):
        pipe.load_lora_weights({"transformer.audio_proj_in.lora_A.weight": torch.zeros(4, 4),
                                "transformer.audio_proj_in.lora_B.weight": torch.zeros(4, 4)}, adapter_name="x")
