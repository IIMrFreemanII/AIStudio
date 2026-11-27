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


def _tiny_image_transformer():
    import inspect

    cls = diffusers.MiniMaxH3ImageTransformer2DModel
    keys = inspect.signature(cls.__init__).parameters
    return cls(**{k: v for k, v in TINY.items() if k in keys})


def test_training_targets_the_blocks_and_its_file_loads_for_drawing():
    # The trainer's adapter covers the transformer blocks' attention and feed-forward layers (not the refiner or the
    # per-step output heads), and the file it writes loads through the drawing worker's LoRA path.
    import re

    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    from aistudio_gpu.tasks.train_h3_image import ADAPTER, TARGETS, lora_file_keys

    names = [n for n, m in _tiny_image_transformer().named_modules() if isinstance(m, torch.nn.Linear)]
    hit = [n for n in names if re.fullmatch(TARGETS, n)]
    assert "transformer_blocks.0.attn.to_q" in hit and "transformer_blocks.0.ff.net.2" in hit
    assert not any(n.startswith(("token_refiner", "pdd_proj_out")) for n in hit)

    trained = _tiny_image_transformer()
    LoRAPipe(types.SimpleNamespace(transformer=trained))  # adds the adapter methods, as the trainer's load does
    trained.requires_grad_(False)
    trained.add_adapter(LoraConfig(r=4, lora_alpha=4, init_lora_weights="gaussian", target_modules=TARGETS),
                        adapter_name=ADAPTER)
    with torch.no_grad():
        for n, p in trained.named_parameters():
            if ".lora_B." in n:
                p.normal_(0, 0.01)
    state = lora_file_keys(get_peft_model_state_dict(trained, adapter_name=ADAPTER))
    assert state and all(k.startswith("transformer.transformer_blocks.") for k in state)

    drawing = LoRAPipe(types.SimpleNamespace(transformer=_tiny_image_transformer()))
    drawing.load_lora_weights(state, adapter_name="lora_trained")
    assert drawing.get_list_adapters() == {"transformer": ["lora_trained"]}


def test_velocity_target_and_capture():
    from aistudio_gpu.tasks.train_h3_image import Captured, capture, noisy_and_target

    clean, noise = torch.ones(3, 2), torch.zeros(3, 2)
    x, v = noisy_and_target(clean, noise, 0.25)
    assert torch.allclose(x, torch.full((3, 2), 0.75)) and torch.allclose(x + 0.25 * v, clean)  # denoised = x + σ·v

    class Wrapper:  # stands in for the denoise loop block
        def get_block_state(self, state):
            return state

        def __call__(self, components, state):
            raise AssertionError("the loop must not run")

    class Step(Wrapper):
        pass

    denoise = types.SimpleNamespace(MiniMaxH3ImageDenoiseLoopWrapper=Wrapper)
    scheduler = types.SimpleNamespace(sigmas=torch.tensor([1.0, 0.5, 0.0]))

    def pipe(**kwargs):
        return Step()(types.SimpleNamespace(scheduler=scheduler), {"prompt": kwargs["prompt"]})

    c = capture(pipe, denoise, prompt="zq_mia, a woman")
    assert isinstance(c, Captured) and c.state == {"prompt": "zq_mia, a woman"} and c.sigmas == [1.0, 0.5, 0.0]
    assert Wrapper.__call__.__name__ == "__call__"  # restored


def test_training_size_matches_the_reference_path():
    from diffusers.modular_pipelines.minimax_h3_image.before_encoder import bound_short_edge

    from aistudio_gpu.tasks.train_h3_image import reference_size

    for w, h in [(832, 1248), (1216, 832), (1024, 1024), (900, 1700)]:
        size = reference_size(w, h, 1024)
        assert min(size) == 1024 and size[0] % 32 == 0 and size[1] % 32 == 0
        assert bound_short_edge(*size, 1024, 2048, 32) == size  # the edit path keeps it as it is
    assert reference_size(832, 1248, 512) == (1024, 1536)  # never below the reference path's 1024

