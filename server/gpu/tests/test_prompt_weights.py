"""Long prompts and A1111 weights for SDXL (``tasks/prompt_weights.py``), on the tiny random CLIP models
``test_models_tiny.py`` builds. The parser needs nothing; the rest is skipped without torch/diffusers."""

import pytest

from aistudio_gpu.tasks.prompt_weights import BREAK, ROUND, SQUARE, parse


def weights(text):
    return [(t, round(w, 4) if w is not None else None) for t, w in parse(text)]


@pytest.mark.parametrize("text, expected", [
    ("a photo of a cat", [("a photo of a cat", 1.0)]),
    ("", [("", 1.0)]),
    ("a (red jacket:1.4), city", [("a ", 1.0), ("red jacket", 1.4), (", city", 1.0)]),
    ("(a) [b]", [("a", 1.1), (" ", 1.0), ("b", round(1 / 1.1, 4))]),
    ("((a))", [("a", 1.21)]),
    ("[[a]]", [("a", round(1 / 1.21, 4))]),
    ("((a:1.5) b)", [("a", 1.65), (" b", 1.1)]),
    ("(a ( b:0.5) c)", [("a ", 1.1), (" b", 0.55), (" c", 1.1)]),
    ("( a : 1.2 )", [(" a ", 1.2)]),
    (r"\(a\) \[b\] c\\d", [("(a) [b] c\\d", 1.0)]),
    ("a (b", [("a (b", 1.0)]),
    ("a b) c]", [("a b) c]", 1.0)]),
    ("((a) b", [("(", 1.0), ("a", 1.1), (" b", 1.0)]),
    ("a:1.3) b", [("a:1.3) b", 1.0)]),
    ("(a:x) b", [("a:x", 1.1), (" b", 1.0)]),
    ("(a:.) b", [("a:.", 1.1), (" b", 1.0)]),
    ("[a (b] c)", [("a ", round(1 / 1.1, 4)), ("b", 1.0), (" c", 1.1)]),
    ("a BREAK b", [("a", 1.0), ("BREAK", None), ("b", 1.0)]),
    ("BREAKFAST", [("BREAKFAST", 1.0)]),
])
def test_parse(text, expected):
    assert weights(text) == expected


def test_parse_constants():
    assert ROUND == 1.1 and SQUARE == pytest.approx(0.909, abs=1e-3) and BREAK is None


# MARK: encoding (tiny CLIP models)

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from aistudio_gpu.tasks import TASKS, make_worker  # noqa: E402
from aistudio_gpu.tasks.prompt_weights import _encode, chunks, encode_sdxl  # noqa: E402
from test_models_tiny import Sink, tiny_sdxl, tiny_sdxl_pipeline  # noqa: E402

LONG = "a " * 200  # 200 one-token words, no commas


@pytest.fixture(scope="module")
def sdxl():
    return tiny_sdxl_pipeline(0)


@pytest.mark.parametrize("negative", ["", "blurry, low quality"])
def test_sdxl_short_unweighted_matches_the_pipeline(sdxl, negative):
    prompt = "a photo of a woman in a red jacket, city street at night"
    with torch.inference_mode():
        ours = encode_sdxl(sdxl, prompt, negative)
        pe, ne, pp, np_ = sdxl.encode_prompt(prompt, do_classifier_free_guidance=True,
                                             negative_prompt=negative.strip() or None)
    assert torch.equal(ours["prompt_embeds"], pe) and torch.equal(ours["negative_prompt_embeds"], ne)
    assert torch.equal(ours["pooled_prompt_embeds"], pp) and torch.equal(ours["negative_pooled_prompt_embeds"], np_)


def test_a_full_chunk_is_still_one_chunk(sdxl):
    tok = sdxl.tokenizer
    assert len(chunks(tok, "a " * 75)) == 1 and len(chunks(tok, "a " * 76)) == 2


def test_a_200_token_prompt_makes_3_chunks(sdxl):
    tok = sdxl.tokenizer
    assert len(tok(LONG, add_special_tokens=False, verbose=False).input_ids) == 200
    cs = chunks(tok, LONG)
    assert [len(ids) for ids, _ in cs] == [77, 77, 77]
    assert [sum(i != tok.pad_token_id for i in ids[1:-1]) for ids, _ in cs] == [75, 75, 50]
    assert all(ids[0] == tok.bos_token_id for ids, _ in cs)
    with torch.inference_mode():
        xl = encode_sdxl(sdxl, LONG, "")
    assert xl["prompt_embeds"].shape == (1, 231, 64) and xl["pooled_prompt_embeds"].shape == (1, 32)
    with torch.inference_mode():  # the pooled embedding is the first chunk's
        first = encode_sdxl(sdxl, "a " * 75, "")
    assert torch.equal(xl["pooled_prompt_embeds"], first["pooled_prompt_embeds"])
    assert torch.equal(xl["prompt_embeds"][:, :77], first["prompt_embeds"])


def test_a_full_chunk_ends_after_a_late_comma(sdxl):
    tok = sdxl.tokenizer
    comma = tok.convert_tokens_to_ids(",</w>")
    cs = chunks(tok, "a " * 70 + ", " + "a " * 10)
    first = cs[0][0][1:-1]
    assert first[70] == comma and all(i == tok.pad_token_id for i in first[71:])
    assert len(cs) == 2 and sum(i != tok.pad_token_id for i in cs[1][0][1:-1]) == 10


def test_break_starts_a_chunk(sdxl):
    assert len(chunks(sdxl.tokenizer, "a a BREAK a")) == 2


def test_a_weight_scales_its_tokens(sdxl):
    tok, enc = sdxl.tokenizer, sdxl.text_encoder
    n_cat = len(tok("cat", add_special_tokens=False).input_ids)
    with torch.inference_mode():
        plain = _encode(enc, chunks(tok, "cat dog"), "penultimate")[0].float()
        weighted = _encode(enc, chunks(tok, "(cat:1.5) dog"), "penultimate")[0].float()
    w = torch.ones(77)
    w[1:1 + n_cat] = 1.5
    scaled = plain * w[None, :, None]
    expected = scaled * (plain.mean() / scaled.mean())  # the mean is kept
    assert torch.allclose(weighted, expected, rtol=1e-4, atol=1e-5)
    assert weighted.mean().item() == pytest.approx(plain.mean().item(), rel=1e-4)
    assert not torch.allclose(weighted, plain)


def test_the_shorter_side_is_padded_with_empty_chunks(sdxl):
    pipe, fn = sdxl, encode_sdxl
    with torch.inference_mode():
        long_pos = fn(pipe, LONG, "blurry")
        long_neg = fn(pipe, "a cat", LONG)
        empty = fn(pipe, "", "")["prompt_embeds"]
        short = fn(pipe, "blurry", "")["prompt_embeds"]
    for out in (long_pos, long_neg):
        assert out["prompt_embeds"].shape == out["negative_prompt_embeds"].shape
        assert out["prompt_embeds"].shape[1] == 231
    assert torch.equal(long_pos["negative_prompt_embeds"][:, :77], short)
    for k in (1, 2):
        assert torch.equal(long_pos["negative_prompt_embeds"][:, 77 * k:77 * (k + 1)], empty)
        assert torch.equal(long_neg["prompt_embeds"][:, 77 * k:77 * (k + 1)], empty)


def test_sdxl_empty_negative_is_zeros_at_the_long_length(sdxl):
    with torch.inference_mode():
        out = encode_sdxl(sdxl, LONG, "  ")
    assert out["negative_prompt_embeds"].shape == (1, 231, 64) and not out["negative_prompt_embeds"].any()
    assert not out["negative_pooled_prompt_embeds"].any()


@pytest.mark.parametrize("family, build", [("sdxl", tiny_sdxl)])
def test_workers_take_long_weighted_prompts(tmp_path, family, build):
    root = tmp_path / "models"
    root.mkdir()
    build(root)
    spec = TASKS[family]
    worker = make_worker(spec.family)
    worker.load(root, lambda s: None)
    params = {**spec.normalize({}), "prompt": "(red jacket:1.4), " + LONG + "BREAK [city street]",
              "negative": "blurry", "steps": 2, "width": 64, "height": 64}
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out = worker.run(spec, params, {}, Sink(), out_dir)
    assert out.path.exists()
