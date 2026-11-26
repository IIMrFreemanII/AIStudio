"""Long prompts and A1111 weights for the CLIP models (SDXL's two text encoders, SD 1.5's one).

Plain diffusers keeps only a prompt's first 77 tokens; this encodes all of it, the way AUTOMATIC1111's web UI does:

* **Weights** (``parse``): ``(word:1.3)`` sets a weight; ``(word)`` multiplies by 1.1 and ``[word]`` by 1/1.1 (≈ 0.9),
  and they nest (``((word))`` = 1.21). ``\\(`` ``\\)`` ``\\[`` ``\\]`` are literal brackets, and so is any bracket
  without its partner (``a (b`` is the text "a (b", unweighted). ``BREAK`` ends the current chunk.
* **Chunks** (``chunks``): each tokenizer turns the text into tokens, each with its weight, cut into chunks of 75
  tokens. When a chunk fills up and one of its last 20 tokens is a comma, it ends after that comma instead, so a
  comma-separated tag stays whole (A1111's "comma padding backtrack"). Each chunk gets BOS and EOS and the tokenizer's
  own padding up to 77, exactly as the tokenizer pads a short prompt.
* **Encoding** (``encode_sdxl``, ``encode_sd15``): each chunk goes through the text encoder on its own (the same
  hidden layer the pipeline uses: SDXL's penultimate, SD 1.5's last), its token embeddings are multiplied by their
  weights, and the chunks are joined along the sequence (75 tokens → 77 embeddings, 150 → 154…). The UNet's cross
  attention takes any length. SDXL's pooled embedding comes from the first chunk.
* **Mean kept** (A1111's "Original" emphasis): after the multiply, a chunk is rescaled so the mean of all its
  embedding values is what it was before. A weight then shifts attention towards its tokens without making the whole
  conditioning louder; without it, a few strong weights push the picture out of range (burned colours). A chunk with
  every weight 1 is left exactly as encoded.
* **Same length**: the positive and the negative prompt get empty chunks (``""`` encoded) until they have the same
  number, as classifier-free guidance batches them together. An empty negative on SDXL is zeros, like the pipeline's
  ``force_zeros_for_empty_prompt``.

A prompt without weights that fits in 77 tokens gives exactly the embeddings the pipeline makes from the string, so
existing prompts render as before. LoRAs that patch the text encoders must be active before encoding (the workers
encode after ``_use_loras``).
"""

from __future__ import annotations

import re

ROUND = 1.1  # ( … )
SQUARE = 1 / 1.1  # [ … ]
BACKTRACK = 20  # a full chunk may end at a comma this many tokens back
BREAK = None  # the weight that marks a BREAK in ``parse``'s result

_TOKEN = re.compile(r"""
    \\[()\[\]\\]       # an escaped bracket or backslash
  | \( | \[             # an opening bracket
  | :\s*([+-]?(?:\d+\.?\d*|\.\d+))\s*\)  # ":1.3)", the end of a weighted group
  | \) | ]              # a closing bracket
  | [^\\()\[\]:]+       # plain text
  | :  | \\             # a lone colon or backslash
""", re.X)
_BREAK = re.compile(r"\s*\bBREAK\b\s*")


def parse(text: str) -> list[tuple[str, float | None]]:
    """The text in pieces, each with its weight; a BREAK is ``("BREAK", BREAK)``. Pieces next to each other with the
    same weight are merged, so a prompt without weights is one piece: the whole text."""
    res: list[list] = []
    opens: dict[str, list[list]] = {"(": [], "[": []}  # each unclosed bracket's placeholder piece in ``res``

    def close(kind: str, weight: float) -> None:
        mark = opens[kind].pop()
        i = next(k for k, piece in enumerate(res) if piece is mark)
        del res[i]
        for piece in res[i:]:
            if piece[1] is not BREAK:
                piece[1] *= weight

    for m in _TOKEN.finditer(text):
        token, weight = m.group(0), m.group(1)
        if token.startswith("\\") and len(token) == 2:
            res.append([token[1], 1.0])
        elif token in opens:
            mark = [token, 1.0]  # stays as literal text if the bracket is never closed
            res.append(mark)
            opens[token].append(mark)
        elif weight is not None and opens["("]:
            close("(", float(weight))
        elif token == ")" and opens["("]:
            close("(", ROUND)
        elif token == "]" and opens["["]:
            close("[", SQUARE)
        else:
            for i, part in enumerate(_BREAK.split(token)):
                if i:
                    res.append(["BREAK", BREAK])
                if part:
                    res.append([part, 1.0])
    merged: list[list] = []
    for text_, weight in res:
        if merged and weight is not BREAK and merged[-1][1] == weight:
            merged[-1][0] += text_
        else:
            merged.append([text_, weight])
    return [(t, w) for t, w in merged] or [("", 1.0)]


def chunks(tokenizer, text: str) -> list[tuple[list[int], list[float]]]:
    """The text as chunks of ``model_max_length`` (77) token ids with BOS, EOS and padding, and a weight per id."""
    size = tokenizer.model_max_length
    room = size - 2
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    comma = tokenizer.convert_tokens_to_ids(",</w>")
    if comma == tokenizer.unk_token_id:
        comma = None
    out: list[tuple[list[int], list[float]]] = []
    ids: list[int] = []
    weights: list[float] = []
    last_comma = -1

    def finish() -> None:
        nonlocal ids, weights, last_comma
        n = len(ids)
        out.append(([bos, *ids, eos] + [pad] * (room - n), [1.0, *weights, 1.0] + [1.0] * (room - n)))
        ids, weights, last_comma = [], [], -1

    for piece, weight in parse(text):
        if weight is BREAK:
            if ids:
                finish()
            continue
        for token in tokenizer(piece, add_special_tokens=False, verbose=False).input_ids:
            if len(ids) == room:
                if last_comma != -1 and len(ids) - last_comma <= BACKTRACK:
                    cut = last_comma + 1
                    rest_ids, rest_weights = ids[cut:], weights[cut:]
                    ids, weights = ids[:cut], weights[:cut]
                    finish()
                    ids, weights = rest_ids, rest_weights
                else:
                    finish()
            if token == comma:
                last_comma = len(ids)
            ids.append(token)
            weights.append(weight)
    if ids or not out:
        finish()
    return out


def _empty(tokenizer) -> tuple[list[int], list[float]]:
    return chunks(tokenizer, "")[0]


def _encode(encoder, tokenizer_chunks, hidden: str):
    """One prompt's chunks through ``encoder``, weighted and joined: (embeddings [1, 77·n, d], the first chunk's
    pooled embedding or None). ``hidden``: "penultimate" (SDXL) or "last" (SD 1.5)."""
    import torch

    device = next(encoder.parameters()).device
    parts, pooled = [], None
    for ids, weights in tokenizer_chunks:
        out = encoder(torch.tensor([ids], device=device), output_hidden_states=hidden == "penultimate")
        z = out.hidden_states[-2] if hidden == "penultimate" else out[0]
        if pooled is None and out[0].ndim == 2:
            pooled = out[0]  # CLIPTextModelWithProjection's text_embeds
        if any(w != 1.0 for w in weights):
            z32 = z.float()
            before = z32.mean()
            z32 = z32 * torch.tensor(weights, device=device, dtype=torch.float32)[None, :, None]
            after = z32.mean()
            if after.abs() > 1e-12:
                z32 = z32 * (before / after)
            z = z32.to(z.dtype)
        parts.append(z)
    return torch.cat(parts, dim=1), pooled


def _padded(tokenizer, texts: list[str]) -> list[list[tuple[list[int], list[float]]]]:
    return [chunks(tokenizer, t) for t in texts]


def encode_sdxl(pipe, prompt: str, negative: str | None) -> dict:
    """``StableDiffusionXLPipeline`` arguments for the prompt and the negative (empty or None: zeros, as the pipeline
    does with ``force_zeros_for_empty_prompt``)."""
    import torch

    pairs = [(t, e) for t, e in ((pipe.tokenizer, pipe.text_encoder), (pipe.tokenizer_2, pipe.text_encoder_2))
             if t is not None and e is not None]
    negative = (negative or "").strip()
    zeros = not negative and pipe.config.force_zeros_for_empty_prompt
    texts = [prompt] if zeros else [prompt, negative]
    per_tokenizer = [_padded(tok, texts) for tok, _ in pairs]
    n = max(len(c) for lists in per_tokenizer for c in lists)
    embeds, pooled = [], []
    for k in range(len(texts)):
        parts, pool = [], None
        for (tok, enc), lists in zip(pairs, per_tokenizer):
            cs = lists[k] + [_empty(tok)] * (n - len(lists[k]))
            z, p = _encode(enc, cs, "penultimate")
            parts.append(z)
            pool = p if p is not None else pool
        embeds.append(torch.cat(parts, dim=-1))
        pooled.append(pool)
    dtype = pipe.text_encoder_2.dtype if pipe.text_encoder_2 is not None else pipe.unet.dtype
    device = pipe._execution_device
    out = {"prompt_embeds": embeds[0].to(dtype=dtype, device=device),
           "pooled_prompt_embeds": pooled[0].to(dtype=dtype, device=device)}
    if zeros:
        out["negative_prompt_embeds"] = torch.zeros_like(out["prompt_embeds"])
        out["negative_pooled_prompt_embeds"] = torch.zeros_like(out["pooled_prompt_embeds"])
    else:
        out["negative_prompt_embeds"] = embeds[1].to(dtype=dtype, device=device)
        out["negative_pooled_prompt_embeds"] = pooled[1].to(dtype=dtype, device=device)
    return out


def encode_sd15(pipe, prompt: str, negative: str | None) -> dict:
    """``StableDiffusionPipeline`` arguments for the prompt and the negative (empty or None: ``""`` encoded, as the
    pipeline does)."""
    tok, enc = pipe.tokenizer, pipe.text_encoder
    lists = _padded(tok, [prompt, (negative or "").strip()])
    n = max(len(c) for c in lists)
    pos, neg = (_encode(enc, c + [_empty(tok)] * (n - len(c)), "last")[0] for c in lists)
    dtype, device = enc.dtype, pipe._execution_device
    return {"prompt_embeds": pos.to(dtype=dtype, device=device),
            "negative_prompt_embeds": neg.to(dtype=dtype, device=device)}
