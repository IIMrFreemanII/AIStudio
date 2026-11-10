"""Normalized Attention Guidance (NAG; Chen et al. 2025, arXiv:2505.21179) for Wan's cross-attention.

Classifier-free guidance applies a negative prompt by running the model twice per step and extrapolating away from
the negative prediction. Few-step distilled models such as the Lightning LoRAs are trained *without* it (guidance 1),
so there a negative prompt does nothing, and turning guidance back on doubles the cost and burns the colours. NAG
instead guides inside each cross-attention layer: it attends to the positive and the negative prompt, extrapolates
the attention output away from the negative one, clips the result's L1 norm to ``tau`` times the positive one's, and
blends it back with ``alpha``. Only the cross-attention runs twice (the video tokens attend to 512 text tokens), so
it costs a few percent of a step instead of a second model pass.

How it is wired: the pipeline gets ``prompt_embeds`` = positive and negative T5 embeddings concatenated along the
sequence (2 × 512 tokens). The transformer's text projection is per token, so both halves arrive at every block's
cross-attention, where this processor splits them again. With any other length (no negative prompt, or classifier-
free guidance, which passes each prompt on its own) it is Wan's normal attention.

Defaults are the paper's for Wan (scale 11, tau 2.5, alpha 0.25).
"""

from __future__ import annotations

TEXT_LEN = 512  # Wan pads every prompt's T5 embeddings to 512 tokens
SCALE, TAU, ALPHA = 11.0, 2.5, 0.25


def make_processor(scale: float = SCALE, tau: float = TAU, alpha: float = ALPHA):
    """A processor for every block's ``attn2``. One instance can be shared; set ``.scale`` per job."""
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from diffusers.models.transformers.transformer_wan import WanAttnProcessor, _get_qkv_projections

    class NAGWanAttnProcessor(WanAttnProcessor):
        def __init__(self):
            super().__init__()
            self.scale, self.tau, self.alpha = scale, tau, alpha

        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
            if (encoder_hidden_states is None or attn.add_k_proj is not None or self.scale <= 1
                    or encoder_hidden_states.shape[1] != 2 * TEXT_LEN):
                return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)
            positive, negative = encoder_hidden_states.split(TEXT_LEN, dim=1)
            query, key, value = _get_qkv_projections(attn, hidden_states, positive)
            query = attn.norm_q(query).unflatten(2, (attn.heads, -1))
            outs = []
            for k, v in ((key, value), _kv(attn, negative)):
                k = attn.norm_k(k).unflatten(2, (attn.heads, -1))
                v = v.unflatten(2, (attn.heads, -1))
                out = dispatch_attention_fn(query, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                                            backend=self._attention_backend, parallel_config=None)
                outs.append(out.flatten(2, 3).type_as(query))
            hidden_states = guide(*outs, self.scale, self.tau, self.alpha)
            hidden_states = attn.to_out[0](hidden_states)
            return attn.to_out[1](hidden_states)

    return NAGWanAttnProcessor()


def _kv(attn, context):
    if attn.fused_projections:
        return attn.to_kv(context).chunk(2, dim=-1)
    return attn.to_k(context), attn.to_v(context)


def guide(positive, negative, scale: float, tau: float, alpha: float):
    """NAG on attention outputs [batch, tokens, channels]: extrapolate, clip the per-token L1 norm, blend."""
    import torch

    guided = positive * scale - negative * (scale - 1)
    norm_pos = positive.norm(p=1, dim=-1, keepdim=True)
    norm_guided = guided.norm(p=1, dim=-1, keepdim=True)
    ratio = torch.nan_to_num(norm_guided / norm_pos, nan=10.0, posinf=10.0)
    guided = torch.where(ratio > tau, guided * (norm_pos * tau / (norm_guided + 1e-7)), guided)
    return guided * alpha + positive * (1 - alpha)


def install(transformers) -> object:
    """Puts one shared NAG processor on every cross-attention of the given Wan transformers and returns it."""
    processor = make_processor()
    for t in transformers:
        if t is None:
            continue
        for block in t.blocks:
            block.attn2.set_processor(processor)
    return processor
