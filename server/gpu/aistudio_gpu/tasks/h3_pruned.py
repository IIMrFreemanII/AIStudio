# Copyright 2025 The MiniMax Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Vendored verbatim (apart from this note, PRUNED_CONFIG at the end and the dropped register_for_auto_class call) from
#   https://huggingface.co/multimodalart/MiniMax-H3-Pruned
#   transformer/modeling_minimax_h3_pruned.py, commit 1a0ef5e65b639e84af81d883817968532180e9c7
# Apache License 2.0, as above. AIStudio loads the pruned MiniMax H3 layout with it (variants.py, minimax_h3.py).
"""AdaLN-pruned MiniMax-H3 transformer.

Everything outside the timestep path is inherited from `MiniMaxH3Transformer3DModel`: the attention, the blocks, the
token refiner, the output heads and `forward` itself are the released implementation, unmodified. Only what feeds the
AdaLN projections changes.

Two other things this file adds. `enable_convrot` / `quantize_8bit`: an opt-in Hadamard conditioning of the
attention and feed-forward linears that makes 8-bit *compute* (int8 or fp8 dynamic activations, via torchao) land
within a rounding step of bfloat16. And `load_lora_adapter`, overridden to project a LoRA trained against the
*released* 2688-wide AdaLN projections onto these 8-wide ones. Both are inert unless used, so the plain pruned path
is byte for byte what it was.
"""

import math
import re
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_utils import get_parameter_dtype
from diffusers.models.transformers.transformer_minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3RotaryPosEmbed,
    MiniMaxH3TokenRefiner,
    MiniMaxH3Transformer3DModel,
    MiniMaxH3TransformerBlock,
)
from diffusers.utils import logging


logger = logging.get_logger(__name__)

_HADAMARD_CACHE: dict = {}

# The 51 AdaLN projections, as a LoRA state dict names them, with or without a `transformer.` / `transformer_ref.`
# component prefix. Group 1 is the module path the model itself knows the projection by.
ADALN_LORA_A_KEY = re.compile(r"(?:^|\.)((?:transformer_blocks\.\d+\.adaln_proj|norm_out)\.linear)\.lora_A\.weight$")

# The linears ConvRot conditions: the block stack's attention and feed-forward projections, and nothing else.
# This is the set ComfyUI's `*_int8_convrot` checkpoints quantize (they carry one fused `qkv_proj`; the three
# split projections here share an input, so rotating each is the same transform). The AdaLN path, the patch
# projections, the output heads, the token refiner and every norm stay high precision.
CONVROT_SUFFIXES = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")


def _hadamard(size: int, device, dtype) -> torch.Tensor:
    r"""Normalized *regular* Hadamard matrix of `size` - symmetric, orthogonal, therefore self-inverse.

    Kronecker powers of the regular order-4 seed, which is why `size` must be a power of 4. Identical, entry for
    entry, to the matrix `comfy_kitchen` builds for its `convrot` kernels.
    """
    key = (size, str(device), dtype)
    if key not in _HADAMARD_CACHE:
        if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
            raise ValueError(f"ConvRot group size must be a power of 4, got {size}")
        seed = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype, device=device)
        matrix, width = seed, 4
        while width < size:
            matrix, width = torch.kron(matrix, seed), width * 4
        _HADAMARD_CACHE[key] = matrix / size**0.5
    return _HADAMARD_CACHE[key]


def _rotate(tensor: torch.Tensor, group_size: int) -> torch.Tensor:
    r"""`tensor @ blockdiag(H, ..., H)` over the last dimension, in the tensor's own dtype."""
    shape = tensor.shape
    if shape[-1] % group_size != 0:
        raise ValueError(f"{shape[-1]} features is not a multiple of the ConvRot group size {group_size}")
    matrix = _hadamard(group_size, tensor.device, tensor.dtype)
    return torch.matmul(tensor.reshape(-1, shape[-1] // group_size, group_size), matrix).reshape(shape)


class MiniMaxH3ConvRotLinear(nn.Linear):
    r"""`nn.Linear` that Hadamard-rotates its input; its weight already carries the same rotation.

    `enable_convrot` bakes `W <- W @ H` into the weight, so computing `(x @ H) @ (W @ H)^T` returns `x @ W^T`
    exactly - `H` is symmetric *and* orthogonal, so it is its own inverse. Nothing about the model's function
    changes. What changes is the distribution a quantizer downstream of this module sees: every coordinate of
    `x @ H` is a +-1 combination of `group_size` input channels, so a single outlier channel no longer sets the
    scale for its whole row. That is all ConvRot is, and because both sides of one matmul are rotated back to
    back, nothing has to commute with the AdaLN modulation.

    Deliberately a bare `nn.Linear` subclass with no parameters of its own: `torchao.quantize_` still converts
    it, the `state_dict` keys are unchanged, and PEFT wraps it as a `base_layer` - which leaves a LoRA's branch
    reading the *unrotated* input, the basis LoRAs are trained in.
    """

    convrot_groupsize: int = 0

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return F.linear(_rotate(input, self.convrot_groupsize), self.weight, self.bias)


class MiniMaxH3PrunedTimeEmbedder(nn.Module):
    r"""The released timestep MLP, replaced by an interpolated table of AdaLN coordinates.

    Every AdaLN projection in the released model consumes `silu(time_embedder(time_proj(t)))`, which depends on the
    scalar timestep alone: over `t` in `[0, 1]` it traces a one-dimensional curve in `R^{time_embed_dim}`. A rank-8
    affine subspace reproduces that curve to about 1.5e-5 relative RMS, so only the curve's coordinates in that
    subspace are stored - sampled on a uniform grid of `table_size` timesteps and linearly interpolated in between.
    The subspace offset is folded into the AdaLN biases and its basis into the AdaLN weights, which is why the
    projections take an `adaln_rank`-wide input here instead of `time_embed_dim`.

    The module stands in for `time_proj` and `time_embedder` together: it consumes the raw timestep, so the released
    `forward` needs no change once `time_proj` is an identity.
    """

    def __init__(self, table_size: int = 1025, adaln_rank: int = 8) -> None:
        super().__init__()
        self.register_buffer("table", torch.zeros(table_size, adaln_rank), persistent=True)

    @property
    def linear_1(self):
        r"""Answer the dtype question a diffusers before #14398 asks of the released timestep MLP.

        `MiniMaxH3Transformer3DModel.forward` aligns the timestep it passes in with the timestep path's own dtype.
        Since [#14398](https://github.com/huggingface/diffusers/pull/14398) it asks
        `get_parameter_dtype(self.time_embedder)`, which on this module returns the table's float32 - there is no
        parameter here, only the buffer. Before it, it read `self.time_embedder.linear_1.weight.dtype`, the first
        `Linear` of the released timestep MLP this table replaces. The two questions have one answer on a pruned
        checkpoint, so the older one is answered rather than raised: the table *is* the timestep path here.

        A property returning a plain namespace, not a registered module: nothing about it reaches `state_dict`,
        `named_modules`, `.to()`, a quantizer's scan or PEFT's target resolution, and it is read for a dtype and
        never called. Delete it once every consumer runs a diffusers that carries #14398.
        """
        return SimpleNamespace(weight=self.table)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        table = self.table
        steps = table.shape[0] - 1
        position = timestep.to(table.dtype).flatten().clamp(0.0, 1.0) * steps
        lower = position.floor().clamp(max=steps - 1).long()
        weight = (position - lower).unsqueeze(-1)
        return torch.lerp(table.index_select(0, lower), table.index_select(0, lower + 1), weight)


class MiniMaxH3PrunedTimeProj(nn.Module):
    r"""Identity stand-in for `Timesteps`: the pruned time embedder indexes the raw timestep."""

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        return timestep


class MiniMaxH3PrunedAdaLN(nn.Module):
    r"""Shared by both pruned AdaLN modules: the folded float32 bias, plus the LoRA offsets that ride alongside it.

    A LoRA trained against the released 2688-wide projection contributes `lora_B @ (lora_A @ x)` to the modulation.
    With `x = mean + c @ basis`, that splits into a coordinate term the projected factors reproduce and a *constant*
    term, `lora_B @ (lora_A @ mean)`, which no `Linear(8 -> out)` can express. That constant is held here, as a
    per-adapter float32 buffer added to `folded_bias` - the same place, and the same precision, the fold's own
    constant term lives in. It is deliberately not the projection's `bias`: rounding it into bfloat16 would spend a
    full rounding step of the modulation on a term that is most of what the adapter does to the AdaLN path.
    """

    def __init__(self, out_features: int) -> None:
        super().__init__()
        self.register_buffer("folded_bias", torch.zeros(out_features), persistent=True)
        # `{adapter name: buffer attribute}`. Plain state, not a submodule: the buffers themselves are what move
        # with the module, and being non-persistent they stay out of the checkpoint, as an adapter should.
        self._lora_adaln_offsets: dict[str, str] = {}

    def register_lora_adaln_offset(self, adapter_name: str, offset: torch.Tensor) -> None:
        r"""Attach one adapter's constant term, on the device and in the precision `folded_bias` is kept in."""
        attribute = self._lora_adaln_offsets.get(adapter_name)
        if attribute is None:
            attribute = f"lora_adaln_offset_{len(self._lora_adaln_offsets)}"
            self._lora_adaln_offsets[adapter_name] = attribute
        value = offset.to(device=self.folded_bias.device, dtype=torch.float32)
        self.register_buffer(attribute, value, persistent=False)

    def lora_adaln_bias(self) -> torch.Tensor:
        r"""`folded_bias` plus every active adapter's constant term at its current scaling.

        PEFT owns everything this reads - `active_adapters`, `scaling`, `disable_adapters` - so the offsets follow
        `set_adapters`, `disable_lora` and `delete_adapters` with no bookkeeping of their own. They apply whether or
        not an adapter is merged: `fuse_lora` folds `lora_B @ lora_A` into the projection's weight, and there is
        nowhere in a bias-free `Linear` for this term to be folded to.
        """
        bias = self.folded_bias
        offsets = self._lora_adaln_offsets
        if not offsets:
            return bias
        layer = self.linear
        scaling = getattr(layer, "scaling", None)
        if not isinstance(scaling, dict) or getattr(layer, "disable_adapters", False):
            return bias
        for adapter_name in getattr(layer, "active_adapters", ()):
            attribute = offsets.get(adapter_name)
            if attribute is None or adapter_name not in scaling:
                continue
            bias = bias + getattr(self, attribute) * float(scaling[adapter_name])
        return bias


class MiniMaxH3PrunedAdaLayerNormModulation(MiniMaxH3PrunedAdaLN):
    r"""`MiniMaxH3AdaLayerNormModulation` over the pruned timestep coordinates.

    Two differences from the released module. It applies no `silu` - the table already holds the coordinates of the
    activated curve. And the folded bias is a float32 buffer applied outside the projection rather than the
    projection's own bias: it carries almost the entire modulation (the coordinate term contributes a few tenths of
    it), so storing it in bfloat16 would put a full output-scale rounding step into every evaluation. Kept in
    float32 it costs 0.4 MB per block and leaves the pruned AdaLN function closer to an exact float64 evaluation
    than the released bfloat16 checkpoint's own arithmetic is.

    `linear` stays a bias-free `nn.Linear` so PEFT wraps it exactly as it wraps the released projection.
    """

    def __init__(self, adaln_rank: int, hidden_size: int) -> None:
        out_features = 6 * hidden_size * MINIMAX_H3_MODALITY_NUM
        super().__init__(out_features)
        self.hidden_size = hidden_size
        self.linear = nn.Linear(adaln_rank, out_features, bias=False)

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        dtype = get_parameter_dtype(self.linear)
        temb = self.linear(temb.to(dtype))
        temb = (temb.float() + self.lora_adaln_bias()).to(dtype)
        temb = temb.view(-1, 6 * self.hidden_size)
        return temb.chunk(6, dim=-1)


class MiniMaxH3PrunedAdaLayerNormOut(MiniMaxH3PrunedAdaLN):
    r"""`MiniMaxH3AdaLayerNormOut` over the pruned timestep coordinates; see the modulation module above."""

    def __init__(self, hidden_size: int, adaln_rank: int, eps: float) -> None:
        super().__init__(2 * hidden_size)
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.linear = nn.Linear(adaln_rank, 2 * hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor, timestep_indices: torch.Tensor) -> torch.Tensor:
        dtype = get_parameter_dtype(self.linear)
        temb = self.linear(temb.to(dtype))
        shift, scale = (temb.float() + self.lora_adaln_bias()).to(dtype).chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
            0, timestep_indices
        )


class MiniMaxH3PrunedTransformer3DModel(MiniMaxH3Transformer3DModel):
    r"""MiniMax-H3's DiT with the AdaLN input projections reduced to their reachable rank.

    The released checkpoint spends 13.03B of its 33.14B parameters on the 50 per-block `adaln_proj.linear` matrices
    plus `norm_out.linear`, all of which read the same 2688-wide timestep embedding. Because that embedding is a
    function of the scalar timestep, its reachable set is a curve an 8-dimensional affine subspace covers to ~1.5e-5
    relative RMS - far below one bfloat16 rounding step of the weights themselves. Folding the subspace into the
    projections leaves an 8-wide input and removes 26 GB per partition.

    Only what builds the timestep path differs from [`MiniMaxH3Transformer3DModel`]: `time_proj` becomes an identity,
    `time_embedder` becomes [`MiniMaxH3PrunedTimeEmbedder`], and the AdaLN projections take `adaln_rank` inputs.
    `forward` is inherited unchanged. The module names are the released ones, so LoRAs trained against a pruned
    checkpoint - what the common trainers use by default - load natively, and `load_lora_adapter` projects LoRAs
    trained against the released checkpoint's `time_embed_dim`-wide projections onto the same coordinates.

    Args:
        adaln_rank (`int`, defaults to `8`):
            The width of the timestep coordinates every AdaLN projection consumes.
        time_table_size (`int`, defaults to `1025`):
            The number of uniformly spaced timesteps the coordinate table holds; values in between are interpolated
            linearly.

        Every other argument is [`MiniMaxH3Transformer3DModel`]'s and carries the same meaning. `freq_dim` and
        `time_embed_hidden_dim` are kept in the config, unused, so a pruned config still records the shape of the
        released timestep MLP it was folded from.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["MiniMaxH3TransformerBlock", "MiniMaxH3TokenRefinerBlock", "MiniMaxH3PrunedAdaLayerNormOut"]
    _repeated_blocks = ["MiniMaxH3TransformerBlock", "MiniMaxH3TokenRefinerBlock"]
    _skip_layerwise_casting_patterns = ["norm"]
    # The released checkpoint's mixed-precision split - patch projections, output heads and the timestep path in
    # float32, the block stack in bfloat16 - plus the folded AdaLN biases, for the reason given on the modulation
    # module. Entries are matched against the dot-separated segments of each parameter name.
    _keep_in_fp32_modules = [
        "proj_in",
        "audio_proj_in",
        "time_embedder",
        "proj_out",
        "audio_proj_out",
        "rope",
        "folded_bias",
        "adaln_basis",
        "adaln_mean",
    ]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        hidden_size: int = 5376,
        num_layers: int = 50,
        num_refiner_layers: int = 2,
        ffn_dim: int = 14336,
        in_channels: int = 24,
        audio_in_channels: int = 32,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_dim: int = 5120,
        freq_dim: int = 256,
        time_embed_hidden_dim: int = 5376,
        time_embed_dim: int = 2688,
        rope_freq_dim: int = 16,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
        adaln_rank: int = 8,
        time_table_size: int = 1025,
    ) -> None:
        # `MiniMaxH3Transformer3DModel.__init__` is itself wrapped by `register_to_config`, so calling it would
        # register the released config over this one - and would allocate the 26 GB of AdaLN projections this class
        # exists to avoid. The module tree is built here instead; everything but the timestep path is verbatim.
        nn.Module.__init__(self)

        video_patch_dim = in_channels * patch_size[0] * patch_size[1] * patch_size[2]

        # 1. Per-modality input projections
        self.proj_in = nn.Linear(video_patch_dim, hidden_size, bias=True)
        self.audio_proj_in = nn.Linear(audio_in_channels, hidden_size, bias=True)
        self.context_embedder = nn.Linear(text_dim, hidden_size, bias=True)

        # 2. Timestep coordinates, shared by every AdaLN projection
        self.time_proj = MiniMaxH3PrunedTimeProj()
        self.time_embedder = MiniMaxH3PrunedTimeEmbedder(table_size=time_table_size, adaln_rank=adaln_rank)

        # 2b. The affine map the fold was performed with: `silu(time_embedder(t)) ~= adaln_mean + c(t) @ adaln_basis`.
        #     Nothing in `forward` reads these - the folded projections already carry them. They are stored so that a
        #     LoRA trained on the released 2688-wide projections can be mapped onto these coordinates at load time;
        #     see `load_lora_adapter`. 97 KB per partition.
        self.register_buffer("adaln_basis", torch.zeros(adaln_rank, time_embed_dim), persistent=True)
        self.register_buffer("adaln_mean", torch.zeros(time_embed_dim), persistent=True)

        # 3. Rotary embedding over the packed (t, h, w) grid
        self.rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=rope_freq_dim, rope_theta=rope_theta)

        # 4. Text stream refiner
        self.token_refiner = MiniMaxH3TokenRefiner(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            ffn_dim=ffn_dim,
            num_layers=num_refiner_layers,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
            final_norm_eps=final_norm_eps,
        )

        # 5. The block stack, with each block's AdaLN projection narrowed to the timestep coordinates. The block is
        #    built with `time_embed_dim=adaln_rank` so its own projection is already the right shape, then swapped
        #    for the pruned module, which drops the `silu` and moves the bias to float32.
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxH3TransformerBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    ffn_dim=ffn_dim,
                    time_embed_dim=adaln_rank,
                    norm_eps=norm_eps,
                    qk_norm_eps=qk_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        for block in self.transformer_blocks:
            block.adaln_proj = MiniMaxH3PrunedAdaLayerNormModulation(adaln_rank=adaln_rank, hidden_size=hidden_size)

        # 6. Shared output norm and the two per-modality output heads
        self.norm_out = MiniMaxH3PrunedAdaLayerNormOut(
            hidden_size=hidden_size, adaln_rank=adaln_rank, eps=final_norm_eps
        )
        self.proj_out = nn.Linear(hidden_size, video_patch_dim, bias=True)
        self.audio_proj_out = nn.Linear(hidden_size, audio_in_channels, bias=True)

        self.gradient_checkpointing = False

    # -- LoRA ---------------------------------------------------------------------------------------------------

    def project_adaln_lora(self, state_dict: dict, prefix: str | None = None) -> tuple[dict, dict]:
        r"""Map a LoRA's AdaLN factors from the released timestep embedding onto the pruned coordinates.

        The released projection reads `x = silu(time_embedder(t))`, so a LoRA on it contributes

            lora_B @ (lora_A @ x)  =  lora_B @ (lora_A @ (mean + basis.T @ c))
                                   =  (lora_B @ (lora_A @ basis.T)) @ c  +  lora_B @ (lora_A @ mean)

        which is a rank-preserving `[rank, adaln_rank]` `lora_A` over the pruned coordinates plus a constant output
        offset. Both are computed in float64 from the file's own factors; the only error is the rank-8 subspace's
        own residual on the timestep curve, 1.5e-5 relative, which is ~250x below one bfloat16 step of the weights
        being adapted.

        Returns `(state_dict, {module path: offset})`, the state dict unchanged and the offsets empty when the AdaLN
        factors are already `adaln_rank`-wide (a LoRA trained against a pruned checkpoint - most of them).

        Every AdaLN module in a file has to be one or the other. A file that mixes widths is not something this can
        half-apply, so it raises.

        `prefix` scopes this to one component's keys, the same way `load_lora_adapter` scopes the load - a file that
        names both partitions holds two different adapters under one roof, and only one of them is going into this
        module.
        """
        adaln_rank = self.config.adaln_rank
        time_embed_dim = self.config.time_embed_dim

        modules = {}
        for key in state_dict:
            if prefix is not None and not key.startswith(f"{prefix}."):
                continue
            match = ADALN_LORA_A_KEY.search(key)
            if match is not None:
                modules[key] = match.group(1)
        if not modules:
            return state_dict, {}

        widths = sorted({int(state_dict[key].shape[1]) for key in modules})
        if widths == [adaln_rank]:
            return state_dict, {}
        if widths != [time_embed_dim]:
            odd = sorted(
                {modules[key] for key in modules if int(state_dict[key].shape[1]) != max(widths)},
                key=lambda name: (name != "norm_out.linear", name),
            )
            raise ValueError(
                f"This LoRA's {len(modules)} AdaLN projections read inputs of width {widths}. On this checkpoint they "
                f"have to be uniformly {adaln_rank} wide (trained against a pruned checkpoint, loaded as they are) or "
                f"uniformly {time_embed_dim} wide (trained against the released checkpoint, projected onto the pruned "
                "coordinates at load). An adapter cannot be applied to some of its AdaLN modules and not others, so "
                f"nothing was loaded. The minority width is on: {odd[:4]}{' ...' if len(odd) > 4 else ''}."
            )

        basis = self.adaln_basis
        mean = self.adaln_mean
        if basis.abs().sum() == 0:
            raise ValueError(
                "Projecting a released-checkpoint LoRA needs `adaln_basis` and `adaln_mean`, the affine map this "
                "checkpoint's AdaLN projections were folded with, and this model was loaded without them. Re-download "
                "the repository: they ship as `adaln_affine.safetensors` next to the weights."
            )
        basis = basis.double()
        mean = mean.double()

        projected = dict(state_dict)
        offsets = {}
        for key, module in modules.items():
            lora_b_key = key[: -len("lora_A.weight")] + "lora_B.weight"
            if lora_b_key not in state_dict:
                raise ValueError(
                    f"{key} has no matching {lora_b_key}. The constant term of the projection is "
                    "`lora_B @ (lora_A @ mean)`, so both factors have to be present; nothing was loaded."
                )
            lora_a = state_dict[key].to(device=basis.device, dtype=torch.float64)
            lora_b = state_dict[lora_b_key].to(device=basis.device, dtype=torch.float64)
            # float32 rather than the file's bfloat16: the projection is exact arithmetic on the file's factors and
            # there is no reason to round it twice. PEFT casts to the adapter's dtype when it loads them.
            projected[key] = (lora_a @ basis.T).to(torch.float32).cpu().contiguous()
            offsets[module] = (lora_b @ (lora_a @ mean)).to(torch.float32).cpu().contiguous()
        return projected, offsets

    def load_lora_adapter(self, pretrained_model_name_or_path_or_dict, prefix="transformer", hotswap=False, **kwargs):
        r"""`PeftAdapterMixin.load_lora_adapter`, with the AdaLN projection of [`project_adaln_lora`] in front of it.

        LoRAs trained against a pruned checkpoint pass through untouched. LoRAs trained against the released
        checkpoint's 2688-wide AdaLN projections - the official turbo LoRA and its conversions - are mapped onto the
        pruned coordinates here, which is the only thing that ever stopped them loading. Everything outside the AdaLN
        path is identical between the two checkpoints and is neither inspected nor changed.
        """
        state_dict = pretrained_model_name_or_path_or_dict
        if not isinstance(state_dict, dict):
            from diffusers.loaders.lora_base import _fetch_state_dict

            state_dict, _ = _fetch_state_dict(
                pretrained_model_name_or_path_or_dict=state_dict,
                weight_name=kwargs.get("weight_name"),
                use_safetensors=kwargs.get("use_safetensors", True),
                local_files_only=kwargs.get("local_files_only"),
                cache_dir=kwargs.get("cache_dir"),
                force_download=kwargs.get("force_download", False),
                proxies=kwargs.get("proxies"),
                token=kwargs.get("token"),
                revision=kwargs.get("revision"),
                subfolder=kwargs.get("subfolder"),
                user_agent={"file_type": "attn_procs_weights", "framework": "pytorch"},
                allow_pickle=False,
                metadata=kwargs.get("metadata"),
            )

        state_dict, offsets = self.project_adaln_lora(state_dict, prefix=prefix)
        if offsets:
            logger.info(
                f"Projecting {len(offsets)} AdaLN LoRA modules from the released {self.config.time_embed_dim}-wide "
                f"timestep embedding onto this checkpoint's {self.config.adaln_rank} pruned coordinates; each one's "
                "constant term is carried as a float32 offset on the modulation."
            )

        before = set(getattr(self, "peft_config", None) or ())
        super().load_lora_adapter(state_dict, prefix=prefix, hotswap=hotswap, **kwargs)
        if not offsets:
            return

        added = set(getattr(self, "peft_config", None) or ()) - before
        adapter_name = added.pop() if len(added) == 1 else kwargs.get("adapter_name")
        if adapter_name is None:
            raise RuntimeError(
                "The AdaLN projection could not tell which adapter was just loaded, so its constant terms were not "
                f"attached and the adapter is incomplete. Adapters before: {sorted(before)}, after: "
                f"{sorted(getattr(self, 'peft_config', None) or ())}. Pass `adapter_name` explicitly."
            )
        for module, offset in offsets.items():
            self.get_submodule(module.rsplit(".", 1)[0]).register_lora_adaln_offset(adapter_name, offset)

    # -- 8-bit compute -----------------------------------------------------------------------------------------
    #
    # `enable_convrot` and `quantize_8bit` are opt-in and change nothing until called.

    def convrot_layers(self) -> list[str]:
        r"""The attention and feed-forward linears ConvRot applies to (300 of them: 50 blocks x 6).

        A PEFT-wrapped target appears as `....ff.net.2.base_layer` and is matched as such, so a model that
        already has a LoRA attached rotates its *base* layers and leaves the adapters alone.
        """
        names = []
        for name, module in self.named_modules():
            if not isinstance(module, nn.Linear) or "transformer_blocks" not in name:
                continue
            stem = name[: -len(".base_layer")] if name.endswith(".base_layer") else name
            if any(stem.endswith(suffix) for suffix in CONVROT_SUFFIXES):
                names.append(name)
        return names

    def enable_convrot(self, group_size: int = 256) -> list[str]:
        r"""Fold `W <- W @ H` into every ConvRot target and switch it to [`MiniMaxH3ConvRotLinear`].

        The fold runs in float64 and rounds back to the weight's dtype once. Idempotence is not claimed: `H` is
        an involution, so calling this twice restores the original weights while leaving the online rotation in
        place, which is wrong. Call it once, on a freshly loaded model, before quantizing.
        """
        names = self.convrot_layers()
        lookup = dict(self.named_modules())
        for name in names:
            module = lookup[name]
            weight = module.weight
            # In place: reassigning `weight.data` 300 times leaves the old storages to the caching allocator
            # and fragments it badly on a card holding the whole model.
            weight.data.copy_(_rotate(weight.data.double(), group_size))
            module.__class__ = MiniMaxH3ConvRotLinear
            module.convrot_groupsize = group_size
        self._convrot_layers = names
        return names

    def convrot_filter(self, module: nn.Module, fqn: str) -> bool:
        r"""`filter_fn` for `torchao.quantize_`: the ConvRot targets and nothing else."""
        return fqn in getattr(self, "_convrot_layers", ()) or fqn in self.convrot_layers()

    def quantize_8bit(self, config=None, group_size: int = 256, device=None):
        r"""Rotate and quantize one linear at a time, so the working set is one weight rather than the model.

        `config` is any torchao config; the default is int8 dynamic activations with int8 weights, which is the
        configuration measured closest to bfloat16 on this model. `device` is where the fold and the
        quantization run - point it at the GPU when the model itself is on the CPU and the transient cost is one
        `[28672, 5376]` weight rather than 40 GB.

        Returns `self`.
        """
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

        if config is None:
            config = Int8DynamicActivationInt8WeightConfig()
        names = self.convrot_layers()
        lookup = dict(self.named_modules())
        for name in names:
            module = lookup[name]
            home = module.weight.device
            module.to(device or home)
            weight = module.weight
            # In place: reassigning `weight.data` 300 times leaves the old storages to the caching allocator
            # and fragments it badly on a card holding the whole model.
            weight.data.copy_(_rotate(weight.data.double(), group_size))
            module.__class__ = MiniMaxH3ConvRotLinear
            module.convrot_groupsize = group_size
            quantize_(module, config)
            module.to(home)
        self._convrot_layers = names
        return self


# The pruned transformer's config.json (the repo's, minus _class_name, _diffusers_version and auto_map).
PRUNED_CONFIG = {
    "adaln_rank": 8,
    "attention_head_dim": 128,
    "audio_in_channels": 32,
    "ffn_dim": 14336,
    "final_norm_eps": 1e-05,
    "freq_dim": 256,
    "hidden_size": 5376,
    "in_channels": 24,
    "norm_eps": 1e-05,
    "num_attention_heads": 56,
    "num_layers": 50,
    "num_refiner_layers": 2,
    "patch_size": [1, 2, 2],
    "qk_norm_eps": 1e-05,
    "rope_freq_dim": 16,
    "rope_theta": 10000.0,
    "text_dim": 5120,
    "time_embed_dim": 2688,
    "time_embed_hidden_dim": 5376,
    "time_table_size": 1025,
}
