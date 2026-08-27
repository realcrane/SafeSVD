"""VICON attention conversion shared by compression and evaluation.

VICON uses native :class:`torch.nn.MultiheadAttention`, whose fused QKV
parameter is invisible to SAFE-SVD's Linear-only compressor.  This module is a
small, serialisable replacement that exposes Q/K/V/output projections as
ordinary ``nn.Linear`` layers.
"""

from __future__ import annotations

import types
from contextlib import contextmanager
from collections.abc import Mapping
from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as functional

from safesvd.adapters.common_runtime import set_module_by_name


class ExplicitMultiheadAttention(nn.Module):
    """A VICON-compatible MultiheadAttention with explicit Linear projections."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # TransformerEncoderLayer inspects these attributes before it calls
        # attention.  ``None`` deliberately disables its fused fast path.
        self.in_proj_weight = None
        self.in_proj_bias = None
        self._qkv_same_embed_dim = True
        self.bias_k = None
        self.bias_v = None
        self.add_zero_attn = False
        self.kdim = embed_dim
        self.vdim = embed_dim

    @classmethod
    def from_native(cls, attention: nn.MultiheadAttention) -> "ExplicitMultiheadAttention":
        converted = cls(
            attention.embed_dim,
            attention.num_heads,
            dropout=attention.dropout,
            bias=attention.in_proj_bias is not None,
            batch_first=attention.batch_first,
        ).to(device=attention.out_proj.weight.device, dtype=attention.out_proj.weight.dtype)
        with torch.no_grad():
            if attention.in_proj_weight is not None:
                query, key, value = attention.in_proj_weight.chunk(3, dim=0)
                converted.q_proj.weight.copy_(query)
                converted.k_proj.weight.copy_(key)
                converted.v_proj.weight.copy_(value)
            else:
                # Separate projections occur for nonstandard kdim/vdim.  The
                # VICON checkpoints use the common equal-dimension case, but
                # retaining this branch produces a useful error if they do not.
                if attention.q_proj_weight is None or attention.k_proj_weight is None or attention.v_proj_weight is None:
                    raise ValueError("Cannot expose MultiheadAttention projections")
                if attention.q_proj_weight.shape != converted.q_proj.weight.shape:
                    raise ValueError("VICON attention conversion requires equal q/k/v dimensions")
                converted.q_proj.weight.copy_(attention.q_proj_weight)
                converted.k_proj.weight.copy_(attention.k_proj_weight)
                converted.v_proj.weight.copy_(attention.v_proj_weight)
            if attention.in_proj_bias is not None:
                query_bias, key_bias, value_bias = attention.in_proj_bias.chunk(3, dim=0)
                converted.q_proj.bias.copy_(query_bias)
                converted.k_proj.bias.copy_(key_bias)
                converted.v_proj.bias.copy_(value_bias)
            converted.out_proj.weight.copy_(attention.out_proj.weight)
            if attention.out_proj.bias is not None:
                converted.out_proj.bias.copy_(attention.out_proj.bias)
        return converted

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        **_unused: object,
    ) -> tuple[torch.Tensor, None]:
        key = query if key is None else key
        value = query if value is None else value
        if self.batch_first:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        time, batch, embedding = query.shape
        source_time = key.shape[0]
        if embedding != self.embed_dim:
            raise ValueError(f"Expected embedding {self.embed_dim}, received {embedding}")

        q = self.q_proj(query).view(time, batch, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = self.k_proj(key).view(source_time, batch, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = self.v_proj(value).view(source_time, batch, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        mask = _attention_mask(attn_mask, key_padding_mask, batch, self.num_heads, time, source_time)
        dropout = self.dropout if self.training else 0.0
        use_causal = is_causal and mask is None
        if hasattr(functional, "scaled_dot_product_attention"):
            output = functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=dropout, is_causal=use_causal,
            )
        else:  # pragma: no cover - only exercised on old PyTorch releases
            scores = q @ k.transpose(-2, -1) / self.head_dim**0.5
            if mask is not None:
                if mask.dtype == torch.bool:
                    # SDPA treats True as *allowed*, unlike MultiheadAttention.
                    scores = scores.masked_fill(~mask, float("-inf"))
                else:
                    scores = scores + mask
            elif is_causal:
                causal = torch.ones(time, source_time, dtype=torch.bool, device=scores.device).tril()
                scores = scores.masked_fill(~causal, float("-inf"))
            output = functional.dropout(functional.softmax(scores, dim=-1), p=dropout, training=self.training) @ v
        output = output.permute(2, 0, 1, 3).contiguous().view(time, batch, embedding)
        output = self.out_proj(output)
        if self.batch_first:
            output = output.transpose(0, 1)
        if need_weights:
            # Matching the source adapter: VICON's model does not consume
            # attention weights, and materialising them defeats the purpose of
            # using scaled-dot-product attention here.
            import warnings

            warnings.warn("ExplicitMultiheadAttention does not materialise attention weights", stacklevel=2)
        return output, None


def _attention_mask(
    attention_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
    batch: int,
    heads: int,
    time: int,
    source_time: int,
) -> torch.Tensor | None:
    """Translate ``MultiheadAttention`` masks to SDPA's mask convention."""
    result: torch.Tensor | None = None
    if attention_mask is not None:
        if attention_mask.ndim == 2:
            result = attention_mask.unsqueeze(0).unsqueeze(0)
        elif attention_mask.ndim == 3:
            result = attention_mask.reshape(batch, heads, time, source_time)
        else:
            result = attention_mask
        # ``nn.MultiheadAttention`` uses True for an excluded position;
        # ``scaled_dot_product_attention`` uses True for an allowed position.
        if result.dtype == torch.bool:
            result = ~result
    if key_padding_mask is not None:
        padding = key_padding_mask.unsqueeze(1).unsqueeze(2)
        if padding.dtype == torch.bool:
            if result is None:
                result = ~padding
            elif result.dtype == torch.bool:
                result = result & ~padding
            else:
                result = result.masked_fill(padding, float("-inf"))
        else:
            if result is None:
                result = padding
            elif result.dtype == torch.bool:
                # Native MHA permits the mixed-mask case (with a warning).
                # Convert SDPA's allowed-position bool mask to additive form.
                result = torch.zeros_like(result, dtype=padding.dtype).masked_fill(~result, float("-inf")) + padding
            else:
                result = result + padding
    return result


def _cropped_safe_forward(*, checkpoint_layers: bool):
    """Return ViCON's non-inplace ``ICON_CROPPED`` forward implementation."""
    from VICON.src.model_utils import depatchify, patchify
    checkpoint = None
    if checkpoint_layers:
        from torch.utils.checkpoint import checkpoint as checkpoint_function

        checkpoint = checkpoint_function

    def forward(self: nn.Module, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg
        c_out = cfg.transformer.dim_channel
        dim_token = cfg.transformer.dim_token
        p_i = cfg.patch_num_in
        p_e = cfg.patch_num_out
        p_res = cfg.patch_resolution
        demo_num = cfg.demo_num
        initial, end, _channel_mask = x
        batch_size = initial.shape[0]
        feature_initial = patchify(initial.view(-1, *initial.shape[-3:]), patch_num=p_i)
        feature_end = patchify(end.view(-1, *end.shape[-3:]), patch_num=p_e)
        feature = self.pre_proj(torch.cat((feature_initial, feature_end), dim=1))
        if cfg.use_patch_pos_encoding:
            feature = torch.cat((
                feature[:, :p_i * p_i] + self.patch_pos_encoding_init,
                feature[:, p_i * p_i:] + self.patch_pos_encoding_end,
            ), dim=1)
        feature = feature.view(batch_size, -1, p_i * p_i + p_e * p_e, dim_token)
        if cfg.use_func_pos_encoding:
            feature = torch.cat((
                feature[:, :, :p_i * p_i] + self.func_pos_encoding_init.view(1, -1, 1, dim_token),
                feature[:, :, p_i * p_i:] + self.func_pos_encoding_end.view(1, -1, 1, dim_token),
            ), dim=2)
        feature = feature.view(batch_size, -1, dim_token)
        if checkpoint is None:
            feature = self.transformer(feature, mask=self.mask)
        else:
            for layer in self.transformer.layers:
                feature = checkpoint(layer, feature, self.mask, None, use_reentrant=False)
            if self.transformer.norm is not None:
                feature = self.transformer.norm(feature)
        feature = feature.view(batch_size, demo_num, -1, dim_token)[:, :, :p_i * p_i, :]
        feature = self.post_proj(feature)
        feature = feature.view(batch_size * demo_num, p_i, p_i, -1)
        feature = feature.view(batch_size * demo_num, p_e, p_i // p_e, p_e, p_i // p_e, -1)
        feature = feature.permute(0, 1, 3, 2, 4, 5).reshape(batch_size * demo_num, p_e * p_e, -1)
        feature = depatchify(feature, patch_num=p_e, c=c_out, h=p_res, w=p_res)
        return feature.view(batch_size, demo_num, *feature.shape[-3:])

    return forward


@contextmanager
def patched_vicon_fisher_forward(model: nn.Module, *, checkpoint_layers: bool = True) -> Iterator[None]:
    """Use ViCON's safe crop-model forward while collecting Fisher gradients.

    Full backward hooks on the original cropped ICON forward can conflict with
    its in-place/fast transformer path.  This is the equivalent of the paper
    collector's temporary patch, kept in SAFE-SVD rather than importing it.
    """
    raw_model = model.module if hasattr(model, "module") else model
    original_forward = None
    if type(raw_model).__name__ == "ICON_CROPPED":
        original_forward = raw_model.forward
        raw_model.forward = types.MethodType(
            _cropped_safe_forward(checkpoint_layers=checkpoint_layers), raw_model,
        )
    try:
        yield
    finally:
        if original_forward is not None:
            raw_model.forward = original_forward


def convert_attn_to_linear(model: nn.Module) -> dict[str, nn.MultiheadAttention]:
    """Replace all native VICON attention modules in-place and return originals."""
    originals: dict[str, nn.MultiheadAttention] = {}
    targets = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.MultiheadAttention)]
    for name, module in targets:
        set_module_by_name(model, name, ExplicitMultiheadAttention.from_native(module))
        originals[name] = module
    return originals


def disable_transformer_fast_path(model: nn.Module) -> nn.Module:
    """Keep TransformerEncoderLayer on its Python path after attention conversion."""
    def self_attention_block(
        layer: nn.TransformerEncoderLayer,
        value: torch.Tensor,
        src_mask: torch.Tensor | None,
        src_key_padding_mask: torch.Tensor | None,
        is_causal: bool,
    ) -> torch.Tensor:
        # ``is_causal`` was added to the private helper in newer PyTorch releases.
        # VICON itself supports older releases, so preserve that compatibility.
        try:
            return layer._sa_block(value, src_mask, src_key_padding_mask, is_causal=is_causal)
        except TypeError:
            return layer._sa_block(value, src_mask, src_key_padding_mask)

    def forward(self: nn.TransformerEncoderLayer, src: torch.Tensor, src_mask=None,
                src_key_padding_mask=None, is_causal: bool = False) -> torch.Tensor:
        value = src
        if self.norm_first:
            value = value + self_attention_block(
                self, self.norm1(value), src_mask, src_key_padding_mask, is_causal,
            )
            value = value + self._ff_block(self.norm2(value))
        else:
            value = self.norm1(value + self_attention_block(
                self, value, src_mask, src_key_padding_mask, is_causal,
            ))
            value = self.norm2(value + self._ff_block(value))
        return value

    for module in model.modules():
        if isinstance(module, nn.TransformerEncoderLayer):
            module.forward = types.MethodType(forward, module)
    return model


def is_explicit_attention_state_dict(state_dict: Mapping[str, torch.Tensor]) -> bool:
    """Whether a checkpoint contains the expanded QKV topology."""
    return any(".q_proj.weight" in key for key in state_dict)
