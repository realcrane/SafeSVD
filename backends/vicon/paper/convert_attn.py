"""
convert_attn.py
===============
Replace PyTorch's native nn.MultiheadAttention (fused in_proj_weight) with an
SVD-friendly version that exposes q_proj, k_proj, v_proj, out_proj as separate
nn.Linear modules.

Usage:
    from convert_attn import convert_attn_to_linear, revert_attn_from_linear

    # Before profiling / SVD
    original_attns = convert_attn_to_linear(model)

    # ... run SVD profiling, decomposition, etc. ...

    # (Optional) If you want to revert (usually you don't — keep the converted version)
    # revert_attn_from_linear(model, original_attns)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from collections import OrderedDict


class ExplicitMultiheadAttention(nn.Module):
    """
    Drop-in replacement for nn.MultiheadAttention that uses four separate
    nn.Linear layers (q_proj, k_proj, v_proj, out_proj) instead of a fused
    in_proj_weight parameter.

    This makes every projection visible to code that iterates over nn.Linear
    modules (e.g., SVD profiling, Fisher collection, weight replacement).
    """

    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.0, bias: bool = True,
                 batch_first: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # ── Compatibility attributes ──
        # nn.TransformerEncoderLayer.forward inspects these on self.self_attn
        # BEFORE calling self_attn(...) to decide fast-path vs standard-path.
        # Setting them to None forces the standard path (which just calls forward).
        self.in_proj_weight = None
        self.in_proj_bias = None
        self._qkv_same_embed_dim = True

    @classmethod
    def from_native(cls, mha: nn.MultiheadAttention) -> "ExplicitMultiheadAttention":
        """
        Create an ExplicitMultiheadAttention from a native nn.MultiheadAttention,
        copying all weights exactly.
        """
        embed_dim = mha.embed_dim
        num_heads = mha.num_heads
        dropout = mha.dropout
        bias = mha.in_proj_bias is not None
        batch_first = mha.batch_first

        module = cls(embed_dim, num_heads, dropout=dropout, bias=bias,
                     batch_first=batch_first)

        # Split the fused in_proj_weight [3*E, E] -> three [E, E]
        with torch.no_grad():
            if mha.in_proj_weight is not None:
                q_w, k_w, v_w = mha.in_proj_weight.chunk(3, dim=0)
                module.q_proj.weight.copy_(q_w)
                module.k_proj.weight.copy_(k_w)
                module.v_proj.weight.copy_(v_w)
            elif hasattr(mha, 'q_proj_weight'):
                # Rare: separate projection weights (when kdim != vdim)
                module.q_proj.weight.copy_(mha.q_proj_weight)
                module.k_proj.weight.copy_(mha.k_proj_weight)
                module.v_proj.weight.copy_(mha.v_proj_weight)

            if bias and mha.in_proj_bias is not None:
                q_b, k_b, v_b = mha.in_proj_bias.chunk(3, dim=0)
                module.q_proj.bias.copy_(q_b)
                module.k_proj.bias.copy_(k_b)
                module.v_proj.bias.copy_(v_b)

            # out_proj is already nn.Linear in native MHA
            module.out_proj.weight.copy_(mha.out_proj.weight)
            if mha.out_proj.bias is not None:
                module.out_proj.bias.copy_(mha.out_proj.bias)

        return module

    def forward(self, query, key=None, value=None,
                key_padding_mask=None, need_weights=False,
                attn_mask=None, is_causal=False):
        """
        Mimics nn.MultiheadAttention.forward signature.
        For VICON, the TransformerEncoder calls self_attn(src, src, src, ...).
        """
        if key is None:
            key = query
        if value is None:
            value = query

        # Handle batch_first
        if self.batch_first:
            # input: [B, T, E] -> we work in [T, B, E] internally
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        T, B, E = query.shape
        S = key.shape[0]

        # Project
        q = self.q_proj(query)  # [T, B, E]
        k = self.k_proj(key)    # [S, B, E]
        v = self.v_proj(value)  # [S, B, E]

        # Reshape for multi-head: [T, B, nH, hD] -> [B, nH, T, hD]
        # (using B*nH as batch dim for bmm)
        nH = self.num_heads
        hD = self.head_dim

        q = q.view(T, B, nH, hD).permute(1, 2, 0, 3)  # [B, nH, T, hD]
        k = k.view(S, B, nH, hD).permute(1, 2, 0, 3)  # [B, nH, S, hD]
        v = v.view(S, B, nH, hD).permute(1, 2, 0, 3)  # [B, nH, S, hD]

        # ── Prepare attn_mask for F.scaled_dot_product_attention ──
        # SDPA expects mask shape [B, nH, T, S] or [T, S] or [1, 1, T, S]
        # Bool mask: True = IGNORED (opposite of masked_fill convention in older code)
        # Float mask: additive
        sdpa_mask = None
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # [T, S] -> [1, 1, T, S]  works for both bool and float
                sdpa_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                # [B*nH, T, S] -> [B, nH, T, S]
                sdpa_mask = attn_mask.view(B, nH, T, S)

        # Merge key_padding_mask into sdpa_mask
        if key_padding_mask is not None:
            # key_padding_mask: [B, S] -> [B, 1, 1, S]
            kpm = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
            if sdpa_mask is not None:
                if sdpa_mask.dtype == torch.bool:
                    sdpa_mask = sdpa_mask | kpm
                else:
                    sdpa_mask = sdpa_mask.masked_fill(kpm, float('-inf'))
            else:
                sdpa_mask = kpm  # bool

        # F.scaled_dot_product_attention uses Flash/Memory-efficient kernels
        # -> never materializes the full [B, nH, T, S] attention matrix
        dropout_p = self.dropout if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=dropout_p,
        )  # [B, nH, T, hD]

        # Reshape back: [B, nH, T, hD] -> [T, B, E]
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(T, B, E)

        # Output projection
        attn_output = self.out_proj(attn_output)

        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)  # [B, T, E]

        if need_weights:
            # SDPA doesn't return attention weights; would need manual fallback.
            # In practice VICON never requests weights, so just warn.
            import warnings
            warnings.warn("need_weights=True not supported with SDPA path, returning None")
        return attn_output, None


# =========================================================================
# Conversion functions
# =========================================================================

def convert_attn_to_linear(model: nn.Module) -> Dict[str, nn.MultiheadAttention]:
    """
    Find all nn.MultiheadAttention modules in the model and replace them
    with ExplicitMultiheadAttention (which uses separate nn.Linear layers).

    Args:
        model: The model to convert (modified in-place).

    Returns:
        A dict mapping module paths to the original nn.MultiheadAttention modules,
        in case you want to revert later.
    """
    originals = {}

    # Collect targets first to avoid modifying dict during iteration
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            targets.append((name, module))

    for name, mha in targets:
        explicit = ExplicitMultiheadAttention.from_native(mha)
        # Move to same device
        device = next(mha.parameters()).device
        explicit = explicit.to(device)

        # Replace in the parent module
        _set_module_by_name(model, name, explicit)
        originals[name] = mha
        print(f"[convert_attn] Replaced {name}: MultiheadAttention -> ExplicitMultiheadAttention")

    print(f"[convert_attn] Converted {len(targets)} attention modules total.")
    return originals


def revert_attn_from_linear(model: nn.Module,
                             originals: Dict[str, nn.MultiheadAttention]):
    """
    Revert the conversion: put back the original nn.MultiheadAttention modules.
    NOTE: This does NOT copy back any weight changes made to the explicit version.
    It literally restores the original (pre-conversion) modules.
    """
    for name, mha in originals.items():
        _set_module_by_name(model, name, mha)
        print(f"[revert_attn] Restored {name}: ExplicitMultiheadAttention -> MultiheadAttention")


def sync_back_to_native(model: nn.Module,
                         originals: Dict[str, nn.MultiheadAttention]):
    """
    After SVD decomposition modifies the explicit Linear weights, sync those
    weights back into the original nn.MultiheadAttention format and restore it.

    This is useful if downstream code expects the native MHA interface.
    """
    for name, orig_mha in originals.items():
        explicit = _get_module_by_name(model, name)
        if not isinstance(explicit, ExplicitMultiheadAttention):
            print(f"[sync_back] Skipping {name}: not ExplicitMultiheadAttention")
            continue

        with torch.no_grad():
            # Fuse q/k/v back into in_proj_weight
            orig_mha.in_proj_weight.copy_(torch.cat([
                explicit.q_proj.weight,
                explicit.k_proj.weight,
                explicit.v_proj.weight,
            ], dim=0))

            if orig_mha.in_proj_bias is not None:
                orig_mha.in_proj_bias.copy_(torch.cat([
                    explicit.q_proj.bias,
                    explicit.k_proj.bias,
                    explicit.v_proj.bias,
                ], dim=0))

            orig_mha.out_proj.weight.copy_(explicit.out_proj.weight)
            if orig_mha.out_proj.bias is not None:
                orig_mha.out_proj.bias.copy_(explicit.out_proj.bias)

        _set_module_by_name(model, name, orig_mha)
        print(f"[sync_back] Synced and restored {name}")


# =========================================================================
# Helpers
# =========================================================================

def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    """Set a submodule by its dot-separated name path."""
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    if parts[-1].isdigit():
        parent[int(parts[-1])] = new_module
    else:
        setattr(parent, parts[-1], new_module)


def _get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    """Get a submodule by its dot-separated name path."""
    parts = name.split(".")
    module = model
    for p in parts:
        if p.isdigit():
            module = module[int(p)]
        else:
            module = getattr(module, p)
    return module