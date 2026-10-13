"""Local non-inplace/checkpoint context copied from VICON MySVD."""

from __future__ import annotations

import torch
from contextlib import contextmanager
from torch.utils.checkpoint import checkpoint

def _make_safe_forward(model_instance):
    from VICON.src.model_utils import patchify, depatchify

    def safe_forward(self, x):
        cfg = self.cfg
        c_out = cfg.transformer.dim_channel
        dim_token = cfg.transformer.dim_token
        p_i = cfg.patch_num_in
        p_e = cfg.patch_num_out
        p_res = cfg.patch_resolution
        demo_num = cfg.demo_num

        init, end, c_mask = x
        bs, _, _, _, _ = init.shape
        feature_init = init.view(-1, *init.shape[-3:])
        feature_end = end.view(-1, *end.shape[-3:])

        feature_init = patchify(feature_init, patch_num=p_i)
        feature_end = patchify(feature_end, patch_num=p_e)
        feature = torch.cat((feature_init, feature_end), dim=1)
        feature = self.pre_proj(feature)

        if self.cfg.use_patch_pos_encoding:
            part_init = feature[:, : p_i * p_i] + self.patch_pos_encoding_init
            part_end  = feature[:, p_i * p_i :]  + self.patch_pos_encoding_end
            feature = torch.cat([part_init, part_end], dim=1)

        feature = feature.view(bs, -1, p_i * p_i + p_e * p_e, dim_token)

        if self.cfg.use_func_pos_encoding:
            part_init = feature[:, :, : p_i * p_i] + self.func_pos_encoding_init.view(1, -1, 1, dim_token)
            part_end  = feature[:, :, p_i * p_i :]  + self.func_pos_encoding_end.view(1, -1, 1, dim_token)
            feature = torch.cat([part_init, part_end], dim=2)

        feature = feature.view(bs, -1, dim_token)
        feature = self.transformer(feature, mask=self.mask)
        feature = feature.view(bs, demo_num, -1, dim_token)
        feature = feature[:, :, : p_i * p_i, :]

        feature = self.post_proj(feature)
        feature = feature.view(bs * demo_num, p_i, p_i, -1)
        feature = feature.view(bs * demo_num, p_e, p_i // p_e, p_e, p_i // p_e, -1)
        feature = feature.permute(0, 1, 3, 2, 4, 5)
        feature = feature.reshape(bs * demo_num, p_e * p_e, -1)
        feature = depatchify(feature, patch_num=p_e, c=c_out, h=p_res, w=p_res)
        feature = feature.view(bs, demo_num, *feature.shape[-3:])

        return feature

    return safe_forward


def _make_safe_forward_checkpointed(model_instance):
    from VICON.src.model_utils import patchify, depatchify

    def safe_forward_ckpt(self, x):
        cfg = self.cfg
        c_out = cfg.transformer.dim_channel
        dim_token = cfg.transformer.dim_token
        p_i = cfg.patch_num_in
        p_e = cfg.patch_num_out
        p_res = cfg.patch_resolution
        demo_num = cfg.demo_num

        init, end, c_mask = x
        bs, _, _, _, _ = init.shape
        feature_init = init.view(-1, *init.shape[-3:])
        feature_end = end.view(-1, *end.shape[-3:])

        feature_init = patchify(feature_init, patch_num=p_i)
        feature_end = patchify(feature_end, patch_num=p_e)
        feature = torch.cat((feature_init, feature_end), dim=1)
        feature = self.pre_proj(feature)

        if self.cfg.use_patch_pos_encoding:
            part_init = feature[:, : p_i * p_i] + self.patch_pos_encoding_init
            part_end  = feature[:, p_i * p_i :]  + self.patch_pos_encoding_end
            feature = torch.cat([part_init, part_end], dim=1)

        feature = feature.view(bs, -1, p_i * p_i + p_e * p_e, dim_token)

        if self.cfg.use_func_pos_encoding:
            part_init = feature[:, :, : p_i * p_i] + self.func_pos_encoding_init.view(1, -1, 1, dim_token)
            part_end  = feature[:, :, p_i * p_i :]  + self.func_pos_encoding_end.view(1, -1, 1, dim_token)
            feature = torch.cat([part_init, part_end], dim=2)

        feature = feature.view(bs, -1, dim_token)

        mask = self.mask
        for layer in self.transformer.layers:
            feature = checkpoint(
                layer, feature, mask,
                None,
                use_reentrant=False,
            )
        if self.transformer.norm is not None:
            feature = self.transformer.norm(feature)

        feature = feature.view(bs, demo_num, -1, dim_token)
        feature = feature[:, :, : p_i * p_i, :]

        feature = self.post_proj(feature)
        feature = feature.view(bs * demo_num, p_i, p_i, -1)
        feature = feature.view(bs * demo_num, p_e, p_i // p_e, p_e, p_i // p_e, -1)
        feature = feature.permute(0, 1, 3, 2, 4, 5)
        feature = feature.reshape(bs * demo_num, p_e * p_e, -1)
        feature = depatchify(feature, patch_num=p_e, c=c_out, h=p_res, w=p_res)
        feature = feature.view(bs, demo_num, *feature.shape[-3:])

        return feature

    return safe_forward_ckpt


@contextmanager
def _patched_forward(model, use_gradient_checkpointing=False):
    raw_model = model.module if hasattr(model, "module") else model
    cls_name = type(raw_model).__name__

    original_forward = None
    if cls_name == "ICON_CROPPED":
        original_forward = type(raw_model).forward
        if use_gradient_checkpointing:
            type(raw_model).forward = _make_safe_forward_checkpointed(raw_model)
            print(f"[Fisher] Patched {cls_name}.forward (non-inplace + gradient checkpointing)")
        else:
            type(raw_model).forward = _make_safe_forward(raw_model)
            print(f"[Fisher] Patched {cls_name}.forward (non-inplace)")

    inplace_states = {}
    for name, m in model.named_modules():
        if hasattr(m, 'inplace') and m.inplace:
            inplace_states[name] = True
            m.inplace = False

    try:
        yield
    finally:
        if original_forward is not None:
            type(raw_model).forward = original_forward
            print(f"[Fisher] Restored original {cls_name}.forward")
        for name, m in model.named_modules():
            if name in inplace_states:
                m.inplace = True

