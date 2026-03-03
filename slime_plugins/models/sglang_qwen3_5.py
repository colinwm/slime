"""SGLang model for Qwen3.5 (VLM variant of Qwen3Next with separate projections).

Qwen3.5 differs from Qwen3Next in:
  - VLM architecture: weights under model.language_model.* prefix
  - Separate linear-attn projections: in_proj_qkv + in_proj_z (vs combined in_proj_qkvz),
    in_proj_b + in_proj_a (vs combined in_proj_ba)
  - Explicit layer_types list (vs computed from full_attention_interval)

This module adapts the Qwen3Next sglang implementation by overriding weight loading
to merge the separate projections on-the-fly and strip the VLM prefix.

Usage: copy or symlink this file into sglang's model directory:
    cp slime_plugins/models/sglang_qwen3_5.py \
       /sgl-workspace/sglang/python/sglang/srt/models/qwen3_5.py
"""

import logging
import re
from typing import Iterable, Optional, Set, Tuple

import torch
from torch import nn

from sglang.srt.models.qwen3_next import Qwen3NextForCausalLM

logger = logging.getLogger(__name__)

# Qwen3.5 layer_types → Qwen3Next layers_block_type
_LAYER_TYPE_MAP = {
    "full_attention": "attention",
    "linear_attention": "linear_attention",
}


class Qwen3_5ForConditionalGeneration(Qwen3NextForCausalLM):
    """Qwen3.5 VLM model for sglang inference.

    Re-uses the full Qwen3Next forward path; only config patching and
    weight-name translation differ.
    """

    def __init__(
        self,
        config,
        quant_config=None,
        prefix: str = "",
    ):
        # --- Patch config so Qwen3NextForCausalLM.__init__ succeeds --------

        # Dense model: no MoE fields
        for key, default in [
            ("num_experts", 0),
            ("num_experts_per_tok", 0),
            ("decoder_sparse_step", 1),
            ("moe_intermediate_size", 0),
            ("shared_expert_intermediate_size", 0),
            ("mlp_only_layers", []),
            ("norm_topk_prob", True),
            ("output_router_logits", False),
            ("router_aux_loss_coef", 0.0),
        ]:
            if not hasattr(config, key):
                setattr(config, key, default)

        # Convert explicit layer_types to layers_block_type
        if hasattr(config, "layer_types") and not hasattr(config, "layers_block_type"):
            config.layers_block_type = [
                _LAYER_TYPE_MAP.get(lt, lt) for lt in config.layer_types
            ]

        super().__init__(config, quant_config, prefix)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        is_mtp: bool = False,
    ) -> Set[str]:
        return super().load_weights(self._transform_weights(weights), is_mtp=is_mtp)

    def _transform_weights(self, weights):
        """Yield weights with VLM prefix stripped and separate projections merged."""
        pending_qkv = {}  # layer_idx -> tensor
        pending_z = {}
        pending_b = {}
        pending_a = {}

        for name, weight in weights:
            # Strip VLM prefix: model.language_model.X -> model.X
            name = name.replace("model.language_model.", "model.")

            # Buffer the four separate linear-attn projections
            m = re.search(r"\.layers\.(\d+)\.", name)
            layer = int(m.group(1)) if m else None

            if ".linear_attn.in_proj_qkv." in name:
                pending_qkv[layer] = weight
                continue
            if ".linear_attn.in_proj_z." in name:
                pending_z[layer] = weight
                continue
            if ".linear_attn.in_proj_b." in name:
                pending_b[layer] = weight
                continue
            if ".linear_attn.in_proj_a." in name:
                pending_a[layer] = weight
                continue

            yield name, weight

        # Merge separate projections into the interleaved format that
        # Qwen3GatedDeltaNet (from qwen3_next.py) expects.
        for layer in sorted(pending_qkv):
            qkvz = self._merge_qkvz(pending_qkv[layer], pending_z[layer])
            yield f"model.layers.{layer}.linear_attn.in_proj_qkvz.weight", qkvz

        for layer in sorted(pending_b):
            ba = self._merge_ba(pending_b[layer], pending_a[layer])
            yield f"model.layers.{layer}.linear_attn.in_proj_ba.weight", ba

    # ------------------------------------------------------------------
    # Projection weight merging helpers
    # ------------------------------------------------------------------

    def _merge_qkvz(self, qkv_weight: torch.Tensor, z_weight: torch.Tensor):
        """Convert flat [Q,K,V] + [Z] into group-interleaved [Q_g,K_g,V_g,Z_g]."""
        cfg = self.config
        num_k = cfg.linear_num_key_heads
        num_v = cfg.linear_num_value_heads
        dk = cfg.linear_key_head_dim
        dv = cfg.linear_value_head_dim
        v_per_k = num_v // num_k
        key_dim = num_k * dk
        val_dim = num_v * dv

        q, k, v = torch.split(qkv_weight, [key_dim, key_dim, val_dim], dim=0)

        q = q.view(num_k, dk, -1)
        k = k.view(num_k, dk, -1)
        v = v.view(num_k, v_per_k * dv, -1)
        z = z_weight.view(num_k, v_per_k * dv, -1)

        return torch.cat([q, k, v, z], dim=1).reshape(-1, qkv_weight.shape[1]).contiguous()

    def _merge_ba(self, b_weight: torch.Tensor, a_weight: torch.Tensor):
        """Convert flat [B] + [A] into group-interleaved [B_g,A_g]."""
        cfg = self.config
        num_k = cfg.linear_num_key_heads
        v_per_k = cfg.linear_num_value_heads // num_k

        b = b_weight.view(num_k, v_per_k, -1)
        a = a_weight.view(num_k, v_per_k, -1)

        return torch.cat([b, a], dim=1).reshape(-1, b_weight.shape[1]).contiguous()


EntryClass = Qwen3_5ForConditionalGeneration
