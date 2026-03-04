"""SGLang model for Qwen3.5 (VLM variant of Qwen3Next with separate projections).

Qwen3.5 differs from Qwen3Next in:
  - VLM architecture: weights under model.language_model.* prefix
  - Separate linear-attn projections: in_proj_qkv + in_proj_z (vs combined in_proj_qkvz),
    in_proj_b + in_proj_a (vs combined in_proj_ba)
  - Explicit layer_types list (vs computed from full_attention_interval)
  - Dense (non-MoE) variants exist (Qwen3.5-4B)

Usage: copy or symlink this file into sglang's model directory:
    cp slime_plugins/models/sglang_qwen3_5.py \
       /sgl-workspace/sglang/python/sglang/srt/models/qwen3_5.py

Additionally, patch model_runner.py to recognize Qwen3_5Config as a hybrid GDN model:
    In ModelRunner.hybrid_gdn_config property, add after the isinstance check:
        if getattr(config, 'model_type', None) == 'qwen3_5':
            return config
"""

import logging
import re
from typing import Iterable, Optional, Set, Tuple

import torch
from torch import nn

import sglang.srt.models.qwen3_next as _qn
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.models.qwen3_next import Qwen3NextForCausalLM

logger = logging.getLogger(__name__)

# Qwen3.5 layer_types -> Qwen3Next layers_block_type
_LAYER_TYPE_MAP = {
    "full_attention": "attention",
    "linear_attention": "linear_attention",
}


def _make_dense_layer_class(base_class):
    """Create a layer subclass that uses dense MLP instead of MoE."""

    class _DenseLayer(base_class):
        def __init__(self, config, layer_id, quant_config=None, prefix="", alt_stream=None):
            # Temporarily give the config fake MoE settings so the parent
            # __init__ does not crash (it hardcodes is_layer_sparse=True).
            tp_size = _qn.get_attention_tp_size()
            fake_moe = {
                "num_experts": max(1, tp_size),
                "num_experts_per_tok": 1,
                "moe_intermediate_size": config.intermediate_size,
                "shared_expert_intermediate_size": 0,
                "norm_topk_prob": True,
                "output_router_logits": False,
                "router_aux_loss_coef": 0.0,
            }
            saved = {k: getattr(config, k, None) for k in fake_moe}
            for k, v in fake_moe.items():
                setattr(config, k, v)

            super().__init__(config, layer_id, quant_config, prefix, alt_stream)

            # Restore original config values
            for k, v in saved.items():
                if v is None and hasattr(config, k):
                    delattr(config, k)
                elif v is not None:
                    setattr(config, k, v)

            # Replace MoE MLP with dense MLP
            self.is_layer_sparse = False
            self.mlp = _qn.Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
            )

    _DenseLayer.__name__ = f"Dense{base_class.__name__}"
    _DenseLayer.__qualname__ = _DenseLayer.__name__
    return _DenseLayer


class Qwen3_5ForConditionalGeneration(Qwen3NextForCausalLM):
    """Qwen3.5 VLM model for sglang inference."""

    def __init__(self, config, quant_config=None, prefix: str = ""):
        # --- Patch config ------------------------------------------------
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

        if hasattr(config, "layer_types") and not hasattr(config, "layers_block_type"):
            config.layers_block_type = [
                _LAYER_TYPE_MAP.get(lt, lt) for lt in config.layer_types
            ]

        # GDN attention backend needs full_attention_layer_ids and linear_layer_ids
        if not hasattr(config, "full_attention_layer_ids"):
            config.full_attention_layer_ids = [
                i for i, lt in enumerate(getattr(config, "layer_types", []))
                if lt == "full_attention"
            ]
        if not hasattr(config, "linear_layer_ids"):
            config.linear_layer_ids = [
                i for i, lt in enumerate(getattr(config, "layer_types", []))
                if lt == "linear_attention"
            ]

        # Mamba cache params needed by model_runner for memory allocation
        if not hasattr(config, "mamba2_cache_params"):
            from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
            from sglang.srt.layers.dp_attention import get_attention_tp_size
            shape = Mamba2StateShape.create(
                tp_world_size=get_attention_tp_size(),
                intermediate_size=config.linear_value_head_dim * config.linear_num_value_heads,
                n_groups=config.linear_num_key_heads,
                num_heads=config.linear_num_value_heads,
                head_dim=config.linear_value_head_dim,
                state_size=config.linear_key_head_dim,
                conv_kernel=config.linear_conv_kernel_dim,
            )
            config.mamba2_cache_params = Mamba2CacheParams(
                shape=shape, layers=config.linear_layer_ids
            )

        # --- For dense models, swap layer classes during init ------------
        is_dense = getattr(config, "num_experts", 0) == 0
        if is_dense:
            orig_types = dict(_qn.ALL_DECODER_LAYER_TYPES)
            _qn.ALL_DECODER_LAYER_TYPES["attention"] = _make_dense_layer_class(
                orig_types["attention"]
            )
            _qn.ALL_DECODER_LAYER_TYPES["linear_attention"] = _make_dense_layer_class(
                orig_types["linear_attention"]
            )

        try:
            super().__init__(config, quant_config, prefix)
        finally:
            if is_dense:
                _qn.ALL_DECODER_LAYER_TYPES.update(orig_types)

    # ------------------------------------------------------------------
    @classmethod
    def get_model_config_for_expert_location(cls, config):
        num_experts = getattr(config, "num_experts", 0)
        if not num_experts:
            return None
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=num_experts,
            num_groups=None,
        )

    # ------------------------------------------------------------------
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp=False) -> Set[str]:
        transformed = list(self._transform_weights(weights))
        logger.info(
            "load_weights: yielded %d weights, sample names: %s",
            len(transformed),
            [n for n, _ in transformed[:5]],
        )
        # Check a reference param before/after to detect silent no-ops
        ref_param = None
        for n, p in self.named_parameters():
            if "embed_tokens" in n:
                ref_param = (n, p.data[:3].clone())
                break
        loaded = super().load_weights(iter(transformed), is_mtp=is_mtp)
        logger.info("load_weights: parent loaded %d params", len(loaded) if loaded else -1)
        if ref_param:
            for n, p in self.named_parameters():
                if n == ref_param[0]:
                    diff = (p.data[:3] - ref_param[1]).abs().max().item()
                    logger.info("load_weights: %s max_diff=%.6f", n, diff)
                    break
        # Log any orphaned buffers
        if self._buf_qkv:
            logger.warning("Orphaned _buf_qkv layers: %s", sorted(self._buf_qkv))
        if self._buf_z:
            logger.warning("Orphaned _buf_z layers: %s", sorted(self._buf_z))
        if self._buf_b:
            logger.warning("Orphaned _buf_b layers: %s", sorted(self._buf_b))
        if self._buf_a:
            logger.warning("Orphaned _buf_a layers: %s", sorted(self._buf_a))
        return loaded

    def _transform_weights(self, weights):
        # Use persistent buffers so paired projections that land in different
        # weight-update chunks can still be merged.  On initial load all pairs
        # arrive in a single call, so the buffers drain immediately.
        if not hasattr(self, "_buf_qkv"):
            self._buf_qkv, self._buf_z = {}, {}
            self._buf_b, self._buf_a = {}, {}

        n_in = 0
        n_skip = 0
        n_buf = 0
        n_yield = 0
        n_merge = 0
        sample_names = []

        for name, weight in weights:
            n_in += 1
            if n_in <= 5:
                sample_names.append(f"{name} {tuple(weight.shape)}")

            # Skip visual/VLM-only weights not present in the text model
            if "visual." in name or "multi_modal_projector." in name:
                n_skip += 1
                continue

            name = name.replace("model.language_model.", "model.")

            # Handle tied embeddings: duplicate embed_tokens -> lm_head
            if name == "model.embed_tokens.weight" and getattr(self.config, "tie_word_embeddings", False):
                yield "lm_head.weight", weight

            m = re.search(r"\.layers\.(\d+)\.", name)
            layer = int(m.group(1)) if m else None

            if ".linear_attn.in_proj_qkv." in name:
                self._buf_qkv[layer] = weight
                n_buf += 1
                continue
            if ".linear_attn.in_proj_z." in name:
                self._buf_z[layer] = weight
                n_buf += 1
                continue
            if ".linear_attn.in_proj_b." in name:
                self._buf_b[layer] = weight
                n_buf += 1
                continue
            if ".linear_attn.in_proj_a." in name:
                self._buf_a[layer] = weight
                n_buf += 1
                continue

            n_yield += 1
            yield name, weight

        # Yield merged projections only when both halves are available.
        complete = set(self._buf_qkv) & set(self._buf_z)
        for layer in sorted(complete):
            n_merge += 1
            yield f"model.layers.{layer}.linear_attn.in_proj_qkvz.weight", \
                self._merge_qkvz(self._buf_qkv.pop(layer), self._buf_z.pop(layer))

        complete = set(self._buf_b) & set(self._buf_a)
        for layer in sorted(complete):
            n_merge += 1
            yield f"model.layers.{layer}.linear_attn.in_proj_ba.weight", \
                self._merge_ba(self._buf_b.pop(layer), self._buf_a.pop(layer))

        logger.info(
            "_transform_weights: in=%d skip=%d buf=%d yield=%d merge=%d | samples: %s",
            n_in, n_skip, n_buf, n_yield, n_merge, sample_names,
        )

    def _merge_qkvz(self, qkv_weight, z_weight):
        cfg = self.config
        num_k, num_v = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        dk, dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        v_per_k = num_v // num_k
        # Infer actual group count (handles TP sharding)
        qkv_group = dk + dk + v_per_k * dv
        actual_k = qkv_weight.shape[0] // qkv_group
        q, k, v = torch.split(qkv_weight, [actual_k*dk, actual_k*dk, actual_k*v_per_k*dv], dim=0)
        q = q.view(actual_k, dk, -1)
        k = k.view(actual_k, dk, -1)
        v = v.view(actual_k, v_per_k * dv, -1)
        z = z_weight.view(actual_k, v_per_k * dv, -1)
        return torch.cat([q, k, v, z], dim=1).reshape(-1, qkv_weight.shape[1]).contiguous()

    def _merge_ba(self, b_weight, a_weight):
        cfg = self.config
        num_k = cfg.linear_num_key_heads
        v_per_k = cfg.linear_num_value_heads // num_k
        # Infer actual group count (handles TP sharding)
        actual_k = b_weight.shape[0] // v_per_k
        b = b_weight.view(actual_k, v_per_k, -1)
        a = a_weight.view(actual_k, v_per_k, -1)
        return torch.cat([b, a], dim=1).reshape(-1, b_weight.shape[1]).contiguous()


EntryClass = Qwen3_5ForConditionalGeneration
