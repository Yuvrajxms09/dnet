from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.minimax import ModelArgs, MiniMaxDecoderLayer

from .base import BaseRingModel


class MiniMax21RingModel(BaseRingModel):
    """MiniMax-21 model for ring topology inference.

    Supports MoE layers and distributed inference.
    """

    model_type = "minimax_21"

    def __init__(
        self,
        model_config: Any,
        assigned_layers: Optional[List[int]] = None,
        is_api_layer: bool = False,
    ):
        super().__init__()

        if is_api_layer and assigned_layers:
            raise RuntimeError("API layer doesn't handle layers")

        self.model_config = model_config
        self.is_api_layer = is_api_layer
        self.config = config = ModelArgs.from_dict(model_config)

        # Always start dense; let nn.quantize handle conversion if configured
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.layers: List[nn.Module] = []
        self.abs_to_local: Dict[int, int] = {}

        for i, layer in enumerate(sorted(assigned_layers or [])):
            self.layers.append(MiniMaxDecoderLayer(config))
            self.abs_to_local[layer] = i

        # Quantization is handled at bind-time in load_weights
        self._converted_to_quantized = False

        self._cached_mask_state: Optional[int] = None
        self._cached_mask = None

        # Init-time lazy param shrink removed to simplify startup behavior.

    def embed(self, x: mx.array) -> mx.array:
        return self.embed_tokens(x)

    def normalize(self, x: mx.array) -> mx.array:
        return self.norm(x)

    def lm_project(self, x: mx.array) -> mx.array:
        use_tied = bool(getattr(self.config, "tie_word_embeddings", False))
        if use_tied or not hasattr(self, "lm_head"):
            return self.embed_tokens.as_linear(x)
        return self.lm_head(x)

    def forward(self, x: mx.array, cache: Optional[List[Any]] = None) -> mx.array:
        mask = create_attention_mask(x, cache)
        if cache is None:
            cache = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            x = layer(x, mask, cache[i] if i < len(cache) else None)
        return x

    def apply_single_layer(
        self, layer_idx: int, x: mx.array, cache: Optional[List[Any]] = None
    ) -> mx.array:
        if layer_idx not in self.abs_to_local:
            raise RuntimeError(f"Layer {layer_idx} not hosted on this model instance")
        # Create/reuse attention mask when T > 1
        try:
            T = int(x.shape[1]) if len(x.shape) > 1 else 1
        except Exception:
            T = 1
        mask = None
        if T > 1:
            if self._cached_mask_state is None or self._cached_mask_state != T:
                mask = create_attention_mask(x, cache)
                self._cached_mask = mask
                self._cached_mask_state = T
            else:
                mask = self._cached_mask
                if mask is None:
                    mask = create_attention_mask(x, cache)
                    self._cached_mask = mask
                    self._cached_mask_state = T
        local_idx = self.abs_to_local[layer_idx]
        c = None
        if cache is not None and local_idx < len(cache):
            c = cache[local_idx]
        return self.layers[local_idx](x, mask, c)

    # load_weights inherited from BaseRingModel

    def sanitize(self, weights):
        """Dequantize FP8 weights and restructure MoE experts."""

        def dequant(weight, scale_inv):
            dtype = weight.dtype
            bs = 128  # block size
            m, n = weight.shape
            pad_bottom = (-m) % bs
            pad_side = (-n) % bs
            weight = mx.pad(weight, ((0, pad_bottom), (0, pad_side)))
            weight = weight.reshape(
                ((m + pad_bottom) // bs, bs, (n + pad_side) // bs, bs)
            )
            weight = (weight * scale_inv[:, None, :, None]).reshape(
                m + pad_bottom, n + pad_side
            )
            return weight[:m, :n].astype(dtype)

        # Dequantize
        new_weights = {}
        for k, v in weights.items():
            if "weight_scale_inv" in k:
                scale_inv = v
                wk = k.replace("_scale_inv", "")
                weight = weights[wk]
                weight = dequant(weight, scale_inv)
                new_weights[wk] = weight
            elif k not in new_weights:
                new_weights[k] = v
        weights = new_weights

        # Step 2: Handle MoE expert weights restructuring
        if "model.layers.0.block_sparse_moe.experts.0.w1.weight" not in weights:
            return weights

        for l in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{l}"
            mapping = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
            for orig_name, new_name in mapping.items():
                if f"{prefix}.block_sparse_moe.experts.0.{orig_name}.weight" in weights:
                    to_join = [
                        weights.pop(
                            f"{prefix}.block_sparse_moe.experts.{e}.{orig_name}.weight"
                        )
                        for e in range(self.config.num_local_experts)
                    ]
                    weights[
                        f"{prefix}.block_sparse_moe.switch_mlp.{new_name}.weight"
                    ] = mx.stack(to_join)

        # Remove unused precomputed rotary freqs
        weights = {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }

        # Respect tied embeddings
        try:
            if bool(getattr(self.config, "tie_word_embeddings", False)):
                weights.pop("lm_head.weight", None)
        except Exception:
            pass
        return weights

    # Map absolute config keys to local module paths for quantization overrides
    # _abskey_to_local_path inherited from BaseRingModel handles mapping

    @property
    def decoding_layers(self):
        return self.layers

    @property
    def head_dim(self) -> Tuple[int, int]:
        head_dim = (
            self.config.head_dim
            or self.config.hidden_size // self.config.num_attention_heads
        )
        return (head_dim, head_dim)

    @property
    def n_kv_heads(self) -> int:
        return self.config.num_key_value_heads

    @property
    def num_layers(self) -> int:
        return len(self.layers)
