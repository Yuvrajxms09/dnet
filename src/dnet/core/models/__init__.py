"""Model implementations for ring topology."""

from typing import Any, List, Optional

from dnet.utils.loader import subclass_where
from .base import BaseRingModel
from .deepseek_v2 import DeepseekV2RingModel
from .deepseek_v3 import DeepseekV3RingModel
# from .deepseek_v32 import DeepseekV32RingModel  # Disabled - requires mlx-lm>=0.30.0
from .llama import LlamaRingModel
from .gpt_oss import GptOssRingModel
from .qwen3 import Qwen3RingModel
from .olmo2 import Olmo2RingModel
from .glm47 import GLM47RingModel
from .minimax_21 import MiniMax21RingModel


def get_ring_model(
    model_type: str,
    model_config: Any,
    assigned_layers: Optional[List[int]] = None,
    is_api_layer: bool = False,
) -> BaseRingModel:
    """Get ring model instance by type.

    Args:
        model_type: Model type identifier
        model_config: Model configuration
        assigned_layers: Assigned layer indices
        is_api_layer: Whether this is an API layer

    Returns:
        Ring model instance
    """
    cls = subclass_where(BaseRingModel, model_type=model_type)
    return cls(
        model_config=model_config,
        assigned_layers=assigned_layers,
        is_api_layer=is_api_layer,
    )


__all__ = [
    "BaseRingModel",
    "DeepseekV2RingModel",
    "DeepseekV3RingModel",
    # "DeepseekV32RingModel",  # Disabled - requires mlx-lm>=0.30.0
    "LlamaRingModel",
    "GptOssRingModel",
    "Qwen3RingModel",
    "Olmo2RingModel",
    "GLM47RingModel",
    "MiniMax21RingModel",
    "get_ring_model",
]
