"""
Compression package exposing a clean, importable API for MLX tensors.

Public API:
- column_sparsify_tensor: zero smallest-norm columns along last dim (GPU accelerated)
- compress_tensor_to_protobuf_data: serialize with true sparse wire formats
- decompress_tensor_from_protobuf_data: deserialize from sparse/dense wire
"""

from .ops import column_sparsify_tensor
from .wire import (
    compress_tensor_to_protobuf_data,
    decompress_tensor_from_protobuf_data,
)

__all__ = [
    "column_sparsify_tensor",
    "compress_tensor_to_protobuf_data",
    "decompress_tensor_from_protobuf_data",
]
