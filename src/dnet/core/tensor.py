import numpy as np
import mlx.core as mx
from dnet.utils.serialization import dtype_map, tensor_to_bytes
from dnet.utils.logger import logger


def to_bytes(
    tensor: mx.array | np.ndarray,
    *,
    wire_dtype_str: str,
    wire_mx_dtype: mx.Dtype,
    compress: bool = False,
    compress_min_bytes: int = 65536,
    compression_percentage: float = 90.0,
) -> tuple[bytes, str]:
    """Serialize an MLX/Numpy tensor to bytes with the given wire dtype.

    Args:
        tensor: MLX or NumPy array
        wire_dtype_str: Canonical dtype string (e.g., "float16", "bfloat16")
        wire_mx_dtype: MLX dtype to cast to when `tensor` is MLX
        compress: Whether to compress payload using sparse compression
        compress_min_bytes: Minimum size for compression to kick in
        compression_percentage: Percentage of columns to keep (higher = more data kept)

    Returns:
        tuple[bytes, str]: (Serialized tensor data, dtype string with metadata)
    """
    # Convert numpy to MLX if needed for compression
    if isinstance(tensor, np.ndarray):
        tensor = mx.array(tensor)

    # Cast to desired wire dtype
    if str(tensor.dtype) != wire_dtype_str:
        tensor = tensor.astype(wire_mx_dtype)

    # Check if we should compress
    tensor_bytes = tensor.size * tensor.dtype.size
    should_compress = compress and tensor_bytes >= compress_min_bytes

    if should_compress:
        try:
            from dnet.compression.wire import compress_tensor_to_protobuf_data

            # For qsparse8_v1: enable quantization when wire_dtype is fp16
            quant_dict = None
            if wire_dtype_str == "fp16":
                # Enable qsparse8_v1 quantization
                try:
                    quantized = mx.quantize(tensor, group_size=64, bits=8)
                    quant_dict = {
                        "scales": quantized["scales"],
                        "biases": quantized.get("biases"),
                        "bits": 8,
                        "group_size": 64,
                        "mode": "affine"
                    }
                    # Use quantized data for sparse compression
                    tensor = quantized["data"]
                    logger.debug("[QSPARSE8_V1] Quantized tensor for compression: shape=%s",
                               tensor.shape)
                except Exception as qe:
                    logger.warning("qsparse8_v1 quantization failed, using sparse_v1: %s", qe)

            data, shape, dtype_meta = compress_tensor_to_protobuf_data(
                tensor,
                compression_percentage=compression_percentage,
                quant=quant_dict,
            )
            logger.debug(
                "[COMPRESS] size_before=%d size_after=%d ratio=%.2f dtype=%s quant=%s",
                tensor_bytes, len(data), len(data) / tensor_bytes if tensor_bytes > 0 else 0,
                dtype_meta, "qsparse8_v1" if quant_dict else "sparse_v1"
            )
            return data, dtype_meta
        except Exception as e:
            logger.warning("Compression failed, falling back to raw: %s", e)
            # Fall through to uncompressed path

    # Uncompressed path
    data = tensor_to_bytes(tensor)
    return data, wire_dtype_str
