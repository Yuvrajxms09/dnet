import numpy as np
import mlx.core as mx
from dnet.utils.logger import logger
from dnet.utils.serialization import dtype_map, tensor_to_bytes


def _quantize_8bit_unpacked(tensor: mx.array, group_size: int = 64):
    """8-bit affine quantization without packing (returns uint8, not packed uint32).
    
    mx.quantize packs values into uint32 which breaks our compression pipeline.
    This produces unpacked uint8 codes with shape matching the input.
    """
    D = tensor.shape[-1]
    R = tensor.size // D
    G = D // group_size
    
    x2d = tensor.reshape(R, D).astype(mx.float32)
    x_grouped = x2d.reshape(R, G, group_size)
    
    mins = x_grouped.min(axis=-1, keepdims=True)
    maxs = x_grouped.max(axis=-1, keepdims=True)
    
    scales = (maxs - mins) / 255.0
    scales = mx.where(scales == 0, mx.ones_like(scales), scales)
    biases = mins
    
    quantized = mx.clip(mx.round((x_grouped - biases) / scales), 0, 255).astype(mx.uint8)
    
    return quantized.reshape(R, D), scales.squeeze(-1), biases.squeeze(-1)


def to_bytes(
    tensor: mx.array | np.ndarray,
    *,
    wire_dtype_str: str,
    wire_mx_dtype: mx.Dtype,
    compress: bool = False,
    compress_min_bytes: int = 65536,
) -> bytes | tuple[bytes, str]:
    """Serialize an MLX/Numpy tensor to bytes with the given wire dtype.

    Args:
        tensor: MLX or NumPy array
        wire_dtype_str: Canonical dtype string (e.g., "float16", "bfloat16")
        wire_mx_dtype: MLX dtype to cast to when `tensor` is MLX
        compress: Whether to compress payload using qsparse8_v1
        compress_min_bytes: Minimum size for compression to kick in

    Returns:
        bytes: Serialized tensor data (uncompressed case - backward compatibility)
        tuple[bytes, str]: (Serialized tensor data, dtype metadata string) when compressed
    """
    from dnet.compression.wire import compress_tensor_to_protobuf_data

    # Cast to desired wire dtype without extra copies when possible
    try:
        wire_np_dtype = dtype_map[wire_dtype_str]
    except Exception:
        wire_np_dtype = np.float16

    if isinstance(tensor, np.ndarray):
        if tensor.dtype != wire_np_dtype:
            tensor = tensor.astype(wire_np_dtype, copy=False)
    else:
        if str(tensor.dtype) != wire_dtype_str:
            tensor = tensor.astype(wire_mx_dtype)

    # Check if we should compress
    tensor_size_bytes = tensor.size * tensor.dtype.size
    should_compress = compress  # E1: Remove threshold so compression always triggers when enabled
    logger.info(f"DEBUG: to_bytes - size: {tensor_size_bytes} bytes, compress: {compress}, should_compress: {should_compress}")

    if should_compress:
        logger.info(f"DEBUG: Compressing tensor with qsparse8_v1 - size: {tensor_size_bytes} bytes, shape: {tensor.shape}")

        try:
            # Use unpacked 8-bit quantization (mx.quantize packs into uint32 which breaks compression)
            D = tensor.shape[-1]
            R = tensor.size // D
            G = D // 64
            
            quantized_2d, scales_2d, biases_2d = _quantize_8bit_unpacked(tensor, group_size=64)

            logger.info(f"DEBUG: Quantized (unpacked) - R={R}, D={D}, G={G}, quantized: {quantized_2d.shape}, scales: {scales_2d.shape}")

            # Prepare quantization parameters
            quant_params = {
                "scales": scales_2d,
                "biases": biases_2d,
                "group_size": 64,
                "bits": 8,
                "mode": "affine"
            }

            # Compress with qsparse8_v1 (90% sparsity)
            compressed_bytes, shape, metadata = compress_tensor_to_protobuf_data(
                quantized_2d,
                compression_percentage=90.0,
                quant=quant_params
            )

            compression_ratio = tensor_size_bytes / len(compressed_bytes)
            logger.info(f"DEBUG: Compression successful - ratio: {compression_ratio:.2f}x, metadata: {metadata}")

            return compressed_bytes, metadata

        except Exception as e:
            logger.warning(f"DEBUG: Compression failed: {e}, falling back to uncompressed")
            should_compress = False

    # Normal uncompressed path - return just bytes for backward compatibility
    if isinstance(tensor, np.ndarray):
        data = tensor.tobytes(order="C")
    else:
        data = tensor_to_bytes(tensor)

    logger.info(f"DEBUG: Using uncompressed tensor - size: {len(data)} bytes, compress={compress}, size_bytes={tensor_size_bytes}")
    return data
