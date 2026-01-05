import numpy as np
import mlx.core as mx
from dnet.utils.serialization import dtype_map, tensor_to_bytes


def to_bytes(
    tensor: mx.array | np.ndarray,
    *,
    wire_dtype_str: str,
    wire_mx_dtype: mx.Dtype,
    compress: bool = False,
    compress_min_bytes: int = 65536,
) -> tuple[bytes, str]:
    """Serialize an MLX/Numpy tensor to bytes with the given wire dtype.

    Args:
        tensor: MLX or NumPy array
        wire_dtype_str: Canonical dtype string (e.g., "float16", "bfloat16")
        wire_mx_dtype: MLX dtype to cast to when `tensor` is MLX
        compress: Whether to compress payload using qsparse8_v1
        compress_min_bytes: Minimum size for compression to kick in

    Returns:
        tuple[bytes, str]: (Serialized tensor data, dtype metadata string)
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
    should_compress = compress and tensor_size_bytes >= compress_min_bytes

    if should_compress:
        print(f"DEBUG: Compressing tensor with qsparse8_v1 - size: {tensor_size_bytes} bytes, shape: {tensor.shape}")

        try:
            # Quantize to 8-bit
            quantized, scales, biases = mx.quantize(tensor, bits=8, group_size=64, mode="affine")

            # Prepare quantization parameters
            quant_params = {
                "scales": scales,
                "biases": biases,
                "group_size": 64,
                "bits": 8,
                "mode": "affine"
            }

            # Compress with qsparse8_v1 (90% sparsity)
            compressed_bytes, shape, metadata = compress_tensor_to_protobuf_data(
                quantized,
                compression_percentage=90.0,
                quant=quant_params
            )

            compression_ratio = tensor_size_bytes / len(compressed_bytes)
            print(f"DEBUG: Compression successful - ratio: {compression_ratio:.2f}x, metadata: {metadata}")

            return compressed_bytes, metadata

        except Exception as e:
            print(f"DEBUG: Compression failed: {e}, falling back to uncompressed")
            should_compress = False

    # Normal uncompressed path
    if isinstance(tensor, np.ndarray):
        data = tensor.tobytes(order="C")
    else:
        data = tensor_to_bytes(tensor)

    print(f"DEBUG: Using uncompressed tensor - size: {len(data)} bytes")
    return data, wire_dtype_str
