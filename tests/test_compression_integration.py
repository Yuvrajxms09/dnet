#!/usr/bin/env python3
"""Test qsparse8_v1 compression integration end-to-end"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dnet/src'))

import mlx.core as mx
import numpy as np

def test_compression_integration():
    """Test the full compression pipeline: to_bytes -> serialize -> deserialize"""
    print("Testing qsparse8_v1 compression integration...")

    # Create test tensor (activation-like)
    batch_size, seq_len, hidden_size = 1, 1024, 4096
    tensor = mx.random.normal((batch_size, seq_len, hidden_size), dtype=mx.float16)
    print(f"Original tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
    print(f"Original size: {tensor.size * tensor.dtype.size} bytes")

    # Import after path setup
    from dnet.core.tensor import to_bytes
    from dnet.compression.wire import decompress_tensor_from_protobuf_data

    # Test compression in to_bytes()
    print("\n=== Testing to_bytes() with compression ===")
    compressed_data, dtype_str = to_bytes(
        tensor,
        wire_dtype_str="float16",
        wire_mx_dtype=mx.float16,
        compress=True,
        compress_min_bytes=1024  # Small threshold for testing
    )

    print(f"Compressed data size: {len(compressed_data)} bytes")
    print(f"Dtype string: {dtype_str}")
    print(f"Is compressed: {'|' in dtype_str}")

    if '|' in dtype_str:
        compression_ratio = (tensor.size * tensor.dtype.size) / len(compressed_data)
        print(f"✅ Compression successful! Ratio: {compression_ratio:.2f}x")

        # Test decompression
        print("\n=== Testing decompression ===")
        try:
            reconstructed = decompress_tensor_from_protobuf_data(
                compressed_data,
                shape=list(tensor.shape),
                dtype_with_metadata=dtype_str
            )

            print(f"Reconstructed shape: {reconstructed.shape}, dtype: {reconstructed.dtype}")

            # Check accuracy
            mse = mx.mean((tensor - reconstructed) ** 2)
            max_diff = mx.max(mx.abs(tensor - reconstructed))

            print(f"MSE: {mse:.6f}")
            print(f"Max difference: {max_diff:.6f}")

            if mse < 0.01:  # Reasonable threshold for fp16
                print("✅ Round-trip compression successful!")
                return True
            else:
                print("❌ Reconstruction accuracy too low")
                return False

        except Exception as e:
            print(f"❌ Decompression failed: {e}")
            return False
    else:
        print("❌ Compression was not applied")
        return False

if __name__ == "__main__":
    success = test_compression_integration()
    sys.exit(0 if success else 1)
