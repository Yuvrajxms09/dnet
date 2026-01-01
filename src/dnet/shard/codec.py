import numpy as np
import mlx.core as mx
from typing import Optional
from dnet.utils.serialization import dtype_map, mlx_dtype_map
from dnet.core.tensor import to_bytes
from dnet.compression import decompress_tensor_from_protobuf_data
from dnet.core.types.messages import ActivationMessage
from dnet.protos.dnet_ring_pb2 import ActivationRequest
from dnet.utils.logger import logger


class ActivationCodec:
    """
    Handles the translation between Network Protobufs (ActivationRequest)
    and Runtime Objects (ActivationMessage + Shared Memory Pools).
    """

    def __init__(self, runtime):
        self.runtime = runtime

    def deserialize(self, request: ActivationRequest) -> Optional[ActivationMessage]:
        """
        Parses an incoming Proto, allocates memory in the pool,
        decompresses data, and returns a ready-to-compute ActivationMessage.
        """
        if self.runtime.input_pool is None:
            logger.error("Shard %s: input pool not initialized", self.runtime.shard_id)
            return None

        activation = request.activation
        pool_id = None

        # Communication budget logging: track bytes received
        try:
            from dnet.core.observability import load_settings
            obs_settings = load_settings()

            if obs_settings.enabled:
                bytes_received = len(activation.data) if activation.data else 0
                activation_size_mb = bytes_received / (1024 * 1024)

                # Estimate tokens in this activation using actual model hidden size and wire dtype
                try:
                    hidden_size = self.runtime.model_metadata.embedding_size() if self.runtime.model_metadata else 8192
                    wire_dtype_size = self.runtime._wire_mx_dtype.size
                except:
                    hidden_size = 8192  # fallback for large models (e.g., 70B)
                    wire_dtype_size = 2  # fallback to fp16

                estimated_tokens = max(1, int(bytes_received / (wire_dtype_size * hidden_size)))
                bytes_per_token = bytes_received / estimated_tokens if estimated_tokens > 0 else 0

                logger.info(
                    "[COMM_BUDGET] stage=%s direction=inbound nonce=%s "
                    "bytes=%d activation_mb=%.2f bytes_per_token=%.1f layer=%s dtype=%s",
                    self.runtime.shard_id, request.nonce, bytes_received, activation_size_mb,
                    bytes_per_token, activation.layer_id, activation.dtype
                )
        except Exception:
            # Silent failure for comm logging
            pass

        try:
            # Compressed Tensors
            if "|" in activation.dtype:
                deq = decompress_tensor_from_protobuf_data(
                    tensor_data=activation.data,
                    shape=list(activation.shape),
                    dtype_with_metadata=activation.dtype,
                )
                pool_id = self.runtime.input_pool.allocate_for_layer(
                    layer_id=activation.layer_id,
                    dtype=deq.dtype,
                    shape=tuple(deq.shape),
                )
                if pool_id is not None:
                    buffer = self.runtime.input_pool.get_buffer(pool_id)
                    flat = deq.reshape(-1)
                    buffer[: flat.size] = flat
                    msg = ActivationMessage.from_proto(request, pool_id)
                    msg.dtype = str(deq.dtype)
                    msg.shape = tuple(deq.shape)
                    return msg

            # Tokens (Integer sequences)
            elif activation.dtype == "tokens":
                tokens = np.frombuffer(activation.data, dtype=np.int32)
                shp = (int(len(tokens)),)
                pool_id = self.runtime.input_pool.allocate_for_layer(
                    layer_id=activation.layer_id, dtype=mx.int32, shape=shp
                )
                if pool_id is not None:
                    buffer = self.runtime.input_pool.get_buffer(pool_id)
                    buffer[: len(tokens)] = tokens
                    msg = ActivationMessage.from_proto(request, pool_id)
                    msg.dtype = "tokens"
                    msg.shape = shp
                    return msg

            # Standard Raw Tensors
            else:
                # Validation
                expected = (
                    int(np.prod(activation.shape))
                    * np.dtype(dtype_map[activation.dtype]).itemsize
                )
                actual = len(activation.data)
                if expected != actual:
                    logger.error(
                        "Payload mismatch nonce=%s: exp=%d act=%d",
                        request.nonce,
                        expected,
                        actual,
                    )
                    return None

                pool_id = self.runtime.input_pool.allocate_for_layer(
                    layer_id=activation.layer_id,
                    dtype=mlx_dtype_map[activation.dtype],
                    shape=tuple(activation.shape),
                )
                if pool_id is not None:
                    buffer = self.runtime.input_pool.get_buffer(pool_id)
                    input_data = np.frombuffer(
                        activation.data, dtype=dtype_map[activation.dtype]
                    )
                    buffer[: len(input_data)] = input_data
                    return ActivationMessage.from_proto(request, pool_id)

        except Exception as e:
            logger.error(f"Deserialization error for nonce {request.nonce}: {e}")
            # Cleanup if allocation happened but fill failed
            if pool_id is not None:
                self.runtime.input_pool.release(pool_id)
            return None

        return None

    def serialize(self, msg: ActivationMessage, transport_config) -> bytes:
        """
        Reads from output pool/tensor, compresses, and returns bytes + wire dtype.
        """
        shaped = msg.tensor
        if shaped is None:
            if self.runtime.output_pool is None:
                raise ValueError("No output pool and no tensor to serialize")
            output_buffer = self.runtime.output_pool.get_buffer(msg.pool_id)
            data_size = int(np.prod(msg.shape))
            shaped = output_buffer[:data_size].reshape(msg.shape)

        data = to_bytes(
            shaped,
            wire_dtype_str=self.runtime._wire_dtype_str,
            wire_mx_dtype=self.runtime._wire_mx_dtype,
            compress=transport_config.compress,
            compress_min_bytes=transport_config.compress_min_bytes,
        )

        # Clean up reference immediately to assist GC
        msg.tensor = None
        return data
