from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict, cast
from ...weight_cache import WeightCache
from ..models import ShardLoadModelRequest
from ...data_types import ActivationMessage
from dnet.ring.shard.new_shard.runtime import ShardRuntime
from ....utils.logger import logger
import mlx.core as mx
import numpy as np
from ....utils.serialization import mlx_dtype_map
from ....utils.time import utc_epoch_now

# Policy is the "weights + windowing + pre/post‑compute" brain.
class ComputePolicy(ABC):
    """Abstract compute policy for ShardRuntime"""
    def __init__(self, runtime: ShardRuntime, resident_windows: int):

        self.runtime = runtime
        self.weight_cache: Optional[WeightCache] = None
        # TODO: Maybe rename this to something prefetch related?
        self._prepared_by_nonce: Dict[str, tuple[list[int], Any]] = {}

        self._resident_windows = resident_windows
        self._recent_windows: List[List[int]] = []

        self._defer_unload = True
        self._await_next_ready = False
        self._warmup_keep_flag = False
        self._warmup_completed = False

        self._bound_versions: Dict[int, int] = {}
        self._mode: Optional[str] = None
        self.window_size = 0 # set dynamically

    @abstractmethod
    def process(self, req: ActivationMessage):
        """
        Decide window layers
        waits on _prepared_by_nonce future if present
        calls weight_cache.get_weight / get_resident_layers
        does early _delta_swap_eviction (for sliding‑fit)
        """
        pass

    @abstractmethod
    def configure_policy_for_model(self, req: ShardLoadModelRequest):
        pass

    @staticmethod
    def _next_local_layers(s: List[int], after_layer: int, count: int) -> List[int]:
        if count <= 0:
            return []

        for i, layer in enumerate(s):
            if layer > after_layer:
                return s[i: i + count]
        return []  # No layers found after the specified one

    def _delta_swap_eviction(self, window_layers: List[int], resident: List[int]) -> int:
        budget = max(1, int(self.window_size or 1))
        curr_set = set(window_layers)
        prev_only = [lid for lid in resident if lid not in curr_set]
        keep_quota = max(0, budget - len(window_layers))
        idx = max(0, len(prev_only) - keep_quota)
        evict_head = prev_only[:idx]
        if not evict_head:
            return 0
        evicted: List[int] = []
        for lid in evict_head:
            try:
                if self.weight_cache and self.weight_cache.evict_layer(lid):
                    evicted.append(lid)
            except Exception:
                continue
        if evicted:
            try:
                self.runtime.model.unload_layers(evicted)
                for lid in evicted:
                    self._bound_versions.pop(lid, None)
            except Exception:
                pass
        return len(evicted)

    def _bind_layer_weights(self, window_layers: List[int], msg) -> Optional[Dict[str, mx.array]]:
        """Bind weights for window layers if needed."""
        # Early exit if all weights are already bound and fit in window
        fast_fit = len(self.runtime._assigned_sorted) <= self.window_size
        if fast_fit and all(wl in self._bound_versions for wl in window_layers):
            return {}

        to_bind = {}
        for wl in window_layers:
            if not self.weight_cache:
                logger.error("Weight cache not initialized")
                self.runtime.input_pool.release(msg.pool_id)
                return None
            weights = self.weight_cache.get_weight(wl)
            if weights is None:
                logger.error("Failed to load weights for layer %s", wl)
                self.runtime.input_pool.release(msg.pool_id)
                return None

            # Check if weights need updating
            current_version = self._get_weight_version(weights)
            if self._bound_versions.get(wl) != current_version:
                to_bind.update(weights)
                self._bound_versions[wl] = current_version

        return to_bind

    @staticmethod
    def _get_weight_version(weights: dict) -> int:
        """Get a version identifier"""
        if not weights:
            return -1
        return id(next(iter(weights.values())))

class FitInMemoryPolicy(ComputePolicy):

    """Everything fits - no offloading needed"""
    def configure_policy_for_model(self, req: ShardLoadModelRequest) -> None:
        self._mode = "fit"
        local_count = max(1, len(self.runtime.assigned_layers))
        requested_w = max(1, int(req.window_size))
        self.window_size = min(local_count, requested_w)
        self.weight_cache = WeightCache(
            self.runtime.assigned_layers,
            self.runtime.model_metadata,
            window_size=self.window_size,
            prefetch_threads=self.runtime.compute_config.prefetch_threads,
            resident_windows=self._resident_windows,
            use_mxload_fastpath=self.runtime.compute_config.mxload_fastpath,
            prefetch_mode=self.runtime.compute_config.prefetch_mode
        )

    def process(self, msg: ActivationMessage) -> None:
        if (
                not self.runtime.model
                or not self.runtime.model_metadata
                or not self.weight_cache
                or not self.runtime.input_pool
                or not self.runtime.output_pool
        ):
            logger.error(
                "Runtime %s: cannot process activation - model not loaded",
                self.runtime.shard_id,
            )
            return

        try:
            # 1) per‑nonce KV
            kv = self.runtime.get_or_make_kv(msg.nonce)

            # 2) get input tensor from pool
            input_buffer = self.runtime.input_pool.get_buffer(msg.pool_id)
            if input_buffer is None:
                logger.error("Failed to get input buffer %s", msg.pool_id)
                return

            # 3) prepare x
            input_size = int(np.prod(msg.shape))
            reshaped_data = input_buffer[:input_size].reshape(msg.shape)
            if msg.dtype == "tokens":
                # Token path: convert to int32, embed, and ensure correct dtype
                toks = mx.array(np.array(reshaped_data, dtype=np.int32), dtype=mx.int32)
                x = self.runtime.model.embed(toks[None])
                target_dtype = self.runtime._wire_mx_dtype
            else:
                # Non-token path: use data as-is, convert dtype if needed
                x = reshaped_data
                target_dtype = mlx_dtype_map[msg.dtype]

            if target_dtype and x.dtype != target_dtype:
                x = x.astype(target_dtype)

            current_layer = msg.layer_id + 1
            while True:
                # build contiguous window inside our shard
                window_layers: list[int] = []
                for i in range(self.window_size):
                    layer = current_layer + i
                    if layer not in self.runtime._assigned_set:
                        break
                    window_layers.append(layer)

                to_bind = self._bind_layer_weights(window_layers, msg)
                if to_bind:
                    self.runtime._compute_busy.set()
                    with self.runtime._mlx_lock:
                        self.runtime.model.load_weights(list(to_bind.items()), strict=False)
                else:
                    return

                # compute window
                try:
                    self.runtime._compute_busy.set()
                except Exception:
                    pass
                for lyr in window_layers:
                    with self.runtime._mlx_lock:
                        x = self.runtime.model.apply_single_layer(lyr, x, cache=kv)
                        try:
                            if str(x.dtype) != str(self.runtime._wire_mx_dtype):
                                x = x.astype(self.runtime._wire_mx_dtype)
                        except Exception:
                            pass

                last_layer = window_layers[-1]
                try:
                    mx.eval(x)
                except Exception:
                    pass

                for lid in window_layers:
                    self.weight_cache.decrease_reference(lid)

                # continue if next is still local
                nxt = last_layer + 1
                if nxt in self.runtime._assigned_set:
                    current_layer = nxt
                    continue

                # boundary reached
                x_cast = (
                    x
                    if x.dtype == self.runtime._wire_mx_dtype
                    else x.astype(self.runtime._wire_mx_dtype)
                )

                # build output ActivationMessage
                if nxt >= self.runtime.model_metadata.num_layers:
                    # end‑shard sampling
                    try:
                        with self.runtime._mlx_lock:
                            y = self.runtime.model.normalize(x_cast)
                            y = self.runtime.model.lm_project(y)
                        if y.ndim == 3:
                            logits_2d = y[:, -1, :]
                        elif y.ndim == 2:
                            logits_2d = y[-1:, :]
                        else:
                            logits_2d = y.reshape(1, -1)
                        tok = mx.argmax(logits_2d, axis=-1)
                        token_id = int(tok.item())
                    except Exception as e:
                        logger.error("End‑shard sampling failed: %s", e)
                        self.runtime.input_pool.release(msg.pool_id)
                        return

                    output_msg = ActivationMessage(
                        nonce=msg.nonce,
                        layer_id=last_layer,
                        pool_id=-1,
                        shape=cast(tuple[int, ...], x.shape),
                        batch_size=msg.batch_size,
                        timestamp=utc_epoch_now(),
                        node_origin=f"shard_{self.runtime.shard_id}",
                        dtype=str(self.runtime._wire_mx_dtype),
                        callback_url=msg.callback_url,
                        is_final=True,
                        token_id=token_id,
                    )
                else:
                    output_msg = ActivationMessage(
                        nonce=msg.nonce,
                        layer_id=last_layer,
                        pool_id=-1,
                        shape=cast(tuple[int, ...], x.shape),
                        batch_size=msg.batch_size,
                        timestamp=utc_epoch_now(),
                        node_origin=f"shard_{self.runtime.shard_id}",
                        dtype=str(self.runtime._wire_mx_dtype),
                        callback_url=msg.callback_url,
                        tensor=x_cast,
                    )

                self.runtime.emit_result(output_msg)
                self.runtime.input_pool.release(msg.pool_id)
                return

        except Exception as e:
            logger.exception("Error in fit policy process: %s", e)
            try:
                if self.runtime.input_pool:
                    self.runtime.input_pool.release(msg.pool_id)
            except Exception:
                pass


class OffloadingPolicy(ComputePolicy):
    def configure_policy_for_model(self, req):
        pass
        # look at runtime.compute_config.mode to decide offload vs sliding
        # set self._mode, window_size, resident_windows
        # create WeightCache with small resident_windows, fastpath, etc.

    def process(self, msg):
        pass
        # basically current _process_activation, including:
        # - _prepared_by_nonce waits
        # - sliding_fit branches when self._mode == "sliding_fit"
        # - evict windows aggressively when offload
