from typing import cast
from dnet.core.memory.weight_cache import WeightCache
from ..models import ShardLoadModelRequest
from dnet.core.types.messages import ActivationMessage
from dnet.core.decoding.sampler import Sampler
from dnet.core.decoding.config import DecodingConfig
from dnet.utils.logger import logger
import mlx.core as mx
import numpy as np
from dnet.utils.serialization import mlx_dtype_map
from dnet.utils.time import utc_epoch_now
from .base import register_policy, ComputePolicy
import time
import gc


@register_policy("fit")
class FitInMemoryPolicy(ComputePolicy):
    """Everything fits - no offloading needed"""
    
    # Cache grammar states by nonce to maintain state across token generations
    # TTL-based cleanup prevents memory growth (similar to _kv_by_nonce pattern in runtime.py)
    _grammar_states: dict = {}
    _grammar_states_last_seen: dict = {}  # Track last access time for TTL cleanup
    _grammar_states_ttl_s: float = 300.0  # 5 minutes TTL for grammar states

    @staticmethod
    def _cleanup_grammar_state(nonce: str) -> None:
        """Clean up a single grammar state by nonce.
        
        Production-ready cleanup that:
        1. Clears bitmask to free memory immediately
        2. Removes from cache
        3. Removes from last_seen tracking
        4. Forces GC for large allocations
        """
        logger.info(f"[MEMORY TRACE] Starting cleanup for grammar state nonce={nonce}")
        bitmask_size = None
        if nonce in FitInMemoryPolicy._grammar_states:
            try:
                grammar_state = FitInMemoryPolicy._grammar_states[nonce]
                # Log bitmask info before cleanup
                if hasattr(grammar_state, '_bitmask') and grammar_state._bitmask is not None:
                    if hasattr(grammar_state._bitmask, 'nbytes'):
                        bitmask_size = grammar_state._bitmask.nbytes / (1024 ** 3)  # GB
                    logger.info(
                        f"[MEMORY TRACE] Clearing bitmask: "
                        f"vocab_size={getattr(grammar_state, 'vocab_size', 'unknown')}, "
                        f"bitmask_size={bitmask_size:.2f}GB" if bitmask_size else f"bitmask_size=unknown"
                    )
                    grammar_state._bitmask = None
                else:
                    logger.debug(f"[MEMORY TRACE] No bitmask to clear (was None or doesn't exist)")
                
                logger.debug(f"[MEMORY TRACE] Removing grammar state from cache...")
                del FitInMemoryPolicy._grammar_states[nonce]
            except Exception as e:
                logger.error(
                    f"[MEMORY TRACE] ❌ Error cleaning up grammar state for nonce {nonce}: {e}"
                )
                logger.warning(f"Error cleaning up grammar state for nonce {nonce}: {e}")
            finally:
                # Always remove from tracking dicts
                FitInMemoryPolicy._grammar_states_last_seen.pop(nonce, None)
                logger.debug(f"[MEMORY TRACE] Running garbage collection...")
                # Force GC for large grammar states (bitmasks can be 10GB+)
                gc.collect()
                logger.debug(f"[MEMORY TRACE] Clearing MLX cache...")
                # Clear MLX cache to free memory immediately (bitmasks are MLX arrays)
                mx.clear_cache()
                logger.info(
                    f"[MEMORY TRACE] ✅ Grammar state cleanup completed for nonce={nonce}"
                    + (f", freed ~{bitmask_size:.2f}GB" if bitmask_size else "")
                )
        else:
            logger.debug(f"[MEMORY TRACE] Grammar state not found in cache for nonce={nonce}")
    
    @staticmethod
    def _cleanup_expired_grammar_states() -> None:
        """TTL-based cleanup of expired grammar states.
        
        Removes grammar states that haven't been accessed within TTL window.
        This prevents memory leaks from orphaned states.
        """
        now = time.perf_counter()
        ttl = FitInMemoryPolicy._grammar_states_ttl_s
        expired_nonces = [
            nonce for nonce, last_seen in FitInMemoryPolicy._grammar_states_last_seen.items()
            if (now - last_seen) > ttl
        ]
        
        if expired_nonces:
            logger.debug(f"Cleaning up {len(expired_nonces)} expired grammar states (TTL={ttl}s)")
            for nonce in expired_nonces:
                FitInMemoryPolicy._cleanup_grammar_state(nonce)

    def configure_policy_for_model(self, req: ShardLoadModelRequest) -> None:
        self._mode = "fit"
        local_count = max(1, len(self.runtime.assigned_layers))
        requested_w = max(1, int(req.window_size))
        self.window_size = min(local_count, requested_w)
        self.weight_cache = WeightCache(
            self.runtime.assigned_layers,
            self.runtime.model_metadata,
            window_size=self.window_size,
            prefetch_threads=self.runtime.prefetch_threads,
            resident_windows=self._resident_windows,
            use_mxload_fastpath=self.runtime.compute_config.mxload_fastpath,
            prefetch_mode=self.runtime.compute_config.prefetch_mode,
        )

    def process(self, msg: ActivationMessage) -> None:
        try:
            with self.runtime._model_lock:
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

                # 1) per-nonce KV
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
                    toks = mx.array(
                        np.array(reshaped_data, dtype=np.int32), dtype=mx.int32
                    )
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
                    if to_bind is None:
                        return
                    if to_bind:
                        self.runtime._compute_busy.set()
                        with self.runtime._mlx_lock:
                            self.runtime.model.load_weights(
                                list(to_bind.items()), strict=False
                            )

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
                        # end-shard sampling
                        try:
                            with self.runtime._mlx_lock:
                                y = self.runtime.model.normalize(x_cast)
                                y = self.runtime.model.lm_project(y)

                                grammar_schema = getattr(msg, "grammar_json_schema", None)
                            
                            decoding_config = DecodingConfig(
                                temperature=msg.temperature,
                                top_p=msg.top_p,
                                top_k=msg.top_k,
                                repetition_penalty=msg.repetition_penalty,
                                min_p=msg.min_p,
                                min_tokens_to_keep=msg.min_tokens_to_keep,
                                grammar_json_schema=grammar_schema,
                            )

                            # Get or create grammar state (cached by nonce for multi-token generation)
                            grammar_state = None
                            if grammar_schema:
                                nonce = msg.nonce
                                
                                # Periodic TTL-based cleanup (run every ~100 requests to avoid overhead)
                                import random
                                if random.random() < 0.01:  # 1% chance per request
                                    FitInMemoryPolicy._cleanup_expired_grammar_states()
                                
                                if nonce in FitInMemoryPolicy._grammar_states:
                                    grammar_state = FitInMemoryPolicy._grammar_states[nonce]
                                    # Update last seen time
                                    FitInMemoryPolicy._grammar_states_last_seen[nonce] = time.perf_counter()
                                    # Check if grammar state was already terminated - if so, don't reuse it
                                    if grammar_state is not None and getattr(grammar_state, '_terminated', False):
                                        logger.info(f"Grammar state for nonce {nonce} already terminated, removing from cache - this should not happen!")
                                        FitInMemoryPolicy._cleanup_grammar_state(nonce)
                                        grammar_state = None
                                    else:
                                        logger.debug(f"Reusing grammar state for nonce {nonce}, _terminated={getattr(grammar_state, '_terminated', False) if grammar_state else None}")
                                
                                if grammar_state is None:
                                    logger.debug(f"Creating new grammar state for nonce {nonce}")
                                    tokenizer = getattr(self.runtime, "tokenizer", None)
                                    model_vocab_size = y.shape[-1] if hasattr(y, 'shape') else None
                                    if tokenizer:
                                        grammar_state = Sampler.create_grammar_state(grammar_schema, tokenizer, model_vocab_size)
                                        if grammar_state:
                                            FitInMemoryPolicy._grammar_states[nonce] = grammar_state
                                            FitInMemoryPolicy._grammar_states_last_seen[nonce] = time.perf_counter()
                            
                            logger.debug(
                                f"[MEMORY TRACE] Calling Sampler.sample(): "
                                f"logits_shape={y.shape if hasattr(y, 'shape') else 'unknown'}, "
                                f"grammar_state={'available' if grammar_state else 'None'}, "
                                f"grammar_terminated={getattr(grammar_state, '_terminated', None) if grammar_state else None}"
                            )
                            result = Sampler.sample(
                                logits=y,
                                config=decoding_config,
                                req_logprobs=msg.req_logprobs,
                                req_top_logprobs=msg.req_top_logprobs,
                                grammar_state=grammar_state,
                            )
                            logger.debug(
                                f"[MEMORY TRACE] Sampler.sample() completed: "
                                f"token_id={result.token_id}, "
                                f"grammar_terminated={result.grammar_terminated}"
                            )

                            token_id = result.token_id
                            token_logprob = result.logprob
                            top_logprobs = result.top_logprobs
                            grammar_terminated = result.grammar_terminated
                            
                            # Clean up grammar state from cache if terminated
                            if grammar_terminated and grammar_state is not None and grammar_schema:
                                nonce = msg.nonce
                                logger.info(
                                    f"[MEMORY TRACE] ===== Cleaning up terminated grammar state ====="
                                    f"nonce={nonce}, token_id={token_id}"
                                )
                                logger.debug(f"[MEMORY TRACE] Calling _cleanup_grammar_state()...")
                                FitInMemoryPolicy._cleanup_grammar_state(nonce)
                                logger.debug(f"[MEMORY TRACE] Grammar state cleaned up, clearing MLX cache...")
                                # Clear MLX cache immediately after cleanup to free memory for subsequent operations
                                # This prevents allocation errors when creating output messages
                                mx.clear_cache()
                                logger.info(f"[MEMORY TRACE] ✅ Grammar cleanup and cache clear completed")

                        except MemoryError as e:
                            logger.error(
                                f"[MEMORY TRACE] ❌❌❌ MemoryError caught in end-shard sampling: {e}"
                            )
                            logger.error(
                                f"[MEMORY TRACE] Error context: nonce={msg.nonce}, "
                                f"grammar_state={'exists' if grammar_state else 'None'}, "
                                f"grammar_schema={'exists' if grammar_schema else 'None'}, "
                                f"logits_shape={y.shape if hasattr(y, 'shape') else 'unknown'}"
                            )
                            logger.error(
                                f"End-shard sampling failed due to memory error: {e}. "
                                f"Attempting to clean up grammar state for nonce {msg.nonce}"
                            )
                            # Try to clean up grammar state on memory error
                            FitInMemoryPolicy._cleanup_grammar_state(msg.nonce)
                            self.runtime.input_pool.release(msg.pool_id)
                            return
                        except Exception as e:
                            error_msg = str(e)
                            logger.error(
                                f"[MEMORY TRACE] ❌❌❌ Exception caught in end-shard sampling: "
                                f"type={type(e).__name__}, error={error_msg}"
                            )
                            logger.error(
                                f"[MEMORY TRACE] Error context: nonce={msg.nonce}, "
                                f"grammar_state={'exists' if grammar_state else 'None'}, "
                                f"grammar_schema={'exists' if grammar_schema else 'None'}, "
                                f"logits_shape={y.shape if hasattr(y, 'shape') else 'unknown'}"
                            )
                            # Check if it's a memory allocation error
                            if "allocate" in error_msg.lower() or "memory" in error_msg.lower():
                                logger.error(
                                    f"[MEMORY TRACE] Detected memory-related error, cleaning up grammar state"
                                )
                                logger.error(
                                    f"End-shard sampling failed due to memory issue: {e}. "
                                    f"This may be related to grammar-constrained generation. "
                                    f"Consider using tool_choice='auto' instead of 'required' for large vocabularies."
                                )
                                # Try to clean up grammar state
                                FitInMemoryPolicy._cleanup_grammar_state(msg.nonce)
                            else:
                                logger.error("End-shard sampling failed: %s", e)
                            self.runtime.input_pool.release(msg.pool_id)
                            return

                            logger.debug(
                                f"[MEMORY TRACE] Creating output ActivationMessage: "
                                f"shape={x.shape}, dtype={self.runtime._wire_mx_dtype}"
                            )
                            logger.debug(
                                f"[MEMORY TRACE] Creating output ActivationMessage: "
                                f"shape={x.shape}, dtype={self.runtime._wire_mx_dtype}, "
                                f"token_id={token_id}, grammar_terminated={grammar_terminated}"
                            )
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
                            logprob=token_logprob,
                            top_logprobs=top_logprobs,
                            grammar_terminated=grammar_terminated,
                        )
                        logger.debug(f"[MEMORY TRACE] ✅ Output message created successfully")
                        logger.debug(f"[MEMORY TRACE] Output message created successfully")
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
                            req_logprobs=msg.req_logprobs,
                            req_top_logprobs=msg.req_top_logprobs,
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

    def clear(self):
        try:
            if self.weight_cache:
                self.weight_cache.cancel_all_prefetch()
        except Exception:
            pass

        for layer_id in list(self._bound_versions.keys()):
            try:
                self.weight_cache.evict_layer(layer_id)
            except Exception:
                pass
        try:
            self._bound_versions.clear()
        except Exception:
            self._bound_versions = {}
