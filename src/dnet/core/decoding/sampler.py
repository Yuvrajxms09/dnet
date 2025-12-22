import mlx.core as mx
import numpy as np
from typing import Optional, Any, Tuple, Dict
from mlx_lm.sample_utils import make_sampler
from dnet.core.types.messages import TokenResult
from dnet.core.decoding.config import DecodingConfig
from dnet.utils.logger import logger


class GrammarState:
    """Holds LLGuidance grammar state for a single generation session.
    
    Uses LLGuidance's dynamic mask computation approach for constrained JSON generation.
    Replaces the previous Outlines implementation to avoid memory allocation issues.
    """
    
    def __init__(self, matcher, bitmask_allocator, vocab_size: int, eos_token_id: Optional[int] = None):
        """Initialize grammar state with LLGuidance LLMatcher.
        
        Args:
            matcher: LLGuidance LLMatcher instance for tracking grammar state
            bitmask_allocator: Function to allocate bitmask (batch_size, vocab_size)
            vocab_size: Size of the model vocabulary
            eos_token_id: EOS token ID for forced termination
        """
        self.matcher = matcher
        self.bitmask_allocator = bitmask_allocator
        self.vocab_size = vocab_size
        self._eos_token_id = eos_token_id
        self._bitmask = None
        self._terminated = False  # Track termination state - once True, always True
    
    def get_bitmask(self):
        """Get or create the token bitmask.
        
        Returns None if allocation fails (memory insufficient).
        LLGuidance uses packed bitmasks (much smaller than full vocab arrays).
        """
        if self._bitmask is None:
            # LLGuidance bitmask is packed: (batch_size, (vocab_size + 31) // 32)
            # This is much smaller than full vocab arrays
            batch_size = 1
            estimated_size_bytes = ((self.vocab_size + 31) // 32) * 4  # int32 = 4 bytes
            estimated_size_mb = estimated_size_bytes / (1024 ** 2)
            logger.debug(
                f"[MEMORY TRACE] Allocating LLGuidance bitmask: "
                f"vocab_size={self.vocab_size}, estimated_size={estimated_size_mb:.2f}MB"
            )
            
            try:
                logger.debug(f"[MEMORY TRACE] Calling bitmask_allocator(batch_size={batch_size}, vocab_size={self.vocab_size})")
                self._bitmask = self.bitmask_allocator(batch_size, self.vocab_size)
                # Get actual size if possible
                if hasattr(self._bitmask, 'nbytes'):
                    actual_size_mb = self._bitmask.nbytes / (1024 ** 2)
                    logger.debug(
                        f"[MEMORY TRACE] Bitmask allocated successfully: "
                        f"actual_size={actual_size_mb:.2f}MB, shape={getattr(self._bitmask, 'shape', 'unknown')}"
                    )
                else:
                    logger.debug(f"[MEMORY TRACE] Bitmask allocated successfully (size unknown)")
            except (MemoryError, RuntimeError) as e:
                error_msg = str(e)
                logger.error(
                    f"[MEMORY TRACE] ❌ Bitmask allocation FAILED: "
                    f"vocab_size={self.vocab_size}, error={error_msg}, "
                    f"estimated_size={estimated_size_mb:.2f}MB"
                )
                if "allocate" in error_msg.lower() or "memory" in error_msg.lower() or "out of memory" in error_msg.lower():
                    logger.error(
                        f"Failed to allocate bitmask for grammar state (vocab_size={self.vocab_size}): {e}. "
                        f"This usually means insufficient memory for grammar-constrained generation. "
                        f"Consider using tool_choice='auto' instead of 'required'."
                    )
                    # Mark as terminated to prevent further attempts
                    self._terminated = True
                    return None
                else:
                    # Re-raise if it's a different error
                    raise
        else:
            logger.debug(f"[MEMORY TRACE] Reusing existing bitmask (already allocated)")
        return self._bitmask
    
    def fill_next_token_bitmask(self):
        """Fill bitmask with allowed tokens for current state.
        
        Returns None if already terminated or if bitmask allocation failed.
        LLGuidance computes masks dynamically - no pre-allocation of large arrays.
        """
        # Don't fill bitmask if already terminated
        if self._terminated:
            return None
        
        # Check for errors in matcher
        if self.matcher.is_error():
            error_msg = self.matcher.get_error()
            logger.error(f"[MEMORY TRACE] ❌ LLGuidance matcher in error state: {error_msg}")
            self._terminated = True
            return None
        
        from llguidance.mlx import fill_next_token_bitmask
        bitmask = self.get_bitmask()
        # If bitmask allocation failed, get_bitmask() returns None
        if bitmask is None:
            return None
        
        # LLGuidance fills bitmask in-place (index=0 for single batch)
        # fill_next_token_bitmask expects numpy array (NDArray[np.int32])
        # allocate_token_bitmask returns numpy array, so this should work
        fill_next_token_bitmask(self.matcher, bitmask, index=0)
        return bitmask
    
    def accept_token(self, token_id: int) -> None:
        """Accept a token and advance the grammar state.
        
        IMPORTANT: Do NOT advance if matcher is already stopped or terminated.
        This prevents the matcher from continuing after JSON completion.
        """
        # Never advance if we've already been terminated
        if self._terminated:
            return
        
        # Check for errors
        if self.matcher.is_error():
            error_msg = self.matcher.get_error()
            logger.warning(f"LLGuidance matcher in error state: {error_msg}")
            self._terminated = True
            return
        
        # Only advance if NOT stopped - once stopped, we should stop
        if not self.matcher.is_stopped():
            success = self.matcher.consume_token(token_id)
            if not success:
                logger.warning(f"Failed to consume token {token_id} in grammar matcher")
                # Mark as terminated if token consumption fails
                self._terminated = True
        else:
            # Matcher is stopped - mark as terminated to prevent further advancement
            self._terminated = True
    
    def is_terminated(self) -> bool:
        """Check if the grammar has reached a final/accepting state.
        
        For JSON schemas, this should return True when we've generated
        a complete valid JSON object and are in a final accepting state.
        
        Important: This should be checked BEFORE accepting the next token
        to prevent generating beyond the valid JSON structure.
        
        Once terminated, always returns True to prevent duplication.
        """
        # If we've already been terminated, always return True
        # This prevents the matcher from resetting/continuing after completion
        if self._terminated:
            return True
        
        # Check for errors
        if self.matcher.is_error():
            self._terminated = True
            return True
        
        # LLGuidance: check if matcher is in accepting state (can terminate) or stopped
        # is_accepting() = can terminate now (complete valid output)
        # is_stopped() = won't accept more tokens (except EOS)
        if self.matcher.is_accepting() or self.matcher.is_stopped():
            # Mark as terminated to prevent further generation
            self._terminated = True
            logger.debug(
                f"Grammar terminated: is_accepting={self.matcher.is_accepting()}, "
                f"is_stopped={self.matcher.is_stopped()}"
            )
            return True
        
        return False


class Sampler:
    """
    Handles the transformation of logits into tokens based on a DecodingConfig.
    Wraps mlx_lm's make_sampler for consistent sampling behavior.
    Supports structured output via grammar-constrained generation using LLGuidance.
    """

    # Cache for compiled LLTokenizer to avoid recomputing per request
    # Creating LLTokenizer from HuggingFace tokenizer is expensive (~1s), so we cache it
    _vocabulary_cache: Dict[int, Any] = {}

    def __init__(self):
        """Initialize sampler."""
        pass

    @staticmethod
    def _get_or_create_lltokenizer(tokenizer, vocab_size: int):
        """Get or create LLGuidance LLTokenizer from HuggingFace tokenizer.
        
        Caches tokenizer by tokenizer object to avoid recomputation.
        This is an expensive operation (~1s), so caching is important.
        
        Args:
            tokenizer: HuggingFace tokenizer (must be PreTrainedTokenizerFast)
            vocab_size: Expected vocabulary size (from model logits or tokenizer.vocab_size)
            
        Returns:
            LLTokenizer instance or None if creation fails
        """
        cache_key = id(tokenizer)
        if cache_key in Sampler._vocabulary_cache:
            return Sampler._vocabulary_cache[cache_key]
        
        try:
            import llguidance.hf
            
            # Validate tokenizer type
            from transformers import PreTrainedTokenizerFast
            if not isinstance(tokenizer, PreTrainedTokenizerFast):
                logger.warning(
                    f"Tokenizer is not PreTrainedTokenizerFast (got {type(tokenizer)}). "
                    f"LLGuidance requires fast tokenizers. Attempting to use anyway..."
                )
            
            # Get actual vocab size from tokenizer
            actual_vocab_size = getattr(tokenizer, 'vocab_size', len(tokenizer.get_vocab()))
            
            # Validate vocab_size matches actual tokenizer vocab size
            if vocab_size != actual_vocab_size:
                logger.warning(
                    f"Vocab size mismatch: expected {vocab_size} (from model/logits) "
                    f"but tokenizer has {actual_vocab_size} tokens. "
                    f"Using model vocab_size {vocab_size} for LLGuidance."
                )
            
            # Get EOS token ID
            eos_token_id = getattr(tokenizer, 'eos_token_id', None)
            
            # Create LLTokenizer from HuggingFace tokenizer
            # This serializes the tokenizer and is expensive (~1s), so we cache it
            logger.debug(f"[MEMORY TRACE] Creating LLGuidance tokenizer (this may take ~1s)...")
            ll_tokenizer = llguidance.hf.from_tokenizer(
                tokenizer,
                n_vocab=vocab_size,
                eos_token=eos_token_id,
                slices=llguidance.LLTokenizer.json_slices()  # Optimize for JSON schemas
            )
            Sampler._vocabulary_cache[cache_key] = ll_tokenizer
            
            logger.debug(
                f"Created LLGuidance tokenizer: vocab_size={vocab_size}, "
                f"actual_tokenizer_size={actual_vocab_size}, eos_token_id={eos_token_id}"
            )
            return ll_tokenizer
            
        except Exception as e:
            logger.warning(f"Failed to create LLGuidance tokenizer: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    @staticmethod
    def create_grammar_state(json_schema: str, tokenizer, model_vocab_size: Optional[int] = None) -> Optional[GrammarState]:
        """Create a grammar state for JSON schema constrained generation.
        
        Uses LLGuidance to compile JSON schema into a grammar matcher.
        LLGuidance computes masks dynamically, avoiding large pre-allocations.
        
        Args:
            json_schema: JSON schema string to constrain generation
            tokenizer: HuggingFace tokenizer for the model
            model_vocab_size: Optional vocab size override
            
        Returns:
            GrammarState instance or None if creation fails
        """
        if not json_schema:
            return None
            
        try:
            from llguidance.mlx import LLMatcher, allocate_token_bitmask
            
            # Get vocab_size: prefer model_vocab_size (from logits shape) over tokenizer.vocab_size
            #   - model_vocab_size comes from logits.shape[-1] (most accurate, matches actual model)
            #   - tokenizer.vocab_size is fallback (may differ if model was extended)
            vocab_size = model_vocab_size or getattr(tokenizer, 'vocab_size', None)
            if vocab_size is None:
                logger.warning("Could not determine vocab size for grammar state")
                return None
            
            # LLGuidance uses packed bitmasks - much smaller than full vocab arrays
            # Bitmask size: (batch_size, (vocab_size + 31) // 32) * 4 bytes
            bitmask_size_bytes = ((vocab_size + 31) // 32) * 4
            bitmask_size_mb = bitmask_size_bytes / (1024 ** 2)
            logger.info(
                f"[MEMORY TRACE] Creating LLGuidance grammar state: vocab_size={vocab_size}, "
                f"bitmask_size={bitmask_size_mb:.2f}MB (packed format, no large pre-allocation)"
            )
            
            # Log which source we used for debugging
            if model_vocab_size:
                logger.debug(f"Using model_vocab_size={vocab_size} (from logits shape)")
            else:
                tokenizer_vocab_size = getattr(tokenizer, 'vocab_size', None)
                logger.debug(f"Using tokenizer.vocab_size={vocab_size} (fallback)")
            
            # Get or create LLTokenizer (cached, expensive operation ~1s)
            logger.debug(f"[MEMORY TRACE] Getting/creating LLGuidance tokenizer...")
            ll_tokenizer = Sampler._get_or_create_lltokenizer(tokenizer, vocab_size)
            if ll_tokenizer is None:
                logger.warning("[MEMORY TRACE] ❌ Failed to create LLGuidance tokenizer")
                return None
            logger.debug(f"[MEMORY TRACE] LLTokenizer created/retrieved successfully")
            
            # Create grammar from JSON schema
            logger.debug(f"[MEMORY TRACE] Creating grammar from JSON schema...")
            grammar = LLMatcher.grammar_from_json_schema(json_schema)
            logger.debug(f"[MEMORY TRACE] Grammar created from JSON schema")
            
            # Validate grammar (optional but recommended)
            validation_result = LLMatcher.validate_grammar(grammar, ll_tokenizer)
            if validation_result:
                # Check if it's a warning or error
                if "WARNING:" in validation_result:
                    logger.warning(f"Grammar validation warning: {validation_result}")
                else:
                    logger.error(f"Grammar validation failed: {validation_result}")
                    return None
            
            # Create LLMatcher from tokenizer and grammar
            logger.debug(f"[MEMORY TRACE] Creating LLMatcher...")
            matcher = LLMatcher(ll_tokenizer, grammar, log_level=1)
            logger.debug(f"[MEMORY TRACE] LLMatcher created successfully")
            
            # Check for errors
            if matcher.is_error():
                error_msg = matcher.get_error()
                logger.error(f"[MEMORY TRACE] ❌ LLMatcher creation failed: {error_msg}")
                return None
            
            # Get EOS token ID for forced termination when grammar completes
            eos_token_id = getattr(tokenizer, 'eos_token_id', None)
            
            logger.info(
                f"[MEMORY TRACE] ✅ Successfully created LLGuidance grammar state: "
                f"vocab_size={vocab_size}, eos_token_id={eos_token_id}, "
                f"bitmask will be allocated lazily on first use (packed format, ~{bitmask_size_mb:.2f}MB)"
            )
            return GrammarState(
                matcher=matcher,
                bitmask_allocator=allocate_token_bitmask,
                vocab_size=vocab_size,
                eos_token_id=eos_token_id
            )
            
        except ImportError as e:
            logger.warning(f"LLGuidance not installed or import error: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to create grammar state: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    @staticmethod
    def sample(
        logits: mx.array,
        config: DecodingConfig,
        req_logprobs: bool = False,
        req_top_logprobs: int = 0,
        grammar_state: Optional[GrammarState] = None,
    ) -> TokenResult:
        """
        Sample a token from logits using the provided configuration.
        If grammar_state is provided, applies grammar constraints before sampling.
        
        Uses LLGuidance's dynamic mask computation for constrained generation.
        """
        sampler_fn = make_sampler(
            temp=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            min_p=config.min_p if hasattr(config, "min_p") else 0.0,
            min_tokens_to_keep=config.min_tokens_to_keep
            if hasattr(config, "min_tokens_to_keep")
            else 1,
        )
        ndim = getattr(logits, "ndim", None)
        original_shape = getattr(logits, "shape", "unknown")
        logger.debug(
            f"[MEMORY TRACE] Extracting logits: original_shape={original_shape}, ndim={ndim}"
        )
        
        # Store original shape for error reporting
        _original_logits_shape = original_shape
        
        if ndim == 3:
            # Extract last token from sequence: (batch, seq_len, vocab) -> (vocab,)
            # CRITICAL: Create a proper copy, not a view, to avoid memory allocation issues
            # Views can cause Metal to try allocating for the full tensor
            # Use mx.array() to ensure we get a new array, not a view
            last_token = logits[:, -1, :]  # Shape: (batch, vocab)
            v = mx.array(last_token[0])  # Extract batch[0] and create new array
            # Double-check: ensure v is a proper 1D array with correct size
            if v.ndim != 1:
                v = mx.reshape(v, (-1,))
            logger.debug(
                f"[MEMORY TRACE] Extracted from 3D: v.shape={v.shape}, "
                f"v.nbytes={v.nbytes / (1024**2):.2f}MB, "
                f"expected_vocab_size={logits.shape[-1]}"
            )
            # Verify we got the right size
            if len(v) != logits.shape[-1]:
                logger.error(
                    f"[MEMORY TRACE] ❌ Size mismatch: extracted {len(v)} tokens, "
                    f"expected {logits.shape[-1]}"
                )
        elif ndim == 2:
            v = mx.array(logits[-1])  # Create copy, not view
            if v.ndim != 1:
                v = mx.reshape(v, (-1,))
            logger.debug(
                f"[MEMORY TRACE] Extracted from 2D: v.shape={v.shape}, "
                f"v.nbytes={v.nbytes / (1024**2):.2f}MB"
            )
        else:
            v = mx.array(logits) if not isinstance(logits, mx.array) else logits
            if v.ndim != 1:
                v = mx.reshape(v, (-1,))
            logger.debug(
                f"[MEMORY TRACE] Using as-is: v.shape={v.shape}, "
                f"v.nbytes={v.nbytes / (1024**2):.2f}MB"
            )

        # Check termination BEFORE generating token to prevent extra tokens
        grammar_terminated_before = False
        if grammar_state is not None:
            grammar_terminated_before = grammar_state.is_terminated()
            if grammar_terminated_before:
                logger.info("Grammar already terminated before token generation - preventing further generation")
        
        # Apply grammar-constrained logits processing if available
        if grammar_state is not None and not grammar_terminated_before:
            logger.debug(
                f"[MEMORY TRACE] Applying grammar constraints: "
                f"vocab_size={grammar_state.vocab_size}, "
                f"bitmask_allocated={grammar_state._bitmask is not None}, "
                f"terminated={grammar_state._terminated}"
            )
            try:
                # Note: We skip llguidance.mlx.apply_token_bitmask due to Metal kernel allocation bug
                # Instead, we use LLGuidance's NumPy-based approach directly (see below)
                
                # Fill bitmask with allowed tokens for current grammar state
                # LLGuidance computes masks dynamically - no large pre-allocation
                logger.debug(f"[MEMORY TRACE] Calling fill_next_token_bitmask()...")
                bitmask = grammar_state.fill_next_token_bitmask()
                bitmask_info = "None" if bitmask is None else f"shape={getattr(bitmask, 'shape', 'unknown')}"
                logger.debug(
                    f"[MEMORY TRACE] fill_next_token_bitmask() returned: bitmask={bitmask_info}"
                )
                
                # If bitmask is None, grammar was terminated or allocation failed
                # This can happen if:
                # 1. Grammar reached terminal state (normal termination)
                # 2. Bitmask allocation failed due to insufficient memory (fallback to non-grammar)
                # 3. Matcher is in error state
                if bitmask is None:
                    if grammar_state._terminated:
                        # Normal termination - mask all tokens except EOS
                        grammar_terminated_before = True
                        eos_token_id = getattr(grammar_state, '_eos_token_id', None)
                        if eos_token_id is not None and eos_token_id < len(v):
                            v = mx.full_like(v, float('-inf'))
                            v[eos_token_id] = 0.0
                        else:
                            v = mx.full_like(v, float('-inf'))
                        logger.debug("Grammar terminated during bitmask fill - masking all tokens except EOS")
                    else:
                        # Allocation failed or matcher error - fall back to non-grammar generation
                        logger.warning(
                            "Grammar bitmask allocation failed or matcher error - falling back to non-grammar generation. "
                            "This may result in less reliable tool calling."
                        )
                        # Continue without grammar constraints (model will generate freely)
                        # Don't set grammar_terminated - let it continue as regular generation
                else:
                    # Apply bitmask to logits (sets disallowed tokens to -inf)
                    # We use NumPy-based approach to avoid Metal kernel allocation bug
                    
                    # Log detailed info before applying bitmask
                    v_size_mb = (v.size * v.itemsize) / (1024 ** 2) if hasattr(v, 'size') and hasattr(v, 'itemsize') else 0
                    bitmask_size_mb = bitmask.nbytes / (1024 ** 2) if hasattr(bitmask, 'nbytes') else 0
                    
                    # Verify v is 1D with correct vocab size
                    if v.ndim != 1:
                        logger.error(
                            f"[MEMORY TRACE] ❌ Invalid logits shape: "
                            f"expected 1D (vocab,), got {v.ndim}D with shape {v.shape}"
                        )
                        # Try to flatten
                        if v.ndim > 1:
                            v = mx.reshape(v, (-1,))
                        else:
                            logger.warning(f"[MEMORY TRACE] Cannot reshape logits, skipping grammar constraint")
                            # Continue without grammar - better than crashing
                            pass
                    
                    # Verify bitmask shape matches expected format
                    # LLGuidance bitmask should be (batch, (vocab+31)//32) packed format
                    expected_bitmask_words = (grammar_state.vocab_size + 31) // 32
                    if bitmask.shape[1] != expected_bitmask_words:
                        logger.warning(
                            f"[MEMORY TRACE] ⚠️ Bitmask shape mismatch: "
                            f"expected (1, {expected_bitmask_words}), got {bitmask.shape}"
                        )
                    
                    # CRITICAL: Log everything before converting to NumPy
                    logger.info(
                        f"[MEMORY TRACE] ⚠️ About to apply NumPy-based bitmask: "
                        f"v.shape={v.shape}, v.ndim={v.ndim}, "
                        f"v.nbytes={v.nbytes / (1024**2):.2f}MB, "
                        f"v.dtype={v.dtype if hasattr(v, 'dtype') else 'unknown'}, "
                        f"bitmask.shape={bitmask.shape}, bitmask.nbytes={bitmask.nbytes / (1024**2):.2f}MB, "
                        f"original_logits_shape={_original_logits_shape}"
                    )
                    
                    # CRITICAL CHECK: Verify v is exactly (vocab_size,)
                    expected_v_shape = (grammar_state.vocab_size,)
                    if v.shape != expected_v_shape:
                        logger.error(
                            f"[MEMORY TRACE] ❌❌❌ CRITICAL: v shape mismatch! "
                            f"Expected {expected_v_shape}, got {v.shape}. "
                            f"Falling back to non-grammar generation."
                        )
                        # Don't proceed with wrong shape
                        raise ValueError(
                            f"v shape {v.shape} != expected {expected_v_shape}"
                        )
                    
                    # PRODUCTION FIX: LLGuidance's apply_token_bitmask Metal kernel has a bug where it tries to allocate
                    # ~9.5GB even with correct shapes. We'll use LLGuidance's production-ready NumPy approach directly.
                    # This is based on llguidance.numpy.apply_token_bitmask_inplace_kernel which is used in production.
                    # We skip the Metal kernel entirely since it's known to be buggy with large vocabularies.
                    logger.info(
                        f"[MEMORY TRACE] Using LLGuidance's production-ready NumPy-based bitmask approach "
                        f"(skipping Metal kernel due to known allocation bug)"
                    )
                    import numpy as np
                    
                    # CRITICAL: Convert v (1D) to NumPy FIRST, then reshape, to avoid Metal allocation issues
                    # Converting v_2d (which is a view) to NumPy can trigger Metal to allocate for the full tensor
                    # By converting v (which is already a copy) first, we avoid this issue
                    if hasattr(v, 'numpy'):
                        # Convert 1D array to NumPy first (v is already a copy, so this is safe)
                        logits_1d_np = np.array(v, dtype=np.float32)
                        # Then reshape to 2D: (vocab,) -> (1, vocab)
                        logits_np = logits_1d_np[None, :]  # Add batch dimension
                    elif isinstance(v, np.ndarray):
                        logits_np = v[None, :] if v.ndim == 1 else v.copy()
                    else:
                        logits_np = np.array(v, dtype=np.float32)
                        if logits_np.ndim == 1:
                            logits_np = logits_np[None, :]
                    
                    # Convert bitmask to NumPy (bitmask is small, so this is safe)
                    if hasattr(bitmask, 'numpy'):
                        bitmask_np = np.array(bitmask, dtype=np.int32)  # Convert directly, no copy needed (small)
                    elif isinstance(bitmask, np.ndarray):
                        bitmask_np = bitmask.astype(np.int32)
                    else:
                        bitmask_np = np.array(bitmask, dtype=np.int32)
                    
                    # Ensure correct shapes (same as LLGuidance's apply_token_bitmask_inplace)
                    if logits_np.ndim == 1:
                        logits_np = np.expand_dims(logits_np, axis=0)
                    if bitmask_np.ndim == 1:
                        bitmask_np = np.expand_dims(bitmask_np, axis=0)
                    
                    # Apply LLGuidance's production-ready bitmask algorithm
                    # Based on llguidance.numpy.apply_token_bitmask_inplace_kernel
                    # This expands the packed mask and extracts bits efficiently
                    mask_expanded = np.repeat(bitmask_np, 32, axis=1)  # Expand packed mask: (1, 4748) -> (1, 151936)
                    bit_indices = np.tile(np.arange(32, dtype=np.int32), bitmask_np.shape[1])  # [0,1,2,...,31,0,1,2,...,31,...]
                    bit_masks = (mask_expanded >> bit_indices) & 1  # Extract each bit: (1, 151936) boolean array
                    bit_masks = bit_masks[:, :logits_np.shape[1]]  # Trim to match vocab size exactly
                    
                    # Apply mask: set disallowed tokens to -inf (same as LLGuidance)
                    logits_np[bit_masks == 0] = -np.inf
                    
                    # Convert back to MLX array
                    v = mx.array(logits_np[0] if logits_np.shape[0] == 1 else logits_np)
                    
                    allowed_count = int(bit_masks.sum())
                    logger.info(
                        f"[MEMORY TRACE] ✅ LLGuidance NumPy-based bitmask applied successfully: "
                        f"allowed_tokens={allowed_count}/{logits_np.shape[1]}"
                    )
            except (MemoryError, RuntimeError) as e:
                error_msg = str(e)
                logger.error(
                    f"[MEMORY TRACE] ❌❌❌ Exception in grammar-constrained generation: "
                    f"type={type(e).__name__}, error={error_msg}"
                )
                logger.error(
                    f"[MEMORY TRACE] Error context: grammar_state exists, "
                    f"vocab_size={grammar_state.vocab_size if grammar_state else 'N/A'}, "
                    f"bitmask_allocated={grammar_state._bitmask is not None if grammar_state else 'N/A'}, "
                    f"logits_shape={v.shape if hasattr(v, 'shape') else 'unknown'}"
                )
                if "allocate" in error_msg.lower() or "memory" in error_msg.lower():
                    logger.error(
                        f"Memory error during grammar-constrained generation: {e}. "
                        f"Falling back to non-grammar generation."
                    )
                    # Continue without grammar - model will generate freely
                    # This is better than failing completely
                else:
                    # Re-raise if it's a different error
                    raise
            except Exception as e:
                logger.warning(f"Failed to apply grammar mask: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        if grammar_terminated_before:
            # Grammar is already terminated - only allow EOS token
            # This prevents generating any more content tokens
            eos_token_id = getattr(grammar_state, '_eos_token_id', None)
            if eos_token_id is not None and eos_token_id < len(v):
                # Mask all tokens except EOS to force termination
                v = mx.full_like(v, float('-inf'))
                v[eos_token_id] = 0.0  # Allow EOS token only
                logger.debug(f"Grammar terminated - only allowing EOS token {eos_token_id}")
            else:
                # No EOS token ID - mask all to prevent further generation
                v = mx.full_like(v, float('-inf'))
                logger.debug("Grammar terminated - masked all tokens (no EOS token ID)")

        token_tensor = sampler_fn(v)
        token_id = int(token_tensor.item())
        
        # Log token generation for debugging
        if grammar_state is not None:
            logger.debug(
                f"Generated token_id={token_id}, grammar_terminated_before={grammar_terminated_before}, "
                f"_terminated={getattr(grammar_state, '_terminated', False)}, "
                f"is_accepting={grammar_state.matcher.is_accepting()}, "
                f"is_stopped={grammar_state.matcher.is_stopped()}"
            )
        
        # Update grammar state with accepted token and check termination
        grammar_terminated = grammar_terminated_before  # Use pre-check result
        if grammar_state is not None and not grammar_terminated_before:
            try:
                # Accept the token first
                grammar_state.accept_token(token_id)
                
                # Check if grammar is satisfied (complete valid output)
                # This should return True when we've generated a complete valid JSON
                if grammar_state.is_terminated():
                    grammar_terminated = True
                    logger.info(
                        f"Grammar terminated after token: token_id={token_id}, "
                        f"is_accepting={grammar_state.matcher.is_accepting()}, "
                        f"is_stopped={grammar_state.matcher.is_stopped()}, "
                        f"_terminated={getattr(grammar_state, '_terminated', False)}"
                    )
            except Exception as e:
                logger.warning(f"Failed to accept token in grammar: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        elif grammar_terminated_before:
            # Grammar was already terminated - don't accept more tokens
            logger.info(f"Grammar already terminated, not accepting token_id={token_id} - this should not happen")

        logprob = 0.0
        top_logprobs = {}

        if req_logprobs or req_top_logprobs > 0:
            log_sum_exp = mx.logsumexp(v, axis=-1)
            log_probs = v - log_sum_exp

            if req_logprobs:
                logprob = float(log_probs[token_id].item())

            if req_top_logprobs > 0:
                ti = mx.argsort(v)
                ti_np = np.array(ti.tolist())[::-1][:req_top_logprobs]
                for idx in ti_np:
                    ii = int(idx)
                    top_logprobs[ii] = float(log_probs[ii].item())

        return TokenResult(
            token_id=token_id,
            logprob=logprob,
            top_logprobs=top_logprobs,
            grammar_terminated=grammar_terminated,
        )
