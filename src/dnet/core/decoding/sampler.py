import mlx.core as mx
import numpy as np
from typing import Optional, Any, Tuple, Dict
from mlx_lm.sample_utils import make_sampler
from dnet.core.types.messages import TokenResult
from dnet.core.decoding.config import DecodingConfig
from dnet.utils.logger import logger


class GrammarState:
    """Holds Outlines grammar state for a single generation session.
    
    Uses Outlines' FSM-based approach for constrained JSON generation.
    Replaces the previous xgrammar implementation.
    """
    
    def __init__(self, guide, index, bitmask_allocator, vocab_size: int, eos_token_id: Optional[int] = None):
        """Initialize grammar state with Outlines Guide.
        
        Args:
            guide: Outlines Guide instance for tracking FSM state
            index: Outlines Index for the compiled regex/grammar
            bitmask_allocator: Function to allocate bitmask for vocab size
            vocab_size: Size of the model vocabulary
            eos_token_id: EOS token ID for forced termination
        """
        self.guide = guide
        self.index = index
        self.bitmask_allocator = bitmask_allocator
        self.vocab_size = vocab_size
        self._eos_token_id = eos_token_id
        self._bitmask = None
        self._terminated = False  # Track termination state - once True, always True
    
    def get_bitmask(self):
        """Get or create the token bitmask."""
        if self._bitmask is None:
            self._bitmask = self.bitmask_allocator(self.vocab_size)
        return self._bitmask
    
    def fill_next_token_bitmask(self):
        """Fill bitmask with allowed tokens for current state.
        
        Returns None if already terminated to prevent further token generation.
        """
        # Don't fill bitmask if already terminated
        if self._terminated:
            return None
        
        from outlines_core.kernels.mlx import fill_next_token_bitmask
        bitmask = self.get_bitmask()
        fill_next_token_bitmask(self.guide, bitmask)
        return bitmask
    
    def accept_token(self, token_id: int) -> None:
        """Accept a token and advance the grammar state.
        
        IMPORTANT: Do NOT advance if guide is already finished or terminated, even if it accepts tokens.
        This prevents the guide from restarting/continuing after JSON completion.
        """
        # Never advance if we've already been terminated
        if self._terminated:
            return
        
        # Only advance if NOT finished - once finished, we should stop
        # The accepts_tokens check was allowing continuation after completion
        if not self.guide.is_finished():
            self.guide.advance(token_id=token_id, return_tokens=False)
        else:
            # Guide is finished - mark as terminated to prevent further advancement
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
        # This prevents the guide from resetting/continuing after completion
        if self._terminated:
            return True
        
        # Primary check: is the guide finished?
        if not self.guide.is_finished():
            return False
        
        # When finished, verify we're in a final accepting state of the FSM
        # This ensures we've completed a valid JSON structure
        try:
            current_state = self.guide.get_state()
            is_final = self.index.is_final_state(current_state)
            
            if is_final:
                # We're in a final state - mark as terminated and return True
                # Once terminated, we'll always return True on subsequent checks
                self._terminated = True
                return True
            
            # If guide is finished but not in final state, still mark as terminated
            # This is a safety measure - if the guide says it's finished, we should stop
            # The issue was that we were returning False here, allowing continuation
            self._terminated = True
            logger.debug(
                f"Guide finished but not in final state - marking as terminated anyway. "
                f"state={current_state}, is_final={is_final}"
            )
            return True
        except Exception as e:
            # Fallback: if we can't check final state, trust is_finished()
            # Since guide.is_finished() returned True, mark as terminated
            self._terminated = True
            logger.debug(
                f"Could not verify final state: {e}, using is_finished()={self.guide.is_finished()}, "
                f"marking as terminated"
            )
            return True


class Sampler:
    """
    Handles the transformation of logits into tokens based on a DecodingConfig.
    Wraps mlx_lm's make_sampler for consistent sampling behavior.
    Supports structured output via grammar-constrained generation using Outlines.
    """

    # Cache for compiled vocabulary to avoid recomputing per request
    _vocabulary_cache: Dict[int, Any] = {}

    def __init__(self):
        """Initialize sampler."""
        pass

    @staticmethod
    def _get_or_create_vocabulary(tokenizer, vocab_size: int):
        """Get or create Outlines Vocabulary from tokenizer.
        
        Caches vocabulary by tokenizer to avoid recomputation.
        Validates that vocab_size matches tokenizer's actual vocabulary size.
        
        Args:
            tokenizer: HuggingFace tokenizer
            vocab_size: Expected vocabulary size (from model logits or tokenizer.vocab_size)
        """
        cache_key = id(tokenizer)
        if cache_key in Sampler._vocabulary_cache:
            return Sampler._vocabulary_cache[cache_key]
        
        try:
            from outlines_core import Vocabulary
            
            # Get vocabulary dict from tokenizer
            vocab = tokenizer.get_vocab()
            actual_vocab_size = len(vocab)
            
            # Validate vocab_size matches actual tokenizer vocab size
            # This is important for bitmask allocation - it must match logits shape
            if vocab_size != actual_vocab_size:
                logger.warning(
                    f"Vocab size mismatch: expected {vocab_size} (from model/logits) "
                    f"but tokenizer has {actual_vocab_size} tokens. "
                    f"Using model vocab_size {vocab_size} for bitmask allocation."
                )
            
            eos_token_id = tokenizer.eos_token_id
            eos_token = tokenizer.eos_token or tokenizer.decode([eos_token_id])
            
            # Build formatted vocabulary for Outlines
            # Need to convert token strings to their actual string representation
            formatted_vocab = {}
            for token, token_id in vocab.items():
                try:
                    # Convert token to its string representation
                    # This handles special tokens like spacing tokens
                    token_as_str = tokenizer.convert_tokens_to_string([token])
                    if token_as_str not in formatted_vocab:
                        formatted_vocab[token_as_str] = [token_id]
                    else:
                        formatted_vocab[token_as_str].append(token_id)
                except Exception:
                    # Fallback: use token as-is
                    if token not in formatted_vocab:
                        formatted_vocab[token] = [token_id]
                    else:
                        formatted_vocab[token].append(token_id)
            
            # Remove EOS token from vocab (Outlines handles it separately)
            formatted_vocab.pop(eos_token, None)
            
            vocabulary = Vocabulary(eos_token_id, formatted_vocab)
            Sampler._vocabulary_cache[cache_key] = vocabulary
            
            logger.debug(
                f"Created Outlines vocabulary: {len(formatted_vocab)} entries, "
                f"vocab_size={vocab_size}, actual_tokenizer_size={actual_vocab_size}"
            )
            return vocabulary
            
        except Exception as e:
            logger.warning(f"Failed to create Outlines vocabulary: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    @staticmethod
    def create_grammar_state(json_schema: str, tokenizer, model_vocab_size: Optional[int] = None) -> Optional[GrammarState]:
        """Create a grammar state for JSON schema constrained generation.
        
        Uses Outlines to compile JSON schema into an FSM-based grammar guide.
        
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
            from outlines_core import Index, Guide
            from outlines_core.outlines_core import json_schema as oc_json_schema
            from outlines_core.kernels.mlx import allocate_token_bitmask
            
            # Get vocab_size: prefer model_vocab_size (from logits shape) over tokenizer.vocab_size
            #   - model_vocab_size comes from logits.shape[-1] (most accurate, matches actual model)
            #   - tokenizer.vocab_size is fallback (may differ if model was extended)
            # The vocab_size is critical for bitmask allocation - must match logits shape
            vocab_size = model_vocab_size or getattr(tokenizer, 'vocab_size', None)
            if vocab_size is None:
                logger.warning("Could not determine vocab size for grammar state")
                return None
            
            # Log which source we used for debugging
            if model_vocab_size:
                logger.debug(f"Using model_vocab_size={vocab_size} (from logits shape)")
            else:
                tokenizer_vocab_size = getattr(tokenizer, 'vocab_size', None)
                logger.debug(f"Using tokenizer.vocab_size={vocab_size} (fallback)")
            
            # Build regex pattern from JSON schema
            regex_pattern = oc_json_schema.build_regex_from_schema(json_schema)
            logger.debug(f"Built regex from JSON schema (length: {len(regex_pattern)})")
            
            # Get or create vocabulary
            vocabulary = Sampler._get_or_create_vocabulary(tokenizer, vocab_size)
            if vocabulary is None:
                logger.warning("Failed to create vocabulary for grammar state")
                return None
            
            # Create Index from regex and vocabulary
            index = Index(regex_pattern, vocabulary)
            
            # Create Guide from Index
            guide = Guide(index)
            
            # Get EOS token ID for forced termination when grammar completes
            eos_token_id = getattr(tokenizer, 'eos_token_id', None)
            
            logger.debug("Successfully created Outlines grammar state")
            return GrammarState(
                guide=guide,
                index=index,
                bitmask_allocator=allocate_token_bitmask,
                vocab_size=vocab_size,
                eos_token_id=eos_token_id
            )
            
        except ImportError as e:
            logger.warning(f"Outlines not installed or import error: {e}")
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
        
        Uses Outlines' FSM-based approach for constrained generation.
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
        if ndim == 3:
            v = logits[:, -1, :]
            v = v[0]
        elif ndim == 2:
            v = logits[-1]
        else:
            v = logits

        # Check termination BEFORE generating token to prevent extra tokens
        grammar_terminated_before = False
        if grammar_state is not None:
            grammar_terminated_before = grammar_state.is_terminated()
            if grammar_terminated_before:
                logger.info("Grammar already terminated before token generation - preventing further generation")
        
        # Apply grammar-constrained logits processing if available
        if grammar_state is not None and not grammar_terminated_before:
            try:
                from outlines_core.kernels.mlx import apply_token_bitmask
                
                # Fill bitmask with allowed tokens for current grammar state
                bitmask = grammar_state.fill_next_token_bitmask()
                
                # If bitmask is None, grammar was terminated during bitmask fill
                # This shouldn't happen if we checked is_terminated() first, but be defensive
                if bitmask is None:
                    grammar_terminated_before = True
                    # Mask all tokens except EOS to prevent further generation
                    eos_token_id = getattr(grammar_state, '_eos_token_id', None)
                    if eos_token_id is not None and eos_token_id < len(v):
                        v = mx.full_like(v, float('-inf'))
                        v[eos_token_id] = 0.0
                    else:
                        v = mx.full_like(v, float('-inf'))
                    logger.debug("Grammar terminated during bitmask fill - masking all tokens except EOS")
                else:
                    # Apply bitmask to logits (sets disallowed tokens to -inf)
                    # Outlines MLX kernel expects 2D input [batch, vocab]
                    v_2d = v[None, :] if v.ndim == 1 else v
                    v_masked = apply_token_bitmask(v_2d, bitmask)
                    v = v_masked[0] if v_masked.ndim == 2 else v_masked
                
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
                f"guide.is_finished()={grammar_state.guide.is_finished()}"
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
                    try:
                        current_state = grammar_state.guide.get_state()
                        is_final = grammar_state.index.is_final_state(current_state)
                        logger.info(
                            f"Grammar terminated after token: token_id={token_id}, "
                            f"guide.is_finished()={grammar_state.guide.is_finished()}, "
                            f"is_final_state={is_final}, state={current_state}, "
                            f"_terminated={getattr(grammar_state, '_terminated', False)}"
                        )
                    except Exception:
                        logger.info(
                            f"Grammar terminated after token: token_id={token_id}, "
                            f"guide.is_finished()={grammar_state.guide.is_finished()}, "
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
