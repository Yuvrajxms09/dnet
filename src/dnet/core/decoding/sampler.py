import mlx.core as mx
import numpy as np
from typing import Optional, Any
from mlx_lm.sample_utils import make_sampler
from dnet.core.types.messages import TokenResult
from dnet.core.decoding.config import DecodingConfig


class Sampler:
    """
    Handles the transformation of logits into tokens based on a DecodingConfig.
    Wraps mlx_lm's make_sampler for consistent sampling behavior.
    Supports structured output via grammar-constrained generation.
    """

    def __init__(self):
        """Initialize sampler with optional grammar backend."""
        self._grammar_backend = None
        self._logits_processor = None

    def _ensure_grammar_backend(self, model, tokenizer):
        """Lazy initialization of grammar backend if needed."""
        if self._grammar_backend is None and model is not None and tokenizer is not None:
            try:
                from outlines.models import MLXLM
                from outlines.backends.xgrammar import XGrammarBackend

                outlines_model = MLXLM(model, tokenizer)
                self._grammar_backend = XGrammarBackend(outlines_model)
            except ImportError:
                # Outlines not installed, grammar support unavailable
                pass
            except Exception as e:
                # Graceful degradation if grammar setup fails
                import warnings
                warnings.warn(f"Failed to initialize grammar backend: {e}")

    def _get_logits_processor(self, json_schema: Optional[str], model, tokenizer):
        """Get or create logits processor for JSON schema."""
        if not json_schema:
            return None

        try:
            self._ensure_grammar_backend(model, tokenizer)
            if self._grammar_backend is None:
                return None

            # Create new processor for this schema (processors are stateful)
            return self._grammar_backend.get_json_schema_logits_processor(json_schema)
        except Exception as e:
            # Graceful degradation if grammar processing fails
            import warnings
            warnings.warn(f"Failed to create grammar logits processor: {e}")
            return None

    @staticmethod
    def sample(
        logits: mx.array,
        config: DecodingConfig,
        req_logprobs: bool = False,
        req_top_logprobs: int = 0,
        logits_processor: Optional[Any] = None,  # XGrammarLogitsProcessor
        input_ids: Optional[mx.array] = None,  # Full token sequence for grammar state
    ) -> TokenResult:
        """
        Sample a token from logits using the provided configuration.
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

        # Apply grammar-constrained logits processing if available
        if logits_processor is not None and input_ids is not None:
            try:
                # Convert input_ids to format expected by processor (numpy array)
                if isinstance(input_ids, mx.array):
                    input_ids_np = np.array(input_ids.tolist(), dtype=np.int32)
                else:
                    input_ids_np = np.array(input_ids, dtype=np.int32)

                # Convert logits to numpy for processing (xgrammar expects numpy)
                v_np = np.array(v.tolist(), dtype=np.float32)

                # Process logits through grammar
                processed_logits = logits_processor.process_logits(input_ids_np, v_np)

                # Convert back to MLX array
                v = mx.array(processed_logits)
            except Exception:
                # Graceful degradation: if grammar processing fails, use original logits
                pass

        token_tensor = sampler_fn(v)
        token_id = int(token_tensor.item())

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
        )
