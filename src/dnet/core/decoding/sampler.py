import mlx.core as mx
import numpy as np
from mlx_lm.sample_utils import make_sampler
from dnet.core.types.messages import TokenResult
from dnet.core.decoding.config import DecodingConfig


class Sampler:
    """
    Handles the transformation of logits into tokens based on a DecodingConfig.
    Wraps mlx_lm's make_sampler for consistent sampling behavior.
    """

    @staticmethod
    def sample(
        logits: mx.array,
        config: DecodingConfig,
        req_logprobs: bool = False,
        req_top_logprobs: int = 0,
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
