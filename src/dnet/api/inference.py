import asyncio
import time
import uuid
import json
import mlx.core as mx
import numpy as np
from typing import Optional, Any, List, Dict
from dnet.core.tensor import to_bytes

from .models import (
    ChatRequestModel,
    ChatResponseModel,
    ChatChoice,
    ChatMessage,
    ChatUsage,
    ChatCompletionReason,
    ChatLogProbs,
)
from .cluster import ClusterManager
from .model_manager import ModelManager
from .strategies.base import ApiAdapterBase
from dnet.core.decoding.config import DecodingConfig
from dnet.utils.logger import logger


async def arange(count: int):
    """Async range generator."""
    for i in range(count):
        yield i


async def azip(*async_iterables):
    """Async zip."""
    iterators = [aiter(it) for it in async_iterables]
    while True:
        try:
            results = await asyncio.gather(*[anext(it) for it in iterators])
            yield results
        except StopAsyncIteration:
            break


class InferenceManager:
    def __init__(
        self,
        cluster_manager: ClusterManager,
        model_manager: ModelManager,
        grpc_port: int,
        adapter: ApiAdapterBase,
    ):
        self.cluster_manager = cluster_manager
        self.model_manager = model_manager
        self.grpc_port = grpc_port
        self.adapter = adapter

        self._api_callback_addr: str = ""

    async def connect_to_ring(
        self, first_shard_ip: str, first_shard_port: int, api_callback_addr: str
    ) -> None:
        """
        `api_callback_addr` must be a reachable `host:port` from shards.
        For internet setups, this should be a public IP/DNS or overlay VPN IP.
        """
        await self.adapter.connect_first_shard(first_shard_ip, first_shard_port)
        self._api_callback_addr = api_callback_addr

    def _format_tools_for_prompt(self, tools: Optional[List[Dict[str, Any]]]) -> str:
        """
        Format tools in OpenAI format for prompt injection.
        
        OpenAI injects tools into the system message in a format models are trained on.
        This creates a similar format that helps models understand available tools.
        """
        if not tools:
            return ""
        
        # Format tools as JSON schema (OpenAI format)
        tools_json = json.dumps(tools, indent=2)
        
        return f"""

You have access to the following tools. When you need to use a tool, respond with a JSON object in this exact format:

{{
  "tool_calls": [
    {{
      "id": "call_<unique_id>",
      "type": "function",
      "function": {{
        "name": "<function_name>",
        "arguments": "<json_string_of_arguments>"
      }}
    }}
  ]
}}

Available tools:
{tools_json}

Important:
- Respond with ONLY the JSON object, no other text
- The "arguments" field must be a JSON string (not an object)
- Use the exact function names from the tools list above
- Include all required parameters
"""
    
    def _detect_tool_calls(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """
        Detect and parse OpenAI-format tool calls from generated text.
        
        Looks for complete JSON object with "tool_calls" key in OpenAI format:
        {
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"location\": \"SF\"}"
                    }
                }
            ]
        }
        
        Returns list of tool calls in OpenAI format, or None if not found.
        """
        if not text:
            return None
        
        # Clean text - remove markdown code blocks if present
        cleaned_text = text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:].strip()
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:].strip()
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3].strip()
        
        # Look for complete JSON object with "tool_calls" key (OpenAI format)
        if '"tool_calls"' in cleaned_text or "'tool_calls'" in cleaned_text:
            # Find the position of tool_calls
            tool_calls_idx = cleaned_text.find('"tool_calls"')
            if tool_calls_idx == -1:
                tool_calls_idx = cleaned_text.find("'tool_calls'")
            
            if tool_calls_idx != -1:
                # Find the opening brace before tool_calls
                start_idx = cleaned_text.rfind('{', 0, tool_calls_idx)
                if start_idx != -1:
                    # Find the matching closing brace
                    brace_count = 0
                    end_idx = -1
                    for i in range(start_idx, len(cleaned_text)):
                        if cleaned_text[i] == '{':
                            brace_count += 1
                        elif cleaned_text[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                    
                    if end_idx != -1:
                        json_str = cleaned_text[start_idx:end_idx]
                        try:
                            parsed = json.loads(json_str)
                            if isinstance(parsed, dict) and "tool_calls" in parsed:
                                tool_calls = parsed["tool_calls"]
                                if isinstance(tool_calls, list) and len(tool_calls) > 0:
                                    # Log raw model output for debugging duplication
                                    logger.debug(f"[TOOL_CALL_DEBUG] Model generated JSON: {json_str[:500]}...")  # First 500 chars
                                    logger.debug(f"[TOOL_CALL_DEBUG] Parsed tool_calls count from model: {len(tool_calls)}")
                                    logger.debug(f"[TOOL_CALL_DEBUG] Tool call IDs from model: {[tc.get('id') for tc in tool_calls]}")
                                    logger.debug(f"[TOOL_CALL_DEBUG] Tool call names from model: {[tc.get('function', {}).get('name') for tc in tool_calls]}")
                                    
                                    # Validate format matches OpenAI spec
                                    valid_calls = []
                                    for call in tool_calls:
                                        if (
                                            isinstance(call, dict)
                                            and "id" in call
                                            and "type" in call
                                            and call.get("type") == "function"
                                            and "function" in call
                                            and isinstance(call["function"], dict)
                                            and "name" in call["function"]
                                            and "arguments" in call["function"]
                                        ):
                                            valid_calls.append(call)
                                    
                                    if valid_calls:
                                        logger.info(f"Detected {len(valid_calls)} tool call(s) in OpenAI format")
                                        logger.debug(f"[TOOL_CALL_DEBUG] Valid tool call IDs after validation: {[tc.get('id') for tc in valid_calls]}")
                                        # Check for duplicates
                                        ids = [tc.get('id') for tc in valid_calls]
                                        if len(ids) != len(set(ids)):
                                            logger.warning(f"[TOOL_CALL_DEBUG] DUPLICATE IDs detected! IDs: {ids}")
                                            # Log full details of duplicates
                                            seen = {}
                                            for i, call in enumerate(valid_calls):
                                                call_id = call.get('id')
                                                if call_id in seen:
                                                    logger.warning(f"[TOOL_CALL_DEBUG] Duplicate at index {i}: ID={call_id}, name={call.get('function', {}).get('name')}, args={call.get('function', {}).get('arguments')[:100]}")
                                                else:
                                                    seen[call_id] = i
                                        return valid_calls
                        except json.JSONDecodeError:
                            pass
        
        return None

    async def generate_stream(self, req: ChatRequestModel):
        """
        Generator for chat completion chunks.
        """
        if not self.model_manager.tokenizer:
            raise RuntimeError(
                "Inference manager not ready (ring not connected or tokenizer not loaded)"
            )

        tokenizer = self.model_manager.tokenizer

        # Prepare messages with tool injection if tools are provided
        messages_for_prompt = req.messages.copy()
        
        # Inject tools into system message or first user message
        if req.tools:
            tools_prompt = self._format_tools_for_prompt(req.tools)
            
            # Try to find system message, otherwise prepend to first user message
            has_system = any(m.role == "system" for m in messages_for_prompt)
            
            if has_system:
                # Append to existing system message
                for i, msg in enumerate(messages_for_prompt):
                    if msg.role == "system":
                        messages_for_prompt[i] = ChatMessage(
                            role="system",
                            content=(msg.content or "") + tools_prompt
                        )
                        break
            else:
                # Prepend system message with tools
                messages_for_prompt.insert(0, ChatMessage(
                    role="system",
                    content=tools_prompt.strip()
                ))

        try:
            if (
                hasattr(tokenizer, "chat_template")
                and tokenizer.chat_template is not None
            ):
                message_dicts = [
                    {
                        "role": m.role,
                        "content": m.content or "",
                    }
                    for m in messages_for_prompt
                ]
                prompt_text = tokenizer.apply_chat_template(
                    message_dicts,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            else:
                prompt_text = (
                    "\n".join(m.content or "" for m in messages_for_prompt) + "\nAssistant:"
                )
        except Exception:
            prompt_text = "\n".join(m.content or "" for m in messages_for_prompt) + "\nAssistant:"

        prompt_tokens = tokenizer.encode(prompt_text)
        prompt_array = mx.array(prompt_tokens)

        stop_id_sequences = []
        if req.stop:
            for stop_word in req.stop:
                stop_id_sequences.append(
                    tokenizer.encode(stop_word, add_special_tokens=False)
                )

        nonce = f"chatcmpl-{uuid.uuid4()}"
        t_start = time.perf_counter()
        t_first_token: Optional[float] = None
        tokens: List[int] = []

        detokenizer = tokenizer.detokenizer
        detokenizer.reset()
        last_text_len = 0

        completion_reason = ChatCompletionReason.LENGTH

        await self.adapter.reset_cache()

        # Yield initial chunk with role
        yield ChatResponseModel(
            id=nonce,
            choices=[
                ChatChoice(
                    index=0,
                    delta=ChatMessage(role="assistant", content=""),
                    finish_reason=None,
                )
            ],
            created=int(time.time()),
            model=req.model,
        )

        y = prompt_array
        for _ in range(req.max_tokens):
            tok_np = (
                y.astype(mx.int32)
                if hasattr(y, "astype")
                else np.array(list(map(int, y)), dtype=np.int32)
            )
            tok_bytes = to_bytes(
                tok_np,
                wire_dtype_str="int32",
                wire_mx_dtype=mx.int32,
            )

            decoding_config = DecodingConfig(
                temperature=req.temperature,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
                min_p=req.min_p if hasattr(req, "min_p") else 0.0,
                min_tokens_to_keep=req.min_tokens_to_keep
                if hasattr(req, "min_tokens_to_keep")
                else 1,
            )

            # Send tokens to first shard
            await self.adapter.send_tokens(
                tokens=tok_bytes,
                nonce=nonce,
                callback_addr=self._api_callback_addr,
                logprobs=req.logprobs if req.logprobs else False,
                top_logprobs=req.top_logprobs if req.top_logprobs else 0,
                decoding_config=decoding_config,
            )
            result = await self.adapter.await_token(nonce, timeout_s=300.0)
            token = int(result.token_id)

            # Accumulate logprobs
            token_logprobs = []
            top_logprobs_list = []
            if req.logprobs:
                token_logprobs.append(result.logprob)
                top_logprobs_list.append(result.top_logprobs)

            if t_first_token is None:
                t_first_token = time.perf_counter()

            detokenizer.add_token(token)
            tokens.append(token)

            # mlx_lm detokenizer usually updates .text property
            full_text = detokenizer.text
            delta_text = full_text[last_text_len:]
            last_text_len = len(full_text)

            # Check for tool calls in the generated text (only if tools are provided)
            detected_tool_calls = None
            if req.tools:
                detected_tool_calls = self._detect_tool_calls(full_text)

            # Yield chunk
            # If tool calls detected, include them in the delta
            delta_message = ChatMessage(role="assistant", content=delta_text)
            if detected_tool_calls:
                # When tool calls are present, content should be None or empty
                delta_message.content = None
                delta_message.tool_calls = detected_tool_calls
            
            yield ChatResponseModel(
                id=nonce,
                choices=[
                    ChatChoice(
                        index=0,
                        delta=delta_message,
                        logprobs=ChatLogProbs(
                            token_logprobs=token_logprobs,
                            top_logprobs=top_logprobs_list,
                            tokens=[token],
                        )
                        if req.logprobs
                        else None,
                        finish_reason=None,
                    )
                ],
                created=int(time.time()),
                model=req.model,
            )

            # If tool calls detected, stop generation and set finish_reason
            if detected_tool_calls:
                completion_reason = ChatCompletionReason.TOOL_CALLS
                break

            # stopping criteria
            if token == tokenizer.eos_token_id:
                completion_reason = ChatCompletionReason.STOP
                break
            y = mx.array([token], dtype=mx.int32)

        detokenizer.finalize()
        
        # Final check for tool calls in complete text
        final_text = detokenizer.text
        final_tool_calls = None
        if req.tools and completion_reason != ChatCompletionReason.TOOL_CALLS:
            final_tool_calls = self._detect_tool_calls(final_text)
            if final_tool_calls:
                completion_reason = ChatCompletionReason.TOOL_CALLS

        metrics_dict = None
        t_end = time.perf_counter()
        if getattr(req, "profile", False):
            total_s = max(t_end - t_start, 1e-9)
            gen_s = max((t_end - (t_first_token or t_start)), 1e-9)
            tokens_generated = len(tokens)
            metrics_dict = {
                "total_ms": round(total_s * 1000.0, 3),
                "ttfb_ms": round(((t_first_token or t_end) - t_start) * 1000.0, 3),
                "token_gen_ms": round(gen_s * 1000.0, 3),
                "tokens_generated": tokens_generated,
                "tps_overall": round(
                    (tokens_generated / total_s) if tokens_generated else 0.0, 4
                ),
                "tps_decoding": round(
                    (tokens_generated / gen_s) if tokens_generated else 0.0, 4
                ),
            }

        # Final chunk with finish reason and tool calls
        # IMPORTANT: If we already yielded tool_calls in a delta chunk, don't include them again
        # in the final chunk's delta to avoid duplication in chat_completions accumulation
        final_message = ChatMessage(role="assistant", content="")
        if completion_reason == ChatCompletionReason.TOOL_CALLS:
            # For tool calls, content should be None
            final_message.content = None
            # Only set tool_calls in final message if we haven't already yielded them in delta
            # (detected_tool_calls means we already yielded them, so don't duplicate)
            if not detected_tool_calls:
                final_message.tool_calls = final_tool_calls
            # If detected_tool_calls exists, we already yielded it in delta, so leave it None here
        else:
            final_message.content = final_text

        yield ChatResponseModel(
            id=nonce,
            choices=[
                ChatChoice(
                    index=0,
                    delta=final_message,
                    finish_reason=completion_reason,
                )
            ],
            created=int(time.time()),
            model=req.model,
            metrics=metrics_dict,
            usage=ChatUsage(
                prompt_tokens=len(prompt_tokens),
                completion_tokens=len(tokens),
                total_tokens=len(prompt_tokens) + len(tokens),
            ),
        )

    async def chat_completions(self, req: ChatRequestModel) -> ChatResponseModel:
        """
        Handles chat completion request (non-streaming).
        """
        full_content = ""
        tokens = []
        token_logprobs = []
        top_logprobs_list = []
        completion_reason = ChatCompletionReason.LENGTH
        nonce = ""
        metrics_dict = None
        usage = None
        tool_calls = None

        async for chunk in self.generate_stream(req):
            nonce = chunk.id
            choice = chunk.choices[0]
            
            # Accumulate content (may be None if tool_calls present)
            if choice.delta:
                if choice.delta.content:
                    full_content += choice.delta.content
                # Collect tool_calls from delta
                if choice.delta.tool_calls:
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.extend(choice.delta.tool_calls)

            if choice.logprobs:
                if choice.logprobs.token_logprobs:
                    token_logprobs.extend(choice.logprobs.token_logprobs)
                if choice.logprobs.top_logprobs:
                    top_logprobs_list.extend(choice.logprobs.top_logprobs)
                if choice.logprobs.tokens:
                    tokens.extend(choice.logprobs.tokens)

            if choice.finish_reason:
                completion_reason = choice.finish_reason

            if chunk.metrics:
                metrics_dict = chunk.metrics

            if chunk.usage:
                usage = chunk.usage

        # Build final message
        # If tool_calls are present, content should be None
        final_message = ChatMessage(
            role="assistant",
            content=None if tool_calls else full_content,
            tool_calls=tool_calls,
        )

        return ChatResponseModel(
            id=nonce,
            choices=[
                ChatChoice(
                    index=0,
                    finish_reason=completion_reason,
                    message=final_message,
                    logprobs=ChatLogProbs(
                        token_logprobs=token_logprobs,
                        top_logprobs=top_logprobs_list,
                        tokens=tokens,
                    )
                    if req.logprobs
                    else None,
                )
            ],
            usage=usage,
            created=int(time.time()),
            model=req.model,
            metrics=metrics_dict,
        )

    def resolve_request(self, nonce: str, result: Any):
        self.adapter.resolve_token(nonce, result)
