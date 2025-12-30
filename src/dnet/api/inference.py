import asyncio
import time
import uuid
import json
import mlx.core as mx
import numpy as np
from typing import Optional, Any, List, Union, Dict
try:
    from toolregistry import ToolRegistry
    TOOL_REGISTRY_AVAILABLE = True
except ImportError:
    TOOL_REGISTRY_AVAILABLE = False
    ToolRegistry = None
from builtins import aiter, anext
from dnet.core.tensor import to_bytes

from .models import (
    ChatRequestModel,
    ChatResponseModel,
    ChatChoice,
    ChatMessage,
    ChatUsage,
    ChatCompletionReason,
    ChatLogProbs,
    StructuredOutputsParams,
    ToolCall,
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
        self._tool_registry: Optional[ToolRegistry] = self._setup_tool_registry()

    async def connect_to_ring(
        self, first_shard_ip: str, first_shard_port: int, api_callback_addr: str
    ) -> None:
        """
        `api_callback_addr` must be a reachable `host:port` from shards.
        For internet setups, this should be a public IP/DNS or overlay VPN IP.
        """
        await self.adapter.connect_first_shard(first_shard_ip, first_shard_port)
        self._api_callback_addr = api_callback_addr

    async def generate_stream(self, req: ChatRequestModel):
        """Generator for chat completion chunks."""
        logger.debug(f"generate_stream called: model={req.model}")

        if not self.model_manager.tokenizer:
            raise RuntimeError(
                "Inference manager not ready (ring not connected or tokenizer not loaded)"
            )

        tokenizer = self.model_manager.tokenizer

        try:
            if (
                hasattr(tokenizer, "chat_template")
                and tokenizer.chat_template is not None
            ):
                # Convert messages to dict format
                message_dicts = []

                # Add tool system message if tools are provided
                if req.tools:
                    tool_system_msg = self._create_tool_system_message(req.tools, self._tool_registry)
                    message_dicts.append({"role": "system", "content": tool_system_msg})

                for m in req.messages:
                    msg_dict = {"role": m.role, "content": m.content or ""}
                    message_dicts.append(msg_dict)

                prompt_text = tokenizer.apply_chat_template(
                    message_dicts,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            else:
                prompt_text = (
                    "\n".join(m.content or "" for m in req.messages) + "\nAssistant:"
                )
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}, using fallback")
            prompt_parts = []

            # Add tool system message if tools are provided
            if req.tools:
                tool_system_msg = self._create_tool_system_message(req.tools, self._tool_registry)
                prompt_parts.append(f"System: {tool_system_msg}")

            prompt_parts.extend(m.content or "" for m in req.messages)
            prompt_parts.append("Assistant:")
            prompt_text = "\n".join(prompt_parts)

        prompt_tokens = tokenizer.encode(prompt_text)
        prompt_array = mx.array(prompt_tokens)

        stop_id_sequences = []
        if req.stop:
            for stop_word in req.stop:
                stop_id_sequences.append(
                    tokenizer.encode(stop_word, add_special_tokens=False)
                )

        # Convert OpenAI response_format to internal structured_outputs format
        if req.response_format and req.response_format.get("type") == "json_schema":
            json_schema = req.response_format["json_schema"]["schema"]
            req.structured_outputs = StructuredOutputsParams(json=json_schema)

        # Get grammar JSON schema for structured output
        grammar_json_schema = None
        if req.structured_outputs and req.structured_outputs.json:
            grammar_json_schema = json.dumps(req.structured_outputs.json)

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
                grammar_json_schema=grammar_json_schema,
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

            # Yield chunk
            yield ChatResponseModel(
                id=nonce,
                choices=[
                    ChatChoice(
                        index=0,
                        delta=ChatMessage(role="assistant", content=delta_text),
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

            # stopping criteria
            if token == tokenizer.eos_token_id:
                completion_reason = ChatCompletionReason.STOP
                break

            y = mx.array([token], dtype=mx.int32)

        detokenizer.finalize()
        final_text = detokenizer.text

        # Strip special tokens from output
        # mlx-lm's NaiveStreamingDetokenizer calls tokenizer.decode() without skip_special_tokens=True
        # (see: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tokenizer_utils.py)
        # So we strip them manually as a post-processing step
        SPECIAL_TOKENS_TO_STRIP = [
            "<|im_end|>",  # Qwen, ChatML format
            "<|im_start|>",  # Qwen, ChatML format
            "<|endoftext|>",  # GPT/generic
            "</s>",  # Llama, Mistral
            "<|eot_id|>",  # Llama 3
            "<|end|>",  # Phi
            "<|assistant|>",  # Some chat templates
            "<|user|>",  # Some chat templates
        ]
        for token in SPECIAL_TOKENS_TO_STRIP:
            final_text = final_text.replace(token, "")
        final_text = final_text.strip()

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

        # Parse tool calls if tools were provided
        tool_calls = None
        final_content = final_text
        if req.tools:
            tool_calls = self._parse_tool_calls(final_text)
            final_content = self._format_tool_call_response(final_text, tool_calls) if tool_calls else final_text

        final_message = ChatMessage(
            role="assistant",
            content=final_content,
            tool_calls=tool_calls,
        )

        # Final chunk
        yield ChatResponseModel(
            id=nonce,
            choices=[
                ChatChoice(
                    index=0,
                    delta=None,
                    message=final_message,
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

        async for chunk in self.generate_stream(req):
            nonce = chunk.id
            choice = chunk.choices[0]
            if choice.delta and choice.delta.content:
                full_content += choice.delta.content

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

        # Clean up structured output responses - remove end tokens
        if req.structured_outputs and req.structured_outputs.json:
            full_content = full_content.strip()
            for token in ["<|im_end|>", "<|endoftext|>", "</s>"]:
                if token in full_content:
                    full_content = full_content.split(token)[0].strip()

        # Parse tool calls if tools were provided
        tool_calls = None
        final_content = full_content
        if req.tools:
            tool_calls = self._parse_tool_calls(full_content)
            final_content = self._format_tool_call_response(full_content, tool_calls) if tool_calls else full_content

        return ChatResponseModel(
            id=nonce,
            choices=[
                ChatChoice(
                    index=0,
                    finish_reason=completion_reason,
                    message=ChatMessage(role="assistant", content=final_content, tool_calls=tool_calls),
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

    def _create_tool_system_message(self, tools: List[Dict[str, Any]], registry: Optional[ToolRegistry] = None) -> str:
        """Create system message with available tools for LangChain-style tool calling."""
        all_tools = list(tools)  # Start with provided tools

        # Add tools from registry if available
        if registry and TOOL_REGISTRY_AVAILABLE:
            try:
                registry_tools = registry.get_tools_json()
                all_tools.extend(registry_tools)
                logger.debug(f"Added {len(registry_tools)} tools from registry")
            except Exception as e:
                logger.warning(f"Failed to get tools from registry: {e}")

        tools_json = json.dumps(all_tools, indent=2)

        system_template = """You have access to the following tools:

{tools}

You must always select one of the above tools and respond with only a JSON object matching the following schema:

{{
  "tool": "<name of the selected tool>",
  "tool_input": "<parameters for the selected tool, matching the tool's JSON schema>"
}}

If you want to respond conversationally without using tools, use the "__conversational_response" tool with a "response" parameter containing your message."""

        return system_template.format(tools=tools_json)

    def _parse_tool_calls(self, content: str) -> Optional[List[ToolCall]]:
        """Parse tool calls from model output using LangChain-style parsing."""
        if not content.strip():
            return None

        try:
            # Try direct JSON parsing first
            parsed = json.loads(content.strip())
            return self._convert_to_tool_calls(parsed)
        except json.JSONDecodeError:
            try:
                # Fallback: extract JSON from mixed text using LangChain-style parsing
                candidates = self._extract_json_candidates(content)
                if candidates:
                    parsed = candidates[0]  # Take first valid JSON
                    return self._convert_to_tool_calls(parsed)
            except Exception:
                pass

        return None

    def _extract_json_candidates(self, s: str) -> List[Any]:
        """Extract JSON objects from text, similar to LangChain's parse_json_garbage."""
        candidates = []
        i = 0
        while i < len(s):
            # Find opening brace
            start = s.find('{', i)
            if start == -1:
                break

            # Try to parse JSON from this position
            try:
                # Find matching closing brace
                brace_count = 0
                end = start
                for j in range(start, len(s)):
                    if s[j] == '{':
                        brace_count += 1
                    elif s[j] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = j + 1
                            break

                if brace_count == 0:
                    json_str = s[start:end]
                    parsed = json.loads(json_str)
                    candidates.append(parsed)
                    i = end
                else:
                    i = start + 1
            except (json.JSONDecodeError, ValueError):
                i = start + 1

        return candidates

    def _convert_to_tool_calls(self, parsed: Dict[str, Any]) -> Optional[List[ToolCall]]:
        """Convert parsed JSON to LangChain-compatible ToolCall format."""
        tool_name = parsed.get("tool") or parsed.get("name")
        tool_input = parsed.get("tool_input") or parsed.get("parameters", {})

        # Handle conversational responses
        if tool_name in ["__conversational_response", "__conversational_response"]:
            return None  # No tool calls, just conversational response

        if tool_name and tool_input is not None:
            return [ToolCall(
                name=tool_name,
                args=tool_input if isinstance(tool_input, dict) else {"input": tool_input},
                id=f"call_{uuid.uuid4().hex}"
            )]

        return None

    def _format_tool_call_response(self, content: str, tool_calls: Optional[List[ToolCall]]) -> str:
        """Format response content when tool calls are present."""
        if tool_calls:
            return ""  # LangChain format: empty content when there are tool calls
        return content

    def _setup_tool_registry(self) -> Optional[ToolRegistry]:
        """Initialize ToolRegistry for MCP tool execution."""
        if not TOOL_REGISTRY_AVAILABLE:
            logger.warning("ToolRegistry not available. Install with: pip install toolregistry[mcp]")
            return None

        registry = ToolRegistry()
        logger.info("ToolRegistry initialized for MCP tool execution")
        return registry

    def _register_mcp_tools(self, registry: ToolRegistry, mcp_configs: List[Dict[str, Any]]) -> None:
        """Register MCP tools dynamically from configurations."""
        for config in mcp_configs:
            try:
                transport = config.get("transport")
                namespace = config.get("namespace", False)

                if TOOL_REGISTRY_AVAILABLE:
                    registry.register_from_mcp(transport, with_namespace=namespace)
                    logger.info(f"Registered MCP tools from {transport}")

            except Exception as e:
                logger.error(f"Failed to register MCP tools from {config}: {e}")

    async def _execute_tool_calls(self, tool_calls: List[ToolCall], registry: ToolRegistry) -> List[Dict[str, Any]]:
        """Execute tool calls using ToolRegistry and return results."""
        results = []

        for tool_call in tool_calls:
            try:
                # Get the callable tool from registry
                tool_func = registry.get_callable(tool_call.name)
                if not tool_func:
                    logger.error(f"Tool {tool_call.name} not found in registry")
                    continue

                # Execute the tool
                if asyncio.iscoroutinefunction(tool_func):
                    result = await tool_func(**tool_call.args)
                else:
                    result = tool_func(**tool_call.args)

                results.append({
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                    "success": True
                })

                logger.info(f"Executed tool {tool_call.name} successfully")

            except Exception as e:
                logger.error(f"Tool execution failed for {tool_call.name}: {e}")
                results.append({
                    "tool_call_id": tool_call.id,
                    "content": f"Tool execution failed: {str(e)}",
                    "success": False
                })

        return results

    def resolve_request(self, nonce: str, result: Any):
        self.adapter.resolve_token(nonce, result)
