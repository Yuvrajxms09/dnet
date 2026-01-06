import asyncio
import time
import uuid
import json
import mlx.core as mx
import numpy as np
from typing import Optional, Any, List, Union, Dict

# Import MCP client
try:
    from .mcp_client import (
        MCPToolClient,
        create_mcp_client_from_presets,
        create_mcp_client_from_config,
        MCP_SERVER_PRESETS,
    )

    MCP_CLIENT_AVAILABLE = True
except ImportError:
    MCP_CLIENT_AVAILABLE = False
    MCPToolClient = None
    create_mcp_client_from_presets = None
    create_mcp_client_from_config = None
    MCP_SERVER_PRESETS = {}

# Fallback to toolregistry for backward compatibility
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
    """Inference manager for dnet with MCP and LangChain-style tool integration.

    Supports:
    - LangChain-style tool calling (prompting + robust parsing)
    - Dynamic MCP tool registration (Exa, GitHub, etc.)
    - Agent-style tool execution with synthesis
    - OpenAI-compatible API responses

    Example:
        # Register MCP tools using stdio transport (correct way)
        await inference_manager.register_mcp_stdio(
            server_name="exa",
            command="npx",
            args=["-y", "@anthropic-ai/exa-mcp-server"],
            env={"EXA_API_KEY": "your-key"}
        )

        # Or use presets
        await inference_manager.register_mcp_preset("exa")

        # Research with automatic tool execution
        response = await inference_manager.execute_tools_and_continue(request)
    """

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

        # New MCP client (preferred)
        self._mcp_client: Optional[MCPToolClient] = None
        if MCP_CLIENT_AVAILABLE:
            self._mcp_client = MCPToolClient()
            logger.info("MCPToolClient initialized for MCP tool integration")

        # Legacy toolregistry (fallback)
        self._tool_registry: Optional[ToolRegistry] = self._setup_tool_registry()

        # LangChain-compatible tool binding
        self._bound_tools: List[Dict[str, Any]] = []

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

                # Add tool system message if tools are available (LangChain-style)
                # Use bound tools (LangChain approach) or request tools (backward compatibility)
                available_tools = self._bound_tools or req.tools or []
                if available_tools:
                    tool_system_msg = self._create_langchain_tool_prompt(
                        available_tools
                    )
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

            # Add tool system message if tools are provided (LangChain-style)
            if req.tools:
                tool_system_msg = self._create_langchain_tool_prompt(req.tools)
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

        # Grammar JSON schema (removed structured outputs support)
        grammar_json_schema = None

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

        # Parse tool calls if tools were available
        tool_calls = None
        final_content = final_text
        available_tools = self._bound_tools or req.tools or []
        if available_tools:
            parsed_calls = self._parse_tool_calls_langchain_style(final_text)
            if parsed_calls:
                logger.info(f"Found {len(parsed_calls)} tool calls in response")
                # Convert parsed dicts to ToolCall objects
                tool_calls = self._convert_to_tool_call_objects(parsed_calls)
                final_content = self._format_tool_call_response(final_text, tool_calls)
            else:
                final_content = final_text

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

        # Parse tool calls if tools were available (LangChain-style, no grammar)
        tool_calls = None
        final_content = full_content
        available_tools = self._bound_tools or req.tools or []
        if available_tools:
            parsed_calls = self._parse_tool_calls_langchain_style(full_content)
            if parsed_calls:
                # Convert parsed dicts to ToolCall objects
                tool_calls = self._convert_to_tool_call_objects(parsed_calls)
                final_content = self._format_tool_call_response(
                    full_content, tool_calls
                )
            else:
                final_content = full_content

        return ChatResponseModel(
            id=nonce,
            choices=[
                ChatChoice(
                    index=0,
                    finish_reason=completion_reason,
                    message=ChatMessage(
                        role="assistant", content=final_content, tool_calls=tool_calls
                    ),
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

    def _create_tool_system_message(
        self, tools: List[Dict[str, Any]], registry: Optional[ToolRegistry] = None
    ) -> str:
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
            start = s.find("{", i)
            if start == -1:
                break

            # Try to parse JSON from this position
            try:
                # Find matching closing brace
                brace_count = 0
                end = start
                for j in range(start, len(s)):
                    if s[j] == "{":
                        brace_count += 1
                    elif s[j] == "}":
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

    def _convert_to_tool_calls(
        self, parsed: Dict[str, Any]
    ) -> Optional[List[ToolCall]]:
        """Convert parsed JSON to LangChain-compatible ToolCall format."""
        tool_name = parsed.get("tool") or parsed.get("name")
        tool_input = parsed.get("tool_input") or parsed.get("parameters", {})

        # Handle conversational responses
        if tool_name in ["__conversational_response", "__conversational_response"]:
            return None  # No tool calls, just conversational response

        if tool_name and tool_input is not None:
            return [
                ToolCall(
                    name=tool_name,
                    args=tool_input
                    if isinstance(tool_input, dict)
                    else {"input": tool_input},
                    id=f"call_{uuid.uuid4().hex}",
                )
            ]

        return None

    def _format_tool_call_response(
        self, content: str, tool_calls: Optional[List[ToolCall]]
    ) -> str:
        """Format response content when tool calls are present."""
        if tool_calls:
            return ""  # LangChain format: empty content when there are tool calls
        return content

    def _convert_to_tool_call_objects(
        self, parsed_calls: List[Dict[str, Any]]
    ) -> List[ToolCall]:
        """Convert parsed tool call dicts (OpenAI format) to ToolCall Pydantic objects.

        OpenAI format:
            {"id": "call_xxx", "type": "function", "function": {"name": "...", "arguments": "..."}}

        ToolCall format:
            {"name": "...", "args": {...}, "id": "..."}
        """
        tool_calls = []
        for call in parsed_calls:
            try:
                # Handle OpenAI format
                if "function" in call:
                    func = call.get("function", {})
                    name = func.get("name", "")
                    args_raw = func.get("arguments", "{}")

                    # Parse arguments if string
                    if isinstance(args_raw, str):
                        try:
                            args = json.loads(args_raw)
                        except json.JSONDecodeError:
                            args = {}
                    else:
                        args = args_raw or {}

                    tool_call = ToolCall(
                        name=name,
                        args=args,
                        id=call.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    )
                    tool_calls.append(tool_call)

                # Handle direct format (name, args already present)
                elif "name" in call:
                    args = call.get("args", call.get("arguments", {}))
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}

                    tool_call = ToolCall(
                        name=call["name"],
                        args=args,
                        id=call.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    )
                    tool_calls.append(tool_call)

            except Exception as e:
                logger.warning(f"Failed to convert tool call: {e}")
                continue

        return tool_calls

    def _setup_tool_registry(self) -> Optional[ToolRegistry]:
        """Initialize ToolRegistry for MCP tool execution."""
        if not TOOL_REGISTRY_AVAILABLE:
            logger.warning(
                "ToolRegistry not available. Install with: pip install toolregistry[mcp]"
            )
            return None

        registry = ToolRegistry()
        logger.info("ToolRegistry initialized for MCP tool execution")
        return registry

    def _register_mcp_tools(
        self, registry: ToolRegistry, mcp_configs: List[Dict[str, Any]]
    ) -> None:
        """Register MCP tools dynamically from configurations."""
        for config in mcp_configs:
            try:
                transport = config.get("transport")
                namespace = config.get("namespace", False)

                if TOOL_REGISTRY_AVAILABLE:
                    registry.register_from_mcp(transport, with_namespace=namespace)

            except Exception as e:
                logger.error(f"Failed to register MCP tools from {config}: {e}")

    async def _execute_tool_calls(
        self, tool_calls: List[ToolCall], registry: ToolRegistry
    ) -> List[Dict[str, Any]]:
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

                results.append(
                    {
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                        "success": True,
                    }
                )

            except Exception as e:
                logger.error(f"Tool execution failed for {tool_call.name}: {e}")
                results.append(
                    {
                        "tool_call_id": tool_call.id,
                        "content": f"Tool execution failed: {str(e)}",
                        "success": False,
                    }
                )

        return results

    def _create_langchain_tool_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """Create LangChain-style tool prompt (no grammar constraints, just instructions)."""
        all_tools = list(tools)  # Start with provided tools

        # Add tools from ToolRegistry if available
        if TOOL_REGISTRY_AVAILABLE and self._tool_registry:
            try:
                registry_tools = self._tool_registry.get_tools_json()
                # Filter out duplicates by name
                existing_names = {t.get("function", {}).get("name") for t in all_tools}
                for reg_tool in registry_tools:
                    reg_name = reg_tool.get("function", {}).get("name")
                    if reg_name and reg_name not in existing_names:
                        all_tools.append(reg_tool)
                        existing_names.add(reg_name)

            except Exception as e:
                logger.error(f"Failed to get tools from registry: {e}")

        if not all_tools:
            return ""

        tool_summaries = []
        for tool in all_tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                name = func.get("name", "unknown")
                desc = func.get("description", "")

                # Keep it concise for prompt
                short_desc = desc.split(".")[0][:100] if desc else ""
                summary = f"- {name}: {short_desc}"
                tool_summaries.append(summary)

        tools_section = "\n".join(tool_summaries)

        prompt = f"""

You have access to the following tools:
{tools_section}

To use a tool, respond with ONLY a JSON object containing tool calls:
{{"tool_calls": [{{"id": "call_1", "type": "function", "function": {{"name": "tool_name", "arguments": "{{\\"param\\": \\"value\\"}}"}}}}]}}

For general conversation or when no tools are needed, respond with plain text.

Important: Only output JSON when you actually want to call tools. For normal responses, just write naturally."""

        return prompt

    def _parse_tool_calls_langchain_style(
        self, content: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Parse tool calls using LangChain-style robust parsing (no grammar required)."""
        if not content or not content.strip():
            return None

        # Remove common prefixes/suffixes that models sometimes add
        clean_content = content.strip()
        original_length = len(clean_content)

        # Remove think tags
        import re

        clean_content = re.sub(
            r"<think>.*?</think>", "", clean_content, flags=re.DOTALL
        ).strip()

        # Remove special tokens
        special_tokens = [
            "<|im_end|>",
            "<|im_start|>",
            "<|endoftext|>",
            "</s>",
            "<|eot_id|>",
            "<|end|>",
        ]
        for token in special_tokens:
            clean_content = clean_content.replace(token, "").strip()

        prefixes_to_remove = ["Assistant:", "AI:", "Response:"]
        for prefix in prefixes_to_remove:
            if clean_content.startswith(prefix):
                clean_content = clean_content[len(prefix) :].strip()

        # Try direct JSON parsing first
        try:
            data = json.loads(clean_content)
            if isinstance(data, dict) and "tool_calls" in data:
                calls = data["tool_calls"]
                if isinstance(calls, list) and calls:
                    return calls
        except json.JSONDecodeError:
            pass

        # Fallback: Extract JSON from mixed text (LangChain-style)
        json_candidates = self._extract_json_from_text(clean_content)

        for candidate in json_candidates:
            try:
                data = json.loads(candidate)
                if isinstance(data, dict) and "tool_calls" in data:
                    calls = data["tool_calls"]
                    if isinstance(calls, list) and calls:
                        return calls
            except (json.JSONDecodeError, TypeError):
                continue

        # Check for single tool call format (OpenAI style)
        try:
            data = json.loads(clean_content)
            if isinstance(data, dict) and "function" in data:
                # Convert single tool call to tool_calls format
                tool_call = {
                    "id": data.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    "type": "function",
                    "function": data["function"],
                }
                return [tool_call]
        except (json.JSONDecodeError, KeyError):
            pass

        return None

    def _extract_json_from_text(self, text: str) -> List[str]:
        """Extract JSON blocks from text (LangChain-inspired approach)."""
        candidates = []

        # Find all JSON-like blocks
        brace_level = 0
        start_pos = -1

        i = 0
        while i < len(text):
            if text[i] == "{":
                if brace_level == 0:
                    start_pos = i
                brace_level += 1
            elif text[i] == "}":
                brace_level -= 1
                if brace_level == 0 and start_pos != -1:
                    json_block = text[start_pos : i + 1]
                    candidates.append(json_block)
                    start_pos = -1
            i += 1

        return candidates

    def _execute_tool_calls_simple(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute tool calls using MCP client or ToolRegistry."""
        # Check if we have any execution capability
        has_mcp = MCP_CLIENT_AVAILABLE and self._mcp_client
        has_registry = TOOL_REGISTRY_AVAILABLE and self._tool_registry

        if not has_mcp and not has_registry:
            logger.error("No tool execution backend available")
            return []

        results = []
        for tc in tool_calls:
            tool_id = tc.get("id", f"call_{uuid.uuid4().hex[:8]}")
            func = tc.get("function", {})
            tool_name = func.get("name", "")

            # Parse arguments
            args_raw = func.get("arguments", "{}")
            if isinstance(args_raw, str):
                try:
                    arguments = json.loads(args_raw)
                except json.JSONDecodeError as e:
                    arguments = {}
                    logger.warning(f"Failed to parse tool arguments: {e}")
            else:
                arguments = args_raw

            logger.info(f"Executing tool: {tool_name}")

            try:
                result = None

                # Try ToolRegistry first (includes HTTP MCP tools)
                if has_registry and tool_name in self._tool_registry.get_available_tools():
                    result = self._tool_registry.invoke(tool_name, **arguments)
                # Fallback to MCP client (for stdio MCP tools)
                elif has_mcp and tool_name in self._mcp_client.get_tool_names():
                    # MCP client execution is async, run in event loop
                    try:
                        result = asyncio.create_task(
                            self._mcp_client.execute_tool(tool_name, arguments)
                        )
                        result = asyncio.get_event_loop().run_until_complete(result)
                    except RuntimeError:
                        result = asyncio.run(
                            self._mcp_client.execute_tool(tool_name, arguments)
                        )
                else:
                    raise ValueError(f"Tool '{tool_name}' not found in any backend")

                result_str = str(result)
                results.append(
                    {"tool_call_id": tool_id, "content": result_str, "success": True}
                )

            except Exception as e:
                error_msg = f"Tool execution failed: {e}"
                logger.error(f"Tool '{tool_name}' failed: {e}")
                results.append(
                    {"tool_call_id": tool_id, "content": error_msg, "success": False}
                )

        return results

    async def _execute_tool_calls_async(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute tool calls asynchronously using MCP client or ToolRegistry."""
        has_mcp = MCP_CLIENT_AVAILABLE and self._mcp_client
        has_registry = TOOL_REGISTRY_AVAILABLE and self._tool_registry

        if not has_mcp and not has_registry:
            logger.error("No tool execution backend available (MCP or ToolRegistry)")
            return [
                {
                    "tool_call_id": "error",
                    "content": "No tool execution backend available",
                    "success": False,
                }
            ]

        mcp_tool_names = self._mcp_client.get_tool_names() if has_mcp else []

        results = []
        for tc in tool_calls:
            tool_id = tc.get("id", f"call_{uuid.uuid4().hex[:8]}")
            func = tc.get("function", {})
            tool_name = func.get("name", "")

            # Parse arguments
            args_raw = func.get("arguments", "{}")
            if isinstance(args_raw, str):
                try:
                    arguments = json.loads(args_raw)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse arguments: {e}")
                    arguments = {}
            else:
                arguments = args_raw

            try:
                result = None

                # Try ToolRegistry first (includes HTTP MCP tools)
                if has_registry and tool_name in self._tool_registry.get_available_tools():
                    result = self._tool_registry.invoke(tool_name, **arguments)
                # Fallback to MCP client (for stdio MCP tools)
                elif has_mcp and tool_name in mcp_tool_names:
                    result = await self._mcp_client.execute_tool(tool_name, arguments)
                else:
                    raise ValueError(
                        f"Tool '{tool_name}' not found. Available MCP tools: {mcp_tool_names}"
                    )

                result_str = str(result)
                results.append(
                    {"tool_call_id": tool_id, "content": result_str, "success": True}
                )
            except Exception as e:
                logger.error(f"Tool '{tool_name}' failed: {e}")
                results.append(
                    {
                        "tool_call_id": tool_id,
                        "content": f"Tool execution failed: {e}",
                        "success": False,
                    }
                )

        return results

    async def _generate_single_completion(
        self, req: ChatRequestModel
    ) -> ChatResponseModel:
        """
        Generate a single completion (non-streaming).

        This is a helper for agent-style execution that needs to make multiple
        LLM calls in sequence.
        """
        return await self.chat_completions(req)

    async def execute_tools_and_continue(
        self, req: ChatRequestModel
    ) -> ChatResponseModel:
        """
        Execute tools and continue conversation (for research use cases).
        This is a standalone agent-style method that doesn't interfere with normal chat completions.
        """
        # Step 1: Generate initial response (may contain tool calls)
        initial_response = await self._generate_single_completion(req)

        choice = initial_response.choices[0]
        if not choice.message or not choice.message.tool_calls:
            # No tool calls, return normal response
            return initial_response

        tool_calls = choice.message.tool_calls

        # Convert ToolCall objects to dict format for execution
        tool_calls_dict = [
            {
                "id": tc.id,
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.args) if tc.args else "{}",
                },
            }
            for tc in tool_calls
        ]

        tool_results = await self._execute_tool_calls_async(tool_calls_dict)

        # Step 3: Create new conversation with tool results
        new_messages = req.messages.copy()

        # Add assistant message with tool calls
        assistant_msg = choice.message.model_copy()
        new_messages.append(assistant_msg)

        # Add tool results
        for result in tool_results:
            tool_call_id = result["tool_call_id"]
            content = result["content"]

            # Find the corresponding tool call to get the tool name
            tool_name = "unknown_tool"
            for tc in tool_calls:
                if tc.id == tool_call_id:
                    tool_name = tc.name
                    break

            new_messages.append(
                ChatMessage(
                    role="tool",
                    name=tool_name,
                    content=content,
                    tool_call_id=tool_call_id,
                )
            )

        # Add guidance for the model to synthesize the final answer
        guidance_msg = ChatMessage(
            role="user",
            content="Based on the tool results above, provide a comprehensive and well-structured answer to my original question. Extract and organize the key information from the tool outputs.",
        )
        new_messages.append(guidance_msg)

        # Add guidance for the model to synthesize the final answer
        guidance_content = "Based on the tool results above, provide a comprehensive and well-structured answer to my original question. Extract and organize the key information from the tool outputs."
        guidance_msg = ChatMessage(role="user", content=guidance_content)
        new_messages.append(guidance_msg)

        # Step 4: Generate final synthesized response
        final_req = req.model_copy(
            update={
                "messages": new_messages,
                "tools": None,  # Don't include tools in final generation
            }
        )

        final_response = await self._generate_single_completion(final_req)

        # Mark as tool-synthesized response
        final_response.choices[0].finish_reason = ChatCompletionReason.STOP

        return final_response

    def bind_tools(self, tools: List[Any]) -> "InferenceManager":
        """
        Bind tools to this inference manager (LangChain-compatible interface).

        Args:
            tools: List of tool definitions. Can be:
                - Pydantic BaseModel classes
                - LangChain tools (@tool decorated functions)
                - Raw tool dictionaries (OpenAI format)
                - Functions with type hints

        Returns:
            InferenceManager: Self for method chaining
        """
        bound_tools = []
        for tool in tools:
            tool_def = self._convert_tool_to_definition(tool)
            if tool_def:
                bound_tools.append(tool_def)

        self._bound_tools = bound_tools
        return self

    def _convert_tool_to_definition(self, tool: Any) -> Optional[Dict[str, Any]]:
        """
        Convert various tool formats to OpenAI-compatible tool definition.
        """
        # If it's already a dict with OpenAI format
        if isinstance(tool, dict) and tool.get("type") == "function":
            return tool

        # If it's a Pydantic model
        if hasattr(tool, "__annotations__") and hasattr(tool, "model_json_schema"):
            try:
                schema = tool.model_json_schema()
                return {
                    "type": "function",
                    "function": {
                        "name": getattr(
                            tool, "__name__", tool.__class__.__name__.lower()
                        ),
                        "description": getattr(tool, "__doc__", "").strip(),
                        "parameters": schema,
                    },
                }
            except Exception as e:
                logger.warning(
                    f"Failed to convert Pydantic model to tool definition: {e}"
                )

        # If it's a function with @tool decorator (basic support)
        if callable(tool) and hasattr(tool, "__name__"):
            # Try to extract function signature
            import inspect

            try:
                sig = inspect.signature(tool)
                params = {}
                for name, param in sig.parameters.items():
                    if name == "self":
                        continue
                    # Basic type inference
                    param_def = {"type": "string"}  # default
                    if param.annotation != inspect.Parameter.empty:
                        if param.annotation == int:
                            param_def["type"] = "integer"
                        elif param.annotation == float:
                            param_def["type"] = "number"
                        elif param.annotation == bool:
                            param_def["type"] = "boolean"

                    params[name] = param_def

                return {
                    "type": "function",
                    "function": {
                        "name": tool.__name__,
                        "description": getattr(tool, "__doc__", "").strip(),
                        "parameters": {
                            "type": "object",
                            "properties": params,
                            "required": list(params.keys()),
                        },
                    },
                }
            except Exception as e:
                logger.warning(f"Failed to convert function to tool definition: {e}")

        logger.warning(f"Unsupported tool format: {type(tool)}")
        return None

    def get_bound_tools(self) -> List[Dict[str, Any]]:
        """Get currently bound tools."""
        return self._bound_tools.copy()

    async def register_mcp_stdio(
        self,
        server_name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Register MCP tools using stdio transport (spawns subprocess).

        This is the CORRECT way to register MCP tools. MCP uses stdio transport
        which spawns a subprocess that communicates via stdin/stdout.

        Based on langchain-mcp-adapters:
        https://github.com/langchain-ai/langchain-mcp-adapters

        Args:
            server_name: Unique name for this MCP server (e.g., "exa", "github")
            command: Command to run (e.g., "npx", "python", "node")
            args: Arguments for the command (e.g., ["-y", "@anthropic-ai/exa-mcp-server"])
            env: Environment variables (e.g., {"EXA_API_KEY": "..."})

        Returns:
            bool: True if registration successful

        Example:
            await manager.register_mcp_stdio(
                server_name="exa",
                command="npx",
                args=["-y", "@anthropic-ai/exa-mcp-server"],
                env={"EXA_API_KEY": os.getenv("EXA_API_KEY")}
            )
        """
        if not MCP_CLIENT_AVAILABLE:
            logger.error(
                "MCP client not available. Install: pip install langchain-mcp-adapters mcp"
            )
            return False

        try:
            # Create config for this server
            config = {
                server_name: {
                    "command": command,
                    "args": args or [],
                    "transport": "stdio",
                }
            }
            if env:
                config[server_name]["env"] = env

            # Initialize or update MCP client
            if self._mcp_client is None:
                self._mcp_client = MCPToolClient(server_configs=config)
            else:
                self._mcp_client._server_configs.update(config)

            # Load tools from this server
            await self._mcp_client.load_tools()

            # Update bound tools with newly registered MCP tools
            mcp_tools = self._mcp_client.get_tools_openai_format()
            for tool in mcp_tools:
                if tool not in self._bound_tools:
                    self._bound_tools.append(tool)

            logger.info(f"MCP server '{server_name}' registered successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to register MCP server '{server_name}': {e}")
            return False

    async def register_mcp_http(
        self, server_name: str, url: str, headers: Optional[Dict[str, str]] = None
    ) -> bool:
        """Register MCP tools using HTTP transport.

        This is the preferred transport for remote MCP servers.
        Uses ToolRegistry for reliable HTTP MCP support instead of langchain-mcp-adapters.

        Args:
            server_name: Unique name for this MCP server
            url: Server URL (e.g., "http://localhost:8000/mcp")
            headers: Optional HTTP headers (e.g., for authentication)

        Returns:
            bool: True if registration successful

        Example:
            await manager.register_mcp_http(
                server_name="weather",
                url="http://localhost:8000/mcp",
                headers={"Authorization": "Bearer token"}
            )
        """
        if not TOOL_REGISTRY_AVAILABLE:
            logger.error("ToolRegistry not available for HTTP MCP registration")
            return False

        try:
            # Initialize tool registry if needed
            if self._tool_registry is None:
                self._tool_registry = self._setup_tool_registry()

            if self._tool_registry is None:
                logger.error("Failed to initialize ToolRegistry")
                return False

            # Create transport for ToolRegistry
            if headers:
                # For custom headers, create StreamableHttpTransport instance
                try:
                    from fastmcp.client.transports import StreamableHttpTransport
                    transport = StreamableHttpTransport(url=url, headers=headers)
                except ImportError:
                    logger.warning("StreamableHttpTransport not available, trying URL without headers")
                    transport = url
            else:
                # Simple URL for transport
                transport = url

            # Register using ToolRegistry's MCP support (async version since we're in async context)
            await self._tool_registry.register_from_mcp_async(transport, with_namespace=True)

            # Get registered tools and add to bound tools
            registry_tools = self._tool_registry.get_tools_json()
            for tool in registry_tools:
                if tool not in self._bound_tools:
                    self._bound_tools.append(tool)

            logger.info(f"MCP server '{server_name}' (HTTP) registered successfully via ToolRegistry")
            return True

        except Exception as e:
            logger.error(f"Failed to register MCP server '{server_name}': {e}")
            return False

    async def register_mcp_sse(
        self, server_name: str, url: str, headers: Optional[Dict[str, str]] = None
    ) -> bool:
        """Register MCP tools using SSE (Server-Sent Events) transport.

        Note: HTTP transport is now preferred over SSE for remote servers.

        Args:
            server_name: Unique name for this MCP server
            url: SSE endpoint URL
            headers: Optional HTTP headers (e.g., for authentication)

        Returns:
            bool: True if registration successful
        """
        if not MCP_CLIENT_AVAILABLE:
            logger.error("MCP client not available")
            return False

        try:
            config = {
                server_name: {
                    "url": url,
                    "transport": "sse",
                }
            }
            if headers:
                config[server_name]["headers"] = headers

            if self._mcp_client is None:
                self._mcp_client = MCPToolClient(server_configs=config)
            else:
                self._mcp_client._server_configs.update(config)

            await self._mcp_client.load_tools()

            mcp_tools = self._mcp_client.get_tools_openai_format()
            for tool in mcp_tools:
                if tool not in self._bound_tools:
                    self._bound_tools.append(tool)

            logger.info(f"MCP server '{server_name}' (SSE) registered successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to register MCP server '{server_name}': {e}")
            return False

    async def register_mcp_preset(
        self, preset_name: str, env: Optional[Dict[str, str]] = None
    ) -> bool:
        """Register MCP tools using a predefined preset.

        Available presets:
        - "exa": Exa AI search (requires EXA_API_KEY)
        - "github": GitHub API (requires GITHUB_TOKEN)
        - "brave-search": Brave Search (requires BRAVE_API_KEY)
        - "filesystem": Local filesystem access
        - "fetch": HTTP fetch requests

        Args:
            preset_name: Name of the preset (e.g., "exa", "github")
            env: Optional environment variables override

        Returns:
            bool: True if registration successful

        Example:
            await manager.register_mcp_preset("exa", env={"EXA_API_KEY": "..."})
        """
        if not MCP_CLIENT_AVAILABLE:
            logger.error("MCP client not available")
            return False

        if preset_name not in MCP_SERVER_PRESETS:
            logger.error(f"Unknown MCP preset: {preset_name}")
            return False

        preset = MCP_SERVER_PRESETS[preset_name]

        # Build environment from preset and override
        import os

        final_env = {}
        if preset.get("env_key"):
            env_value = (env or {}).get(preset["env_key"]) or os.getenv(
                preset["env_key"]
            )
            if env_value:
                final_env[preset["env_key"]] = env_value
            else:
                logger.warning(
                    f"Environment variable {preset['env_key']} not set for preset '{preset_name}'"
                )

        return await self.register_mcp_stdio(
            server_name=preset_name,
            command=preset["command"],
            args=preset["args"],
            env=final_env,
        )

    async def register_mcp_servers(self, config: Dict[str, Dict[str, Any]]) -> bool:
        """Register multiple MCP servers from a configuration dict.

        This matches the langchain-mcp-adapters MultiServerMCPClient config format.

        Args:
            config: Server configurations:
                {
                    "math": {
                        "command": "python",
                        "args": ["/path/to/math_server.py"],
                        "transport": "stdio",
                    },
                    "weather": {
                        "url": "http://localhost:8000/mcp",
                        "transport": "http",
                    }
                }

        Returns:
            bool: True if all servers registered successfully

        Example:
            await manager.register_mcp_servers({
                "math": {"command": "python", "args": ["math_server.py"], "transport": "stdio"},
                "weather": {"url": "http://localhost:8000/mcp", "transport": "http"}
            })
        """
        if not MCP_CLIENT_AVAILABLE:
            logger.error("MCP client not available")
            return False

        try:
            self._mcp_client = MCPToolClient(server_configs=config)
            await self._mcp_client.load_tools()

            mcp_tools = self._mcp_client.get_tools_openai_format()
            for tool in mcp_tools:
                if tool not in self._bound_tools:
                    self._bound_tools.append(tool)

            logger.info(f"Registered {len(config)} MCP servers")
            return True

        except Exception as e:
            logger.error(f"Failed to register MCP servers: {e}")
            return False

    def register_mcp_tools(
        self, transport: str, namespace: Optional[str] = None
    ) -> bool:
        """[DEPRECATED] Register tools from an MCP server.

        ⚠️ This method is deprecated. Use register_mcp_stdio() or register_mcp_preset() instead.

        MCP servers do NOT use HTTP URLs. They use stdio (subprocess) or SSE transport.

        Args:
            transport: This parameter was incorrectly named. MCP doesn't use URLs.
            namespace: Optional namespace prefix for tool names

        Returns:
            bool: Always returns False with a deprecation warning
        """
        logger.warning("register_mcp_tools() is DEPRECATED!")
        logger.warning("   MCP servers use stdio or SSE transport, NOT HTTP URLs.")
        logger.warning("   Use register_mcp_stdio() or register_mcp_preset() instead.")
        logger.warning("")
        logger.warning("   Example:")
        logger.warning("     await manager.register_mcp_preset('exa')")
        logger.warning("   or:")
        logger.warning("     await manager.register_mcp_stdio(")
        logger.warning("       server_name='exa',")
        logger.warning("       command='npx',")
        logger.warning("       args=['-y', '@anthropic-ai/exa-mcp-server'],")
        logger.warning("       env={'EXA_API_KEY': 'your-key'}")
        logger.warning("     )")

        return False

    async def register_mcp_tools_async(
        self, transport: str, namespace: Optional[str] = None
    ) -> bool:
        """[DEPRECATED] Async version of MCP tool registration.

        ⚠️ This method is deprecated. Use register_mcp_stdio() or register_mcp_preset() instead.

        Args:
            transport: This parameter was incorrectly named. MCP doesn't use URLs.
            namespace: Optional namespace prefix

        Returns:
            bool: Always returns False with a deprecation warning
        """
        logger.warning("register_mcp_tools_async() is DEPRECATED!")
        logger.warning("   Use register_mcp_preset() or register_mcp_stdio() instead.")

        # If it looks like a preset name, try to use the preset
        if transport in ["exa", "github", "brave-search", "filesystem", "fetch"]:
            logger.info(
                f"Detected preset name '{transport}', using register_mcp_preset()"
            )
            return await self.register_mcp_preset(transport)

        return False

    def register_openapi_tools(
        self, openapi_spec: Union[str, Dict], client_config: Optional[Dict] = None
    ) -> bool:
        """Register tools from OpenAPI specification.

        Args:
            openapi_spec: OpenAPI spec as dict or URL/file path
            client_config: HTTP client configuration

        Returns:
            bool: True if registration successful
        """
        if not TOOL_REGISTRY_AVAILABLE or not self._tool_registry:
            logger.warning("ToolRegistry not available for OpenAPI tool registration")
            return False

        try:
            # Register OpenAPI tools using ToolRegistry
            if client_config:
                # ToolRegistry expects HttpxClientConfig, but we'll pass dict for now
                self._tool_registry.register_from_openapi(
                    client_config=client_config, openapi_spec=openapi_spec
                )
            else:
                # Try with just the spec
                self._tool_registry.register_from_openapi(openapi_spec=openapi_spec)

            logger.info("Successfully registered OpenAPI tools")
            return True
        except Exception as e:
            logger.error(f"Failed to register OpenAPI tools: {e}")
            return False

    def get_registered_tools(self) -> List[str]:
        """Get list of all registered tool names from all sources."""
        all_tools = []

        # Get tools from MCP client
        if MCP_CLIENT_AVAILABLE and self._mcp_client:
            mcp_tools = self._mcp_client.get_tool_names()
            all_tools.extend(mcp_tools)

        # Get tools from ToolRegistry (legacy)
        if TOOL_REGISTRY_AVAILABLE and self._tool_registry:
            try:
                registry_tools = self._tool_registry.get_available_tools()
                # Add only tools not already in list
                for tool in registry_tools:
                    if tool not in all_tools:
                        all_tools.append(tool)
            except Exception as e:
                logger.warning(f"Failed to get ToolRegistry tools: {e}")

        # Get tools from bound tools
        for tool in self._bound_tools:
            name = tool.get("function", {}).get("name", "")
            if name and name not in all_tools:
                all_tools.append(name)

        return all_tools

    def get_all_tools_openai_format(self) -> List[Dict[str, Any]]:
        """Get all tools in OpenAI-compatible format."""
        tools = list(self._bound_tools)

        # Add MCP tools
        if MCP_CLIENT_AVAILABLE and self._mcp_client:
            mcp_tools = self._mcp_client.get_tools_openai_format()
            for tool in mcp_tools:
                if tool not in tools:
                    tools.append(tool)

        return tools

    def resolve_request(self, nonce: str, result: Any):
        """Resolve a pending request with the given result.

        Called by gRPC servicer when a token is received from a shard.
        """
        self.adapter.resolve_token(nonce, result)


class StructuredOutputInferenceManager:
    """
    LangGraph-style structured output wrapper for InferenceManager.

    Forces the agent to return responses in a specific structured format by binding
    the response schema as a tool that must be called (LangGraph "Option 1").

    This ensures the agent provides structured output without requiring a second LLM call.
    """

    def __init__(self, inference_manager: "InferenceManager", schema: Any):
        self.inference_manager = inference_manager
        self.schema = schema

        # Generate tool definition from schema
        self._structured_output_tool = self._schema_to_tool(schema)

    def _schema_to_tool(self, schema: Any) -> Dict[str, Any]:
        """Convert Pydantic schema or JSON schema to tool definition."""
        if hasattr(schema, "model_json_schema"):
            # Pydantic model
            json_schema = schema.model_json_schema()
            return {
                "type": "function",
                "function": {
                    "name": schema.__name__,
                    "description": getattr(schema, "__doc__", "").strip()
                    or "Structured response",
                    "parameters": json_schema,
                },
            }
        elif isinstance(schema, dict):
            # Raw JSON schema
            return {
                "type": "function",
                "function": {
                    "name": "StructuredResponse",
                    "description": "Structured response",
                    "parameters": schema,
                },
            }
        else:
            raise ValueError(f"Unsupported schema type: {type(schema)}")

    async def chat_completions(self, req: ChatRequestModel) -> ChatResponseModel:
        """
        Generate completion with guaranteed structured output.

        The agent will be forced to call the structured output tool to provide its final answer.
        """
        # Temporarily bind the structured output tool
        original_tools = (
            self.inference_manager._bound_tools.copy()
            if hasattr(self.inference_manager, "_bound_tools")
            else []
        )

        try:
            # Add structured output tool to bound tools or request tools
            if hasattr(self.inference_manager, "_bound_tools"):
                # LangChain-style: add to bound tools
                self.inference_manager._bound_tools.append(self._structured_output_tool)
            else:
                # Fallback: add to request tools
                if not req.tools:
                    req.tools = []
                req.tools.append(self._structured_output_tool)

            # Force tool calling by setting tool_choice
            if hasattr(req, "tool_choice"):
                req.tool_choice = "any"  # Force at least one tool call

            # Generate response (agent should call the structured output tool)
            response = await self.inference_manager.chat_completions(req)

            # Extract structured data from tool calls
            if response.choices and response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    tool_name = (
                        tool_call.name
                        if hasattr(tool_call, "name")
                        else tool_call.get("function", {}).get("name", "")
                    )
                    if tool_name == self._structured_output_tool["function"]["name"]:
                        # Parse the structured arguments
                        if hasattr(tool_call, "args"):
                            structured_data = tool_call.args
                        else:
                            # Handle dict format
                            args_str = tool_call.get("function", {}).get(
                                "arguments", "{}"
                            )
                            try:
                                structured_data = (
                                    json.loads(args_str)
                                    if isinstance(args_str, str)
                                    else args_str
                                )
                            except:
                                structured_data = {}

                        # Replace response content with structured data
                        response.choices[0].message.content = str(structured_data)

                        # Add structured output field to response
                        response.structured_output = structured_data
                        break

            return response

        finally:
            # Restore original tools
            if hasattr(self.inference_manager, "_bound_tools"):
                self.inference_manager._bound_tools = original_tools

    def bind_tools(self, tools: List[Any]) -> "StructuredOutputInferenceManager":
        """
        Bind additional tools while keeping the structured output tool.

        This allows binding action tools + maintaining structured output.
        """
        # Bind tools on the underlying inference manager
        if hasattr(self.inference_manager, "bind_tools"):
            self.inference_manager.bind_tools(tools)
        return self

    def __getattr__(self, name):
        """Delegate other methods to the underlying inference manager."""
        return getattr(self.inference_manager, name)

    def with_structured_output(self, schema: Any) -> "StructuredOutputInferenceManager":
        """
        Create an inference manager that returns structured output (LangGraph-style).

        This implements the "Option 1: Bind output as tool" approach from LangGraph,
        where the response schema is bound as a tool that the agent must call to respond.

        Args:
            schema: Pydantic model or JSON schema for structured output

        Returns:
            StructuredOutputInferenceManager: Wrapper that ensures structured responses

        Example:
            class WeatherResponse(BaseModel):
                temperature: float
                wind_direction: str

            # Create structured output wrapper
            structured_llm = inference_manager.with_structured_output(WeatherResponse)

            # Agent will call WeatherResponse tool with structured data when ready to respond
        """
        return StructuredOutputInferenceManager(self.inference_manager, schema)
