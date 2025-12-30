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

                # Add tool system message if tools are provided (LangChain-style)
                if req.tools:
                    tool_system_msg = self._create_langchain_tool_prompt(req.tools)
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

        # Parse tool calls if tools were provided (LangChain-style, no grammar)
        tool_calls = None
        final_content = full_content
        if req.tools:
            tool_calls = self._parse_tool_calls_langchain_style(full_content)
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

    def _create_langchain_tool_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """Create LangChain-style tool prompt (no grammar constraints, just instructions)."""
        if not tools:
            return ""

        tool_summaries = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                name = func.get("name", "unknown")
                desc = func.get("description", "")

                # Keep it concise for prompt
                short_desc = desc.split(".")[0][:100] if desc else ""
                tool_summaries.append(f"- {name}: {short_desc}")

        tools_section = "\n".join(tool_summaries)

        return f"""

You have access to the following tools:
{tools_section}

To use a tool, respond with ONLY a JSON object containing tool calls:
{{"tool_calls": [{{"id": "call_1", "type": "function", "function": {{"name": "tool_name", "arguments": "{{\\"param\\": \\"value\\"}}"}}}}]}}

For general conversation or when no tools are needed, respond with plain text.

Important: Only output JSON when you actually want to call tools. For normal responses, just write naturally."""

    def _parse_tool_calls_langchain_style(self, content: str) -> Optional[List[Dict[str, Any]]]:
        """Parse tool calls using LangChain-style robust parsing (no grammar required)."""
        if not content or not content.strip():
            return None

        # Remove common prefixes/suffixes that models sometimes add
        clean_content = content.strip()
        prefixes_to_remove = ['Assistant:', 'AI:', 'Response:']
        for prefix in prefixes_to_remove:
            if clean_content.startswith(prefix):
                clean_content = clean_content[len(prefix):].strip()

        # Try direct JSON parsing first
        try:
            data = json.loads(clean_content)
            if isinstance(data, dict) and "tool_calls" in data:
                calls = data["tool_calls"]
                if isinstance(calls, list) and calls:
                    logger.debug(f"Parsed {len(calls)} tool calls via direct JSON")
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
                        logger.debug(f"Parsed {len(calls)} tool calls via extracted JSON")
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
                    "function": data["function"]
                }
                logger.debug("Converted single tool call format")
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
            if text[i] == '{':
                if brace_level == 0:
                    start_pos = i
                brace_level += 1
            elif text[i] == '}':
                brace_level -= 1
                if brace_level == 0 and start_pos != -1:
                    json_block = text[start_pos:i+1]
                    candidates.append(json_block)
                    start_pos = -1
            i += 1

        return candidates

    def _execute_tool_calls_simple(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute tool calls using ToolRegistry (simple version for standalone tool calling)."""
        if not TOOL_REGISTRY_AVAILABLE or not self._tool_registry:
            logger.warning("ToolRegistry not available for tool execution")
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
                except json.JSONDecodeError:
                    arguments = {}
                    logger.warning(f"Failed to parse tool arguments: {args_raw[:100]}")
            else:
                arguments = args_raw

            logger.info(f"Executing tool: {tool_name} with args: {arguments}")

            try:
                # Execute via ToolRegistry
                tool_func = self._tool_registry.get_callable(tool_name)
                if tool_func:
                    if asyncio.iscoroutinefunction(tool_func):
                        result = asyncio.run(tool_func(**arguments))
                    else:
                        result = tool_func(**arguments)

                    results.append({
                        "tool_call_id": tool_id,
                        "content": str(result),
                        "success": True
                    })
                    logger.info(f"Tool '{tool_name}' executed successfully")
                else:
                    raise ValueError(f"Tool '{tool_name}' not found in registry")

            except Exception as e:
                error_msg = f"Tool execution failed: {e}"
                logger.error(f"Tool '{tool_name}' failed: {e}")
                results.append({
                    "tool_call_id": tool_id,
                    "content": error_msg,
                    "success": False
                })

        return results

    async def execute_tools_and_continue(self, req: ChatRequestModel) -> ChatResponseModel:
        """
        Execute tools and continue conversation (for research use cases).
        This is a standalone agent-style method that doesn't interfere with normal chat completions.
        """
        logger.info(f"Starting tool execution flow for: {req.messages[-1].content[:100] if req.messages else 'empty'}")

        # Step 1: Generate initial response (may contain tool calls)
        initial_response = await self._generate_single_completion(req)

        choice = initial_response.choices[0]
        if not choice.message or not choice.message.tool_calls:
            # No tool calls, return normal response
            logger.info("No tool calls generated, returning normal response")
            return initial_response

        tool_calls = choice.message.tool_calls
        logger.info(f"Executing {len(tool_calls)} tool calls")

        # Step 2: Execute tools
        tool_results = self._execute_tool_calls_simple([
            {
                "id": tc.id,
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.args) if tc.args else "{}"
                }
            } for tc in tool_calls
        ])

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

            new_messages.append(ChatMessage(
                role="tool",
                name=tool_name,
                content=content,
                tool_call_id=tool_call_id
            ))

        # Add guidance for the model to synthesize the final answer
        guidance_msg = ChatMessage(
            role="user",
            content="Based on the tool results above, provide a comprehensive and well-structured answer to my original question. Extract and organize the key information from the tool outputs."
        )
        new_messages.append(guidance_msg)

        # Step 4: Generate final synthesized response
        final_req = req.model_copy(update={
            "messages": new_messages,
            "tools": None,  # Don't include tools in final generation
        })

        logger.info("Generating final synthesized response")
        final_response = await self._generate_single_completion(final_req)

        # Mark as tool-synthesized response
        final_response.choices[0].finish_reason = ChatCompletionReason.STOP

        return final_response

    def resolve_request(self, nonce: str, result: Any):
        self.adapter.resolve_token(nonce, result)
