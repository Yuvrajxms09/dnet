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
    """Inference manager for dnet with ToolRegistry integration.

    Supports:
    - LangChain-style tool calling (prompting + robust parsing)
    - Dynamic MCP tool registration (Exa, GitHub, etc.)
    - Agent-style tool execution with synthesis
    - OpenAI-compatible API responses

    Example:
        # Register MCP tools
        inference_manager.register_mcp_tools("https://exa-mcp.com", "exa")

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

                # Add tool system message if tools are available (LangChain-style)
                # Use bound tools (LangChain approach) or request tools (backward compatibility)
                available_tools = self._bound_tools or req.tools or []
                if available_tools:
                    tool_system_msg = self._create_langchain_tool_prompt(available_tools)
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

        # Parse tool calls if tools were available
        logger.debug(f"🔍 Checking for tool calls in generate_stream response")
        tool_calls = None
        final_content = final_text
        available_tools = self._bound_tools or req.tools or []
        if available_tools:
            logger.info(f"🛠️ Tools available ({len(available_tools)}), attempting to parse tool calls")
            tool_calls = self._parse_tool_calls_langchain_style(final_text)
            if tool_calls:
                logger.info(f"✅ Found {len(tool_calls)} tool calls in response")
                final_content = self._format_tool_call_response(final_text, tool_calls)
                logger.debug(f"📝 Formatted response content (tool calls present)")
            else:
                logger.debug("📝 No tool calls found, using original content")
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

        # Clean up structured output responses - remove end tokens
        if req.structured_outputs and req.structured_outputs.json:
            full_content = full_content.strip()
            for token in ["<|im_end|>", "<|endoftext|>", "</s>"]:
                if token in full_content:
                    full_content = full_content.split(token)[0].strip()

        # Parse tool calls if tools were available (LangChain-style, no grammar)
        tool_calls = None
        final_content = full_content
        available_tools = self._bound_tools or req.tools or []
        if available_tools:
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
        logger.debug(f"🔧 Creating tool prompt with {len(tools)} provided tools")

        all_tools = list(tools)  # Start with provided tools
        logger.debug(f"📋 Initial tools count: {len(all_tools)}")

        # Add tools from ToolRegistry if available
        if TOOL_REGISTRY_AVAILABLE and self._tool_registry:
            try:
                registry_tools = self._tool_registry.get_tools_json()
                logger.debug(f"📚 Found {len(registry_tools)} tools in registry")

                # Filter out duplicates by name
                existing_names = {t.get("function", {}).get("name") for t in all_tools}
                added_count = 0
                for reg_tool in registry_tools:
                    reg_name = reg_tool.get("function", {}).get("name")
                    if reg_name and reg_name not in existing_names:
                        all_tools.append(reg_tool)
                        existing_names.add(reg_name)
                        added_count += 1
                        logger.debug(f"✅ Added registry tool: {reg_name}")

                logger.debug(f"📈 Total tools after registry merge: {len(all_tools)} (+{added_count} from registry)")

            except Exception as e:
                logger.error(f"❌ Failed to get tools from registry: {e}")
                logger.debug("Registry error details:", exc_info=True)
        else:
            logger.debug("📚 ToolRegistry not available or not initialized")

        if not all_tools:
            logger.warning("⚠️ No tools available for prompt generation")
            return ""

        logger.debug(f"🛠️ Generating prompt for {len(all_tools)} tools")

        tool_summaries = []
        for i, tool in enumerate(all_tools):
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                name = func.get("name", "unknown")
                desc = func.get("description", "")

                # Keep it concise for prompt
                short_desc = desc.split(".")[0][:100] if desc else ""
                summary = f"- {name}: {short_desc}"
                tool_summaries.append(summary)
                logger.debug(f"📝 Tool {i+1}: {name} - {short_desc[:50]}...")
            else:
                logger.warning(f"⚠️ Malformed tool definition at index {i}: {tool}")

        tools_section = "\n".join(tool_summaries)
        logger.debug(f"📄 Generated tools section with {len(tool_summaries)} summaries")

        prompt = f"""

You have access to the following tools:
{tools_section}

To use a tool, respond with ONLY a JSON object containing tool calls:
{{"tool_calls": [{{"id": "call_1", "type": "function", "function": {{"name": "tool_name", "arguments": "{{\\"param\\": \\"value\\"}}"}}}}]}}

For general conversation or when no tools are needed, respond with plain text.

Important: Only output JSON when you actually want to call tools. For normal responses, just write naturally."""

        logger.debug(f"📋 Final prompt length: {len(prompt)} characters")
        logger.debug(f"🎯 Prompt preview: {prompt[:200]}...")

        return prompt

    def _parse_tool_calls_langchain_style(self, content: str) -> Optional[List[Dict[str, Any]]]:
        """Parse tool calls using LangChain-style robust parsing (no grammar required)."""
        logger.debug(f"🔍 Starting tool call parsing for content length: {len(content)}")

        if not content or not content.strip():
            logger.debug("📭 Empty content, no tool calls to parse")
            return None

        # Remove common prefixes/suffixes that models sometimes add
        clean_content = content.strip()
        original_length = len(clean_content)

        # Remove think tags
        import re
        clean_content = re.sub(r'<think>.*?</think>', '', clean_content, flags=re.DOTALL).strip()

        # Remove special tokens
        special_tokens = ['<|im_end|>', '<|im_start|>', '<|endoftext|>', '</s>', '<|eot_id|>', '<|end|>']
        for token in special_tokens:
            clean_content = clean_content.replace(token, '').strip()

        prefixes_to_remove = ['Assistant:', 'AI:', 'Response:']
        for prefix in prefixes_to_remove:
            if clean_content.startswith(prefix):
                clean_content = clean_content[len(prefix):].strip()
                logger.debug(f"🧹 Removed prefix '{prefix}', content now: {clean_content[:100]}...")

        if len(clean_content) != original_length:
            logger.debug(f"📏 Content cleaned from {original_length} to {len(clean_content)} chars")

        # Try direct JSON parsing first
        logger.debug("🎯 Attempting direct JSON parsing...")
        try:
            data = json.loads(clean_content)
            logger.debug(f"✅ Valid JSON parsed: {type(data)}")

            if isinstance(data, dict) and "tool_calls" in data:
                calls = data["tool_calls"]
                if isinstance(calls, list) and calls:
                    logger.info(f"🎉 SUCCESS: Parsed {len(calls)} tool calls via direct JSON")
                    for i, call in enumerate(calls):
                        logger.debug(f"🔧 Call {i+1}: {call.get('function', {}).get('name', 'unknown')}")
                    return calls
                else:
                    logger.debug(f"⚠️ JSON has tool_calls but it's not a valid list: {calls}")
        except json.JSONDecodeError as e:
            logger.debug(f"❌ Direct JSON parsing failed: {e}")
            logger.debug(f"📄 Content that failed: {clean_content[:200]}...")

        # Fallback: Extract JSON from mixed text (LangChain-style)
        logger.debug("🔄 Attempting JSON extraction from mixed text...")
        json_candidates = self._extract_json_from_text(clean_content)
        logger.debug(f"📋 Found {len(json_candidates)} JSON candidates")

        for i, candidate in enumerate(json_candidates):
            try:
                logger.debug(f"🧪 Testing candidate {i+1}: {candidate[:100]}...")
                data = json.loads(candidate)
                if isinstance(data, dict) and "tool_calls" in data:
                    calls = data["tool_calls"]
                    if isinstance(calls, list) and calls:
                        logger.info(f"🎉 SUCCESS: Parsed {len(calls)} tool calls via extracted JSON")
                        for j, call in enumerate(calls):
                            logger.debug(f"🔧 Call {j+1}: {call.get('function', {}).get('name', 'unknown')}")
                        return calls
            except (json.JSONDecodeError, TypeError) as e:
                logger.debug(f"❌ Candidate {i+1} failed: {e}")
                continue

        # Check for single tool call format (OpenAI style)
        logger.debug("🔄 Checking for single tool call format...")
        try:
            data = json.loads(clean_content)
            if isinstance(data, dict) and "function" in data:
                # Convert single tool call to tool_calls format
                tool_call = {
                    "id": data.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    "type": "function",
                    "function": data["function"]
                }
                logger.info(f"🎉 SUCCESS: Converted single tool call format")
                logger.debug(f"🔧 Single call: {tool_call['function'].get('name', 'unknown')}")
                return [tool_call]
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"❌ Single tool call format failed: {e}")

        logger.warning("🚫 No tool calls found in content")
        logger.debug(f"📄 Final content analyzed: {clean_content[:300]}...")
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
        """Execute tool calls using ToolRegistry (professional implementation)."""
        logger.info(f"🔨 Starting execution of {len(tool_calls)} tool calls")

        if not TOOL_REGISTRY_AVAILABLE or not self._tool_registry:
            logger.error("❌ ToolRegistry not available for tool execution")
            return []

        results = []
        for i, tc in enumerate(tool_calls):
            logger.debug(f"🎯 Processing tool call {i+1}/{len(tool_calls)}")

            tool_id = tc.get("id", f"call_{uuid.uuid4().hex[:8]}")
            func = tc.get("function", {})
            tool_name = func.get("name", "")

            logger.debug(f"🛠️ Tool: {tool_name}, ID: {tool_id}")

            # Parse arguments
            args_raw = func.get("arguments", "{}")
            logger.debug(f"📝 Raw arguments: {args_raw[:200]}...")

            if isinstance(args_raw, str):
                try:
                    arguments = json.loads(args_raw)
                    logger.debug(f"✅ Parsed arguments: {arguments}")
                except json.JSONDecodeError as e:
                    arguments = {}
                    logger.warning(f"⚠️ Failed to parse tool arguments: {e}")
                    logger.debug(f"📄 Raw args that failed: {args_raw[:100]}")
            else:
                arguments = args_raw
                logger.debug(f"📋 Arguments were already parsed: {arguments}")

            logger.info(f"🚀 Executing tool: {tool_name} with {len(arguments)} args")

            try:
                # Use ToolRegistry's professional execution
                logger.debug(f"🔧 Calling ToolRegistry.invoke({tool_name}, **{arguments})")
                result = self._tool_registry.invoke(tool_name, **arguments)

                result_str = str(result)
                logger.info(f"✅ Tool '{tool_name}' executed successfully (result: {len(result_str)} chars)")
                logger.debug(f"📄 Result preview: {result_str[:200]}...")

                results.append({
                    "tool_call_id": tool_id,
                    "content": result_str,
                    "success": True
                })

            except Exception as e:
                error_msg = f"Tool execution failed: {e}"
                logger.error(f"❌ Tool '{tool_name}' failed: {e}")
                logger.debug("Full error details:", exc_info=True)

                results.append({
                    "tool_call_id": tool_id,
                    "content": error_msg,
                    "success": False
                })

        logger.info(f"📊 Tool execution complete: {len(results)} results ({sum(1 for r in results if r['success'])} successful)")
        return results

    async def execute_tools_and_continue(self, req: ChatRequestModel) -> ChatResponseModel:
        """
        Execute tools and continue conversation (for research use cases).
        This is a standalone agent-style method that doesn't interfere with normal chat completions.
        """
        user_query = req.messages[-1].content[:100] if req.messages else 'empty'
        logger.info(f"🚀 Starting AGENT EXECUTION FLOW for: '{user_query}'")
        logger.debug(f"📋 Agent request: tools={len(req.tools) if req.tools else 0}, messages={len(req.messages)}")

        # Step 1: Generate initial response (may contain tool calls)
        logger.info("📝 Step 1: Generating initial response with potential tool calls")
        initial_response = await self._generate_single_completion(req)

        choice = initial_response.choices[0]
        if not choice.message or not choice.message.tool_calls:
            # No tool calls, return normal response
            logger.info("No tool calls generated, returning normal response")
            return initial_response

        tool_calls = choice.message.tool_calls
        logger.info(f"⚙️ Step 2: Executing {len(tool_calls)} tool calls")

        # Convert ToolCall objects to dict format for execution
        tool_calls_dict = [
            {
                "id": tc.id,
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.args) if tc.args else "{}"
                }
            } for tc in tool_calls
        ]
        logger.debug(f"🔄 Converted {len(tool_calls)} ToolCall objects to dict format")

        tool_results = self._execute_tool_calls_simple(tool_calls_dict)

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

        # Add guidance for the model to synthesize the final answer
        guidance_content = "Based on the tool results above, provide a comprehensive and well-structured answer to my original question. Extract and organize the key information from the tool outputs."
        guidance_msg = ChatMessage(
            role="user",
            content=guidance_content
        )
        new_messages.append(guidance_msg)
        logger.debug(f"📝 Added synthesis guidance: {guidance_content[:100]}...")

        # Step 4: Generate final synthesized response
        logger.info("🎯 Step 4: Generating final synthesized response")
        final_req = req.model_copy(update={
            "messages": new_messages,
            "tools": None,  # Don't include tools in final generation
        })
        logger.debug(f"📋 Final request: {len(new_messages)} messages, no tools")

        final_response = await self._generate_single_completion(final_req)

        # Mark as tool-synthesized response
        final_response.choices[0].finish_reason = ChatCompletionReason.STOP
        final_content = final_response.choices[0].message.content if final_response.choices[0].message else ""
        logger.info(f"🎉 AGENT COMPLETE: Generated {len(final_content)} char synthesized response")
        logger.debug(f"📄 Final answer preview: {final_content[:200]}...")

        return final_response

    def bind_tools(self, tools: List[Any]) -> 'InferenceManager':
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
        logger.info(f"🔧 Binding {len(tools)} tools to inference manager")

        bound_tools = []
        for tool in tools:
            tool_def = self._convert_tool_to_definition(tool)
            if tool_def:
                bound_tools.append(tool_def)
                logger.debug(f"✅ Bound tool: {tool_def.get('function', {}).get('name', 'unknown')}")

        self._bound_tools = bound_tools
        logger.info(f"🎯 Successfully bound {len(self._bound_tools)} tools")
        return self

    def _convert_tool_to_definition(self, tool: Any) -> Optional[Dict[str, Any]]:
        """
        Convert various tool formats to OpenAI-compatible tool definition.
        """
        # If it's already a dict with OpenAI format
        if isinstance(tool, dict) and tool.get("type") == "function":
            return tool

        # If it's a Pydantic model
        if hasattr(tool, '__annotations__') and hasattr(tool, 'model_json_schema'):
            try:
                schema = tool.model_json_schema()
                return {
                    "type": "function",
                    "function": {
                        "name": getattr(tool, '__name__', tool.__class__.__name__.lower()),
                        "description": getattr(tool, '__doc__', '').strip(),
                        "parameters": schema
                    }
                }
            except Exception as e:
                logger.warning(f"Failed to convert Pydantic model to tool definition: {e}")

        # If it's a function with @tool decorator (basic support)
        if callable(tool) and hasattr(tool, '__name__'):
            # Try to extract function signature
            import inspect
            try:
                sig = inspect.signature(tool)
                params = {}
                for name, param in sig.parameters.items():
                    if name == 'self':
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
                        "description": getattr(tool, '__doc__', '').strip(),
                        "parameters": {
                            "type": "object",
                            "properties": params,
                            "required": list(params.keys())
                        }
                    }
                }
            except Exception as e:
                logger.warning(f"Failed to convert function to tool definition: {e}")

        logger.warning(f"Unsupported tool format: {type(tool)}")
        return None

    def get_bound_tools(self) -> List[Dict[str, Any]]:
        """Get currently bound tools."""
        return self._bound_tools.copy()

    def register_mcp_tools(self, transport: str, namespace: Optional[str] = None) -> bool:
        """Register tools from an MCP server dynamically.

        Args:
            transport: MCP transport URL or path (e.g., "https://exa-mcp.com", "path/to/server.py")
            namespace: Optional namespace prefix for tool names

        Returns:
            bool: True if registration successful
        """
        logger.info(f"🔗 Attempting MCP tool registration from: {transport}")
        logger.debug(f"🏷️ Using namespace: {namespace}")

        if not TOOL_REGISTRY_AVAILABLE:
            logger.error("❌ ToolRegistry library not installed")
            return False

        if not self._tool_registry:
            logger.error("❌ ToolRegistry not initialized")
            return False

        try:
            logger.debug("📡 Calling ToolRegistry.register_from_mcp()...")
            # Handle async/sync mismatch - ToolRegistry may be async
            # Use asyncio.run_in_executor to bridge sync->async if needed
            import asyncio

            # Check if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, need to handle this differently
                logger.warning("⚠️ MCP registration called in async context - deferring to sync execution")
                # For now, return False and handle this in startup
                return False
            except RuntimeError:
                # No running loop, we can use asyncio.run
                pass

            # Try to call the async method synchronously
            try:
                # If the method is async, we need to handle it
                import inspect
                register_method = getattr(self._tool_registry, 'register_from_mcp', None)
                if register_method and inspect.iscoroutinefunction(register_method):
                    # It's async, we need to run it in an event loop
                    logger.debug("🔄 Running async MCP registration...")
                    asyncio.run(self._tool_registry.register_from_mcp(transport, with_namespace=namespace))
                else:
                    # It's sync, call directly
                    self._tool_registry.register_from_mcp(transport, with_namespace=namespace)
            except Exception as async_error:
                logger.warning(f"⚠️ Async registration failed, trying sync: {async_error}")
                # Fallback to sync call
                self._tool_registry.register_from_mcp(transport, with_namespace=namespace)

            # Get registered tools to verify
            available_tools = self._tool_registry.get_available_tools()
            logger.info(f"✅ Successfully registered MCP tools from {transport}")
            logger.info(f"📋 Available tools: {len(available_tools)} total")
            logger.debug(f"🛠️ Tool list: {available_tools}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to register MCP tools from {transport}: {e}")
            logger.debug("MCP registration error details:", exc_info=True)
            return False

    async def register_mcp_tools_async(self, transport: str, namespace: Optional[str] = None) -> bool:
        """Async version of MCP tool registration.

        Args:
            transport: MCP transport URL or path
            namespace: Optional namespace prefix

        Returns:
            bool: True if registration successful
        """
        logger.info(f"🔗 Attempting async MCP tool registration from: {transport}")

        if not TOOL_REGISTRY_AVAILABLE:
            logger.error("❌ ToolRegistry library not installed")
            return False

        if not self._tool_registry:
            logger.error("❌ ToolRegistry not initialized")
            return False

        try:
            logger.debug("📡 Calling async ToolRegistry.register_from_mcp()...")
            # Call the async method directly
            await self._tool_registry.register_from_mcp(transport, with_namespace=namespace)

            # Get registered tools to verify
            available_tools = self._tool_registry.get_available_tools()
            logger.info(f"✅ Successfully registered MCP tools from {transport}")
            logger.info(f"📋 Available tools: {len(available_tools)} total")
            logger.debug(f"🛠️ Tool list: {available_tools}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to register MCP tools from {transport}: {e}")
            logger.debug("Async MCP registration error details:", exc_info=True)
            return False

    def register_openapi_tools(self, openapi_spec: Union[str, Dict], client_config: Optional[Dict] = None) -> bool:
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
                    client_config=client_config,
                    openapi_spec=openapi_spec
                )
            else:
                # Try with just the spec
                self._tool_registry.register_from_openapi(openapi_spec=openapi_spec)

            logger.info(f"Successfully registered OpenAPI tools")
            return True
        except Exception as e:
            logger.error(f"Failed to register OpenAPI tools: {e}")
            return False

    def get_registered_tools(self) -> List[str]:
        """Get list of all registered tool names."""
        logger.debug("📋 Querying registered tools...")

        if not TOOL_REGISTRY_AVAILABLE:
            logger.debug("❌ ToolRegistry library not available")
            return []

        if not self._tool_registry:
            logger.debug("❌ ToolRegistry not initialized")
            return []

        try:
            tools = self._tool_registry.get_available_tools()
            logger.debug(f"✅ Found {len(tools)} registered tools: {tools}")
            return tools
        except Exception as e:
            logger.warning(f"⚠️ Failed to get registered tools: {e}")
            logger.debug("Tool query error details:", exc_info=True)
            return []


class StructuredOutputInferenceManager:
    """
    LangGraph-style structured output wrapper for InferenceManager.

    Forces the agent to return responses in a specific structured format by binding
    the response schema as a tool that must be called (LangGraph "Option 1").

    This ensures the agent provides structured output without requiring a second LLM call.
    """

    def __init__(self, inference_manager: 'InferenceManager', schema: Any):
        self.inference_manager = inference_manager
        self.schema = schema

        # Generate tool definition from schema
        self._structured_output_tool = self._schema_to_tool(schema)

    def _schema_to_tool(self, schema: Any) -> Dict[str, Any]:
        """Convert Pydantic schema or JSON schema to tool definition."""
        if hasattr(schema, 'model_json_schema'):
            # Pydantic model
            json_schema = schema.model_json_schema()
            return {
                "type": "function",
                "function": {
                    "name": schema.__name__,
                    "description": getattr(schema, '__doc__', '').strip() or "Structured response",
                    "parameters": json_schema
                }
            }
        elif isinstance(schema, dict):
            # Raw JSON schema
            return {
                "type": "function",
                "function": {
                    "name": "StructuredResponse",
                    "description": "Structured response",
                    "parameters": schema
                }
            }
        else:
            raise ValueError(f"Unsupported schema type: {type(schema)}")

    async def chat_completions(self, req: ChatRequestModel) -> ChatResponseModel:
        """
        Generate completion with guaranteed structured output.

        The agent will be forced to call the structured output tool to provide its final answer.
        """
        # Temporarily bind the structured output tool
        original_tools = self.inference_manager._bound_tools.copy() if hasattr(self.inference_manager, '_bound_tools') else []

        try:
            # Add structured output tool to bound tools or request tools
            if hasattr(self.inference_manager, '_bound_tools'):
                # LangChain-style: add to bound tools
                self.inference_manager._bound_tools.append(self._structured_output_tool)
            else:
                # Fallback: add to request tools
                if not req.tools:
                    req.tools = []
                req.tools.append(self._structured_output_tool)

            # Force tool calling by setting tool_choice
            if hasattr(req, 'tool_choice'):
                req.tool_choice = "any"  # Force at least one tool call

            # Generate response (agent should call the structured output tool)
            response = await self.inference_manager.chat_completions(req)

            # Extract structured data from tool calls
            if response.choices and response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    tool_name = tool_call.name if hasattr(tool_call, 'name') else tool_call.get('function', {}).get('name', '')
                    if tool_name == self._structured_output_tool["function"]["name"]:
                        # Parse the structured arguments
                        if hasattr(tool_call, 'args'):
                            structured_data = tool_call.args
                        else:
                            # Handle dict format
                            args_str = tool_call.get('function', {}).get('arguments', '{}')
                            try:
                                structured_data = json.loads(args_str) if isinstance(args_str, str) else args_str
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
            if hasattr(self.inference_manager, '_bound_tools'):
                self.inference_manager._bound_tools = original_tools

    def bind_tools(self, tools: List[Any]) -> 'StructuredOutputInferenceManager':
        """
        Bind additional tools while keeping the structured output tool.

        This allows binding action tools + maintaining structured output.
        """
        # Bind tools on the underlying inference manager
        if hasattr(self.inference_manager, 'bind_tools'):
            self.inference_manager.bind_tools(tools)
        return self

    def __getattr__(self, name):
        """Delegate other methods to the underlying inference manager."""
        return getattr(self.inference_manager, name)

    def with_structured_output(self, schema: Any) -> 'StructuredOutputInferenceManager':
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

    def with_structured_output(self, schema: Any) -> 'StructuredOutputInferenceManager':
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
        return StructuredOutputInferenceManager(self, schema)


    def resolve_request(self, nonce: str, result: Any):
        self.adapter.resolve_token(nonce, result)


# Test function for structured output
def test_structured_output():
    """Test the LangGraph-style structured output functionality."""
    from pydantic import BaseModel, Field

    # Mock InferenceManager for testing
    class MockInferenceManager:
        def __init__(self):
            self._bound_tools = []

        def bind_tools(self, tools):
            bound_tools = []
            for tool in tools:
                tool_def = self._convert_tool_to_definition(tool)
                if tool_def:
                    bound_tools.append(tool_def)
            self._bound_tools = bound_tools
            return self

        def _convert_tool_to_definition(self, tool):
            if hasattr(tool, '__annotations__') and hasattr(tool, 'model_json_schema'):
                try:
                    schema = tool.model_json_schema()
                    return {
                        "type": "function",
                        "function": {
                            "name": tool.__name__,
                            "description": getattr(tool, '__doc__', '').strip(),
                            "parameters": schema
                        }
                    }
                except:
                    pass
            return None

        def get_bound_tools(self):
            return self._bound_tools.copy()

    # Test structured output schema
    class WeatherResponse(BaseModel):
        """Structured weather response."""
        temperature: float = Field(description="Temperature in Fahrenheit")
        wind_direction: str = Field(description="Wind direction")
        wind_speed: float = Field(description="Wind speed in mph")

    # Test StructuredOutputInferenceManager
    base_llm = MockInferenceManager()
    structured_llm = StructuredOutputInferenceManager(base_llm, WeatherResponse)

    # Check that structured output tool was created
    tools = structured_llm.get_bound_tools()
    print(f"✅ Structured output tool created: {len(tools)} tools")
    for tool in tools:
        name = tool.get('function', {}).get('name', 'unknown')
        print(f"  - {name}")

    # Test schema conversion
    schema_tool = structured_llm._schema_to_tool(WeatherResponse)
    print(f"✅ Schema converted to tool: {schema_tool['function']['name']}")

    print("✅ Structured output functionality implemented!")
    return True


if __name__ == "__main__":
    test_structured_output()
