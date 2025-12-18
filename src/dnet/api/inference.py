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
from .mcp_tools import MCPToolProvider
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
        mcp_provider: Optional[MCPToolProvider] = None,
    ):
        self.cluster_manager = cluster_manager
        self.model_manager = model_manager
        self.grpc_port = grpc_port
        self.adapter = adapter
        self.mcp_provider = mcp_provider

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

    def set_mcp_provider(self, provider: MCPToolProvider) -> None:
        """Set MCP tool provider for external tool execution."""
        self.mcp_provider = provider
        if provider.enabled:
            logger.info(f"MCP provider set with {len(provider.get_tool_names())} tools")

    async def _execute_tool_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[ChatMessage]:
        """Execute tool calls via MCP and return tool result messages.

        Args:
            tool_calls: List of tool calls in OpenAI format

        Returns:
            List of ChatMessage with role="tool" containing results
        """
        results = []

        for tc in tool_calls:
            tool_id = tc.get("id", f"call_{uuid.uuid4().hex[:8]}")
            func = tc.get("function", {})
            tool_name = func.get("name", "")
            
            # Parse arguments (may be string or dict)
            args_raw = func.get("arguments", "{}")
            if isinstance(args_raw, str):
                try:
                    arguments = json.loads(args_raw)
                except json.JSONDecodeError:
                    arguments = {}
                    logger.warning(f"Failed to parse tool arguments: {args_raw[:100]}")
            else:
                arguments = args_raw

            # Execute via MCP
            if self.mcp_provider and self.mcp_provider.enabled:
                result_text = await self.mcp_provider.execute(tool_name, arguments)
            else:
                result_text = f"Error: MCP provider not available for tool '{tool_name}'"
                logger.warning(result_text)

            # Create tool result message (OpenAI format)
            results.append(ChatMessage(
                role="tool",
                name=tool_name,
                content=result_text,
                tool_call_id=tool_id,
            ))

            logger.info(f"Tool '{tool_name}' executed, result: {len(result_text)} chars")

        return results

    def _format_tools_for_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """
        Format tools for prompt injection.

        The model needs to know what tools are available. This creates a
        description that gets injected into the system message.
        """
        if not tools:
            logger.debug("No tools provided for prompt formatting")
            return ""

        logger.debug(f"Formatting {len(tools)} tools for prompt injection")
        tools_description = json.dumps(tools, indent=2)

        return f"""

You have access to the following tools.

CRITICAL INSTRUCTIONS:
- ONLY use tools when the user's request SPECIFICALLY requires the tool's functionality
- For greetings (hi, hello, how are you), casual chat, or general knowledge questions - respond with NORMAL TEXT, do NOT call any tools
- For requests that clearly match a tool's purpose (e.g., "what's the weather" → use get_weather) - use the appropriate tool

When you DO need to use a tool, respond with a JSON object in this exact format:
{{
  "tool_calls": [
    {{
      "id": "call_<unique_id>",
      "type": "function",
      "function": {{
        "name": "<function_name>",
        "arguments": "{{\\"param1\\": \\"value1\\", \\"param2\\": 123}}"
      }}
    }}
  ]
}}

Available tools:
{tools_description}

Rules:
- The "arguments" field MUST be a valid JSON string with double quotes
- Use the exact function names from the tools list above
- Include all required parameters
- If unsure whether to use a tool, respond with normal text instead
"""

    def _build_tool_call_schema(self, tools: List[Dict[str, Any]]) -> Optional[str]:
        """
        Build JSON schema for tool calls based on available tools.

        This schema is used by Outlines to constrain generation to valid
        tool call JSON. The function names are restricted to an enum of
        available tool names.

        Returns None if no valid tools found (instead of empty string).
        """
        tool_names = []
        for t in tools:
            try:
                if t.get("type") == "function" and "function" in t:
                    func = t["function"]
                    if isinstance(func, dict) and "name" in func:
                        tool_names.append(func["name"])
                    else:
                        logger.warning(f"Tool missing 'name' in function definition: {t}")
            except (KeyError, TypeError) as e:
                logger.warning(f"Malformed tool definition, skipping: {e}")
                continue

        if not tool_names:
            logger.warning("No valid tool names extracted from tools list")
            return None

        logger.debug(f"Building tool call schema for tools: {tool_names}")

        schema = {
            "type": "object",
            "properties": {
                "tool_calls": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "type": {"const": "function"},
                            "function": {
                                "type": "object",
                                "properties": {
                                    "name": {"enum": tool_names},
                                    "arguments": {"type": "string"},
                                },
                                "required": ["name", "arguments"],
                            },
                        },
                        "required": ["id", "type", "function"],
                    },
                }
            },
            "required": ["tool_calls"],
        }

        schema_str = json.dumps(schema)
        logger.debug(f"Generated tool call schema: {schema_str[:200]}...")
        return schema_str

    async def generate_stream(self, req: ChatRequestModel):
        """
        Generator for chat completion chunks.
        """
        if not self.model_manager.tokenizer:
            raise RuntimeError(
                "Inference manager not ready (ring not connected or tokenizer not loaded)"
            )

        tokenizer = self.model_manager.tokenizer

        # Prepare messages - inject tool descriptions if tools are provided
        messages_for_prompt = req.messages.copy()
        use_tool_grammar = False

        if req.tools and req.tool_choice not in [None, "none"]:
            logger.info(
                f"Tool calling enabled: {len(req.tools)} tools, tool_choice={req.tool_choice}"
            )
            tools_prompt = self._format_tools_for_prompt(req.tools)

            # Find system message or prepend one
            has_system = any(m.role == "system" for m in messages_for_prompt)

            if has_system:
                # Append tools to existing system message
                for i, msg in enumerate(messages_for_prompt):
                    if msg.role == "system":
                        messages_for_prompt[i] = ChatMessage(
                            role="system", content=(msg.content or "") + tools_prompt
                        )
                        logger.debug("Appended tool descriptions to existing system message")
                        break
            else:
                # Prepend system message with tools
                messages_for_prompt.insert(
                    0, ChatMessage(role="system", content=tools_prompt.strip())
                )
                logger.debug("Created new system message with tool descriptions")

            # IMPORTANT: Only apply grammar constraint for "required"
            # For "auto", let model decide freely whether to call tools or respond with text
            # This is the production-standard approach used by vLLM, OpenAI, etc.
            if req.tool_choice == "required":
                use_tool_grammar = True
                logger.debug("tool_choice='required': applying Outlines constraint")
            else:
                # tool_choice is "auto" or specific function - don't force grammar
                # Model can choose to call tools or respond with text
                use_tool_grammar = False
                logger.debug(f"tool_choice='{req.tool_choice}': grammar constraint disabled, model decides")
        elif req.tools and req.tool_choice == "none":
            logger.debug("Tools provided but tool_choice='none', skipping tool grammar")

        try:
            if (
                hasattr(tokenizer, "chat_template")
                and tokenizer.chat_template is not None
            ):
                message_dicts = [
                    {"role": m.role, "content": m.content or ""}
                    for m in messages_for_prompt
                ]
                prompt_text = tokenizer.apply_chat_template(
                    message_dicts,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            else:
                logger.debug("No chat template available, using basic prompt format")
                prompt_text = (
                    "\n".join(m.content or "" for m in messages_for_prompt)
                    + "\nAssistant:"
                )
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}, using fallback format")
            prompt_text = (
                "\n".join(m.content or "" for m in messages_for_prompt) + "\nAssistant:"
            )

        prompt_tokens = tokenizer.encode(prompt_text)
        prompt_array = mx.array(prompt_tokens)

        stop_id_sequences = []
        if req.stop:
            for stop_word in req.stop:
                stop_id_sequences.append(
                    tokenizer.encode(stop_word, add_special_tokens=False)
                )

        # Get grammar JSON schema - tool calling takes priority
        grammar_json_schema = None

        if use_tool_grammar and req.tools:
            # Use tool call schema for constrained generation
            tool_schema = self._build_tool_call_schema(req.tools)
            if tool_schema:
                grammar_json_schema = tool_schema
                logger.info(
                    f"Using Outlines tool call schema for {len(req.tools)} tools"
                )
            else:
                # Failed to build schema - disable tool grammar but continue
                logger.warning(
                    "Failed to build tool call schema, proceeding without grammar constraint"
                )
                use_tool_grammar = False
        elif hasattr(req, "grammar_json_schema") and req.grammar_json_schema:
            grammar_json_schema = req.grammar_json_schema
            logger.debug("Using user-provided grammar_json_schema")
        elif hasattr(req, "response_format") and req.response_format:
            # Support OpenAI-style response_format with JSON schema
            if isinstance(req.response_format, dict):
                if "schema" in req.response_format:
                    grammar_json_schema = json.dumps(req.response_format["schema"])
                    logger.debug("Using response_format schema for grammar")
                elif (
                    "type" in req.response_format
                    and req.response_format["type"] == "json_object"
                ):
                    grammar_json_schema = json.dumps({"type": "object"})
                    logger.debug("Using basic JSON object grammar for response_format")

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
            
            # Debug logging for grammar termination
            if getattr(result, "grammar_terminated", False):
                logger.info(
                    f"Grammar terminated - token_id={token}, delta_text='{delta_text}', "
                    f"full_text='{full_text[:100]}...', tokens_count={len(tokens)}"
                )

            # Yield chunk
            logger.debug(
                f"Yielding chunk: token_id={token}, delta_text='{delta_text}', "
                f"full_text='{full_text[:100]}...', last_text_len={last_text_len}"
            )
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

            # Check Outlines' is_terminated() signal from the shard
            # This is the proper way to detect grammar completion
            if getattr(result, "grammar_terminated", False):
                logger.info("Grammar terminated signal received from shard")
                if use_tool_grammar:
                    completion_reason = ChatCompletionReason.TOOL_CALLS
                else:
                    completion_reason = ChatCompletionReason.STOP
                break

            y = mx.array([token], dtype=mx.int32)

        detokenizer.finalize()
        final_text = detokenizer.text

        # Parse tool calls from generated text
        # - For tool_choice="required" with grammar: always parse
        # - For tool_choice="auto" without grammar: attempt to parse if contains tool_calls JSON
        tool_calls = None
        
        # For auto mode, check if output contains tool_calls JSON (may be after <think> tags)
        has_tool_calls_json = '"tool_calls"' in final_text and "{" in final_text
        should_attempt_parse = use_tool_grammar or (
            req.tools and req.tool_choice == "auto" and has_tool_calls_json
        )

        if should_attempt_parse and final_text:
            logger.debug(f"Attempting to parse tool call output (length={len(final_text)}, grammar={use_tool_grammar})")
            clean_text = final_text.strip()

            # Remove any trailing special tokens (safety fallback)
            for eos_pattern in ["<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>"]:
                if eos_pattern in clean_text:
                    clean_text = clean_text.split(eos_pattern)[0].strip()
                    break
            
            # Handle Qwen3 <think> tags - extract content after </think>
            if "</think>" in clean_text:
                # Get content after the LAST </think> tag
                clean_text = clean_text.split("</think>")[-1].strip()
                logger.debug("Extracted content after </think> tags")

            try:
                parsed = json.loads(clean_text)
                if isinstance(parsed, dict) and "tool_calls" in parsed:
                    tool_calls = parsed["tool_calls"]
                    if tool_calls and isinstance(tool_calls, list):
                        completion_reason = ChatCompletionReason.TOOL_CALLS
                        tool_names = [
                            tc.get("function", {}).get("name", "unknown")
                            for tc in tool_calls
                            if isinstance(tc, dict)
                        ]
                        logger.info(
                            f"Parsed {len(tool_calls)} tool call(s): {tool_names}"
                        )
                    else:
                        logger.debug(f"tool_calls is empty or invalid, treating as text response")
                        tool_calls = None
                elif use_tool_grammar:
                    # Grammar was enforced but no tool_calls - this shouldn't happen
                    logger.warning(
                        f"Grammar enforced but JSON missing 'tool_calls' key: {type(parsed)}"
                    )
                else:
                    # For auto mode, valid JSON without tool_calls is fine - just text response
                    logger.debug("Model output JSON without tool_calls, treating as text response")
            except json.JSONDecodeError as e:
                if use_tool_grammar:
                    # Grammar was enforced, JSON parse failure is unexpected
                    logger.error(f"Failed to parse tool call JSON (grammar was enforced): {e}")
                    logger.debug(f"Raw output: {clean_text[:300]}...")
                else:
                    # For auto mode, non-JSON output is expected when model chooses not to use tools
                    logger.debug("Model chose to respond with text instead of tool calls")

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

        # Build final message - include tool_calls if present
        final_message = ChatMessage(
            role="assistant",
            content=None if tool_calls else final_text,
            tool_calls=tool_calls,
        )

        # Final chunk with finish reason
        # Note: Use delta=None to avoid duplication in chat_completions accumulation
        # The delta should only contain incremental content, not the full message
        yield ChatResponseModel(
            id=nonce,
            choices=[
                ChatChoice(
                    index=0,
                    delta=None,  # Final chunk has no delta to avoid re-accumulating full text
                    message=final_message,  # Use message field for final complete message
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

    async def chat_completions(
        self, 
        req: ChatRequestModel, 
        execute_tools: bool = True,
        max_tool_rounds: int = 5,
    ) -> ChatResponseModel:
        """
        Handles chat completion request (non-streaming) with optional tool execution.

        Args:
            req: Chat request
            execute_tools: If True, automatically execute tool calls via MCP
            max_tool_rounds: Maximum tool execution iterations (prevents infinite loops)
        """
        # Inject MCP tools if available and no tools provided in request
        working_req = req
        if self.mcp_provider and self.mcp_provider.enabled and not req.tools:
            mcp_tools = self.mcp_provider.get_tools()
            if mcp_tools:
                # Create new request with MCP tools
                working_req = ChatRequestModel(
                    messages=req.messages,
                    model=req.model,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    stop=req.stop,
                    stream=req.stream,
                    logprobs=req.logprobs,
                    top_logprobs=req.top_logprobs,
                    repetition_penalty=req.repetition_penalty,
                    tools=mcp_tools,
                    tool_choice=req.tool_choice or "auto",
                )
                logger.info(f"Injected {len(mcp_tools)} MCP tools into request")

        # Tool execution loop
        current_messages = list(working_req.messages)
        tool_round = 0

        while tool_round < max_tool_rounds:
            tool_round += 1

            # Generate response
            response = await self._generate_single_completion(
                ChatRequestModel(
                    messages=current_messages,
                    model=working_req.model,
                    temperature=working_req.temperature,
                    max_tokens=working_req.max_tokens,
                    top_p=working_req.top_p,
                    top_k=working_req.top_k,
                    stop=working_req.stop,
                    stream=False,
                    logprobs=working_req.logprobs,
                    top_logprobs=working_req.top_logprobs,
                    repetition_penalty=working_req.repetition_penalty,
                    tools=working_req.tools,
                    tool_choice=working_req.tool_choice,
                )
            )

            # Check if we need to execute tools
            choice = response.choices[0]
            if (
                execute_tools
                and choice.finish_reason == ChatCompletionReason.TOOL_CALLS
                and choice.message
                and choice.message.tool_calls
                and self.mcp_provider
                and self.mcp_provider.enabled
            ):
                tool_calls = choice.message.tool_calls
                logger.info(f"Tool round {tool_round}: executing {len(tool_calls)} tool(s)")

                # Add assistant message with tool calls
                current_messages.append(choice.message)

                # Execute tools and add results
                tool_results = await self._execute_tool_calls(tool_calls)
                current_messages.extend(tool_results)

                # Continue loop for next response
                continue

            # No tool calls or tool execution disabled - return response
            return response

        # Max rounds reached
        logger.warning(f"Max tool rounds ({max_tool_rounds}) reached")
        return response

    async def _generate_single_completion(self, req: ChatRequestModel) -> ChatResponseModel:
        """Generate a single completion without tool execution loop."""
        full_content = ""
        tokens = []
        token_logprobs = []
        top_logprobs_list = []
        completion_reason = ChatCompletionReason.LENGTH
        nonce = ""
        metrics_dict = None
        usage = None
        tool_calls = None
        final_message_from_chunk = None

        async for chunk in self.generate_stream(req):
            nonce = chunk.id
            choice = chunk.choices[0]

            if choice.message:
                final_message_from_chunk = choice.message
                if final_message_from_chunk.tool_calls:
                    tool_calls = final_message_from_chunk.tool_calls
            elif choice.delta:
                if choice.delta.content:
                    full_content += choice.delta.content
                if choice.delta.tool_calls:
                    tool_calls = choice.delta.tool_calls

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
        if final_message_from_chunk is not None:
            final_message = final_message_from_chunk
        else:
            final_message = ChatMessage(
                role="assistant",
                content=None if tool_calls else full_content,
                tool_calls=tool_calls,
            )

        # Log completion
        if tool_calls:
            tool_names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
            logger.info(f"Completion with tool calls: {tool_names}")
        else:
            logger.debug(f"Completion finished, content_len={len(full_content)}")

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
