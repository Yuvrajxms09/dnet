"""Inference manager for dnet API server.

Handles chat completions with optional:
- Tool calling (prompt injection + grammar-constrained generation)
- MCP tool execution (server-side tool execution loop)
"""

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


# Prevents infinite loops from tool calling chains
DEFAULT_MAX_TOOL_ROUNDS = 10

DEFAULT_TOKEN_TIMEOUT_SECONDS = 600.0  # 10 minutes

# System message guidance when tool execution encounters errors
TOOL_EXECUTION_GUIDANCE_ERROR = (
    "Some tools encountered errors. Please inform the user about what went wrong "
    "in a clear, helpful way. If possible, suggest alternative approaches or what the user "
    "could try instead. Do not make up information if the tools failed."
)

# System message guidance when tool execution succeeds
TOOL_EXECUTION_GUIDANCE_SUCCESS = (
    "Based on the tool results above, provide a helpful, accurate response to the user's "
    "original question. Extract and present the key information from the tool outputs in a "
    "clear, organized manner. If the tool results contain specific data or structured "
    "information, present that information directly to the user."
)

# Optional MCP import - graceful degradation if not available
try:
    from .mcp_tools import MCPToolProvider
    MCP_AVAILABLE = True
except ImportError:
    MCPToolProvider = None
    MCP_AVAILABLE = False
    logger.debug("MCP tools module not available")


# =============================================================================
# Tool Execution Logging
# =============================================================================
# Helper for consistent tool execution logging with context.
# Includes key info in the message for readability, and adds structured fields
# via `extra` for log aggregation systems that support it (JSON formatters, etc.).
def _log_tool_execution(
    level: str,
    message: str,
    tool_name: Optional[str] = None,
    tool_id: Optional[str] = None,
    round_num: Optional[int] = None,
    duration_ms: Optional[float] = None,
    success: Optional[bool] = None,
    error_type: Optional[str] = None,
    result_size: Optional[int] = None,
) -> None:
    """
    Log tool execution event with context.
    
    Builds a readable message and includes structured fields for log aggregation.
    The message is always readable even without JSON formatters.
    
    Args:
        level: Log level ("info", "debug", "warning", "error")
        message: Base log message
        tool_name: Name of the tool being executed
        tool_id: Unique ID for this tool call
        round_num: Tool execution round number (1-indexed)
        duration_ms: Execution duration in milliseconds
        success: Whether execution succeeded
        error_type: Type of error if execution failed
        result_size: Size of result (e.g., character count)
    """
    # Build readable message with key context
    parts = [message]
    if round_num is not None:
        parts.append(f"round={round_num}")
    if tool_name:
        parts.append(f"tool={tool_name}")
    if duration_ms is not None:
        parts.append(f"duration={duration_ms:.2f}ms")
    if result_size is not None:
        parts.append(f"result_size={result_size}")
    if success is not None:
        parts.append(f"success={success}")
    if error_type:
        parts.append(f"error={error_type}")
    
    formatted_message = " | ".join(parts)
    
    # Build structured context for log aggregation systems
    # (only include non-None values to keep logs clean)
    context = {"component": "tool_execution"}
    if tool_name is not None:
        context["tool_name"] = tool_name
    if tool_id is not None:
        context["tool_id"] = tool_id
    if round_num is not None:
        context["round"] = round_num
    if duration_ms is not None:
        context["duration_ms"] = round(duration_ms, 2)
    if success is not None:
        context["success"] = success
    if error_type is not None:
        context["error_type"] = error_type
    if result_size is not None:
        context["result_size"] = result_size
    
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(formatted_message, extra=context)


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
        mcp_provider: Optional[Any] = None,  # MCPToolProvider, but Optional to avoid import issues
    ):
        self.cluster_manager = cluster_manager
        self.model_manager = model_manager
        self.grpc_port = grpc_port
        self.adapter = adapter
        self.mcp_provider = mcp_provider
        self._api_callback_addr: str = ""
        
        # Log MCP status
        if mcp_provider:
            mcp_enabled = getattr(mcp_provider, 'enabled', False)
            logger.info(f"InferenceManager initialized with MCP provider (enabled={mcp_enabled})")
        else:
            logger.debug("InferenceManager initialized without MCP provider")

    async def connect_to_ring(
        self, first_shard_ip: str, first_shard_port: int, api_callback_addr: str
    ) -> None:
        """
        `api_callback_addr` must be a reachable `host:port` from shards.
        For internet setups, this should be a public IP/DNS or overlay VPN IP.
        """
        await self.adapter.connect_first_shard(first_shard_ip, first_shard_port)
        self._api_callback_addr = api_callback_addr

    
    def _format_tools_for_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """Inject tools into system message using compact format.
        
        Uses compact format (name + description) in prompt to avoid memory issues
        with large tool schemas. Full schemas are still available for Outlines
        grammar constraint validation (used separately).
        """
        if not tools:
            return ""

        logger.debug(f"Injecting {len(tools)} tools into prompt (compact format)")
        
        # Compact format: just name + description (for model awareness)
        # Full schemas are used separately for grammar constraint validation
        tool_list = []
        for t in tools:
            if t.get("type") == "function" and "function" in t:
                func = t["function"]
                name = func.get("name", "unknown")
                desc = func.get("description", "")
                tool_list.append(f"- {name}: {desc}")
        
        tools_text = "\n".join(tool_list)

        return f"""

You have access to {len(tools)} tools. Use them ONLY when the user's request requires external data.
For greetings or general questions, respond normally without tools.

Available tools:
{tools_text}

To use a tool, respond with JSON:
{{"tool_calls": [{{"id": "call_1", "type": "function", "function": {{"name": "<tool_name>", "arguments": "{{\\"param\\": \\"value\\"}}"}}}}]}}
"""

    def _build_tool_call_schema(self, tools: List[Dict[str, Any]]) -> Optional[str]:
        """Build JSON schema for tool calls (used by Outlines for grammar constraint)."""
        tool_names = []
        for t in tools:
            try:
                if t.get("type") == "function" and "function" in t:
                    func = t["function"]
                    if isinstance(func, dict) and "name" in func:
                        tool_names.append(func["name"])
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

        return json.dumps(schema)

    # =========================================================================
    # MCP Tool Execution
    # =========================================================================

    async def _execute_tool_calls(
        self, tool_calls: List[Dict[str, Any]], round_num: Optional[int] = None
    ) -> List[ChatMessage]:
        """
        Execute tool calls via MCP and return tool result messages.
        
        Uses structured logging to enable tool execution trace analysis.
        """
        if not self.mcp_provider or not getattr(self.mcp_provider, 'enabled', False):
            _log_tool_execution(
                "warning",
                "MCP provider not available for tool execution",
                error_type="provider_unavailable",
            )
            return []

        results = []
        execution_start = time.perf_counter()
        
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
                    _log_tool_execution(
                        "warning",
                        f"Failed to parse tool arguments: {args_raw[:100]}",
                        tool_name=tool_name,
                        tool_id=tool_id,
                        round_num=round_num,
                        error_type="json_parse_error",
                    )
            else:
                arguments = args_raw

            _log_tool_execution(
                "info",
                f"Executing MCP tool: {tool_name}",
                tool_name=tool_name,
                tool_id=tool_id,
                round_num=round_num,
            )
            
            tool_start = time.perf_counter()
            try:
                result_text = await self.mcp_provider.execute(tool_name, arguments)
                tool_duration_ms = (time.perf_counter() - tool_start) * 1000.0
                
                _log_tool_execution(
                    "info",
                    f"Tool '{tool_name}' completed successfully",
                    tool_name=tool_name,
                    tool_id=tool_id,
                    round_num=round_num,
                    duration_ms=tool_duration_ms,
                    success=True,
                    result_size=len(result_text) if result_text else 0,
                )
            except Exception as e:
                tool_duration_ms = (time.perf_counter() - tool_start) * 1000.0
                error_type = type(e).__name__
                result_text = f"Error executing tool: {e}"
                
                _log_tool_execution(
                    "error",
                    f"Tool execution failed: {e}",
                    tool_name=tool_name,
                    tool_id=tool_id,
                    round_num=round_num,
                    duration_ms=tool_duration_ms,
                    success=False,
                    error_type=error_type,
                )

            results.append(ChatMessage(
                role="tool",
                name=tool_name,
                content=result_text,
                tool_call_id=tool_id,
            ))

        total_duration_ms = (time.perf_counter() - execution_start) * 1000.0
        _log_tool_execution(
            "info",
            f"Completed execution of {len(tool_calls)} tool call(s)",
            round_num=round_num,
            duration_ms=total_duration_ms,
            success=all(
                not (r.content and r.content.startswith("Error"))
                for r in results
            ),
        )

        return results

    # =========================================================================
    # Core Generation
    # =========================================================================

    async def generate_stream(self, req: ChatRequestModel):
        """Generator for chat completion chunks."""
        logger.debug(f"generate_stream called: model={req.model}, tools={len(req.tools) if req.tools else 0}")
        
        if not self.model_manager.tokenizer:
            raise RuntimeError(
                "Inference manager not ready (ring not connected or tokenizer not loaded)"
            )

        tokenizer = self.model_manager.tokenizer

        # Prepare messages - inject tool descriptions if tools are provided
        messages_for_prompt = req.messages.copy()
        use_tool_grammar = False

        if req.tools and req.tool_choice not in [None, "none"]:
            logger.debug(f"Tool calling enabled: {len(req.tools)} tools, tool_choice={req.tool_choice}")
            tools_prompt = self._format_tools_for_prompt(req.tools)

            # Find system message or prepend one
            has_system = any(m.role == "system" for m in messages_for_prompt)

            if has_system:
                for i, msg in enumerate(messages_for_prompt):
                    if msg.role == "system":
                        messages_for_prompt[i] = ChatMessage(
                            role="system", content=(msg.content or "") + tools_prompt
                        )
                        break
            else:
                messages_for_prompt.insert(
                    0, ChatMessage(role="system", content=tools_prompt.strip())
                )

            # Apply grammar constraint only for "required"
            # For "auto", let model decide - if it calls tools, we'll parse them
            if req.tool_choice == "required":
                use_tool_grammar = True
                logger.debug("tool_choice='required': applying grammar constraint")
            else:
                # tool_choice is "auto" or specific function - don't force grammar
                # Model can choose to call tools or respond with text naturally
                use_tool_grammar = False
                logger.debug(f"tool_choice='{req.tool_choice}': no grammar constraint, model decides")

        # Build prompt
        try:
            if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
                # Convert messages to dict format
                message_dicts = []
                for m in messages_for_prompt:
                    msg_dict = {"role": m.role, "content": m.content or ""}
                    # Include tool-related fields for proper chat template
                    if m.role == "tool" and m.name:
                        msg_dict["name"] = m.name
                    if m.role == "tool" and m.tool_call_id:
                        msg_dict["tool_call_id"] = m.tool_call_id
                    if m.role == "assistant" and m.tool_calls:
                        msg_dict["tool_calls"] = m.tool_calls
                    message_dicts.append(msg_dict)
                
                prompt_text = tokenizer.apply_chat_template(
                    message_dicts,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            else:
                prompt_text = "\n".join(m.content or "" for m in messages_for_prompt) + "\nAssistant:"
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}, using fallback")
            prompt_text = "\n".join(m.content or "" for m in messages_for_prompt) + "\nAssistant:"

        prompt_tokens = tokenizer.encode(prompt_text)
        prompt_array = mx.array(prompt_tokens)

        stop_id_sequences = []
        if req.stop:
            for stop_word in req.stop:
                stop_id_sequences.append(
                    tokenizer.encode(stop_word, add_special_tokens=False)
                )

        # Get grammar JSON schema
        grammar_json_schema = None

        if use_tool_grammar and req.tools:
            tool_schema = self._build_tool_call_schema(req.tools)
            if tool_schema:
                grammar_json_schema = tool_schema
                logger.info(f"Using Outlines tool call schema for {len(req.tools)} tools")
            else:
                use_tool_grammar = False
        elif hasattr(req, "grammar_json_schema") and req.grammar_json_schema:
            grammar_json_schema = req.grammar_json_schema
        elif hasattr(req, "response_format") and req.response_format:
            if isinstance(req.response_format, dict):
                if "schema" in req.response_format:
                    grammar_json_schema = json.dumps(req.response_format["schema"])
                elif req.response_format.get("type") == "json_object":
                    grammar_json_schema = json.dumps({"type": "object"})

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
                min_tokens_to_keep=req.min_tokens_to_keep if hasattr(req, "min_tokens_to_keep") else 1,
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
            # Use increased timeout for tool-calling scenarios
            result = await self.adapter.await_token(
                nonce, 
                timeout_s=DEFAULT_TOKEN_TIMEOUT_SECONDS
            )
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
                        ) if req.logprobs else None,
                        finish_reason=None,
                    )
                ],
                created=int(time.time()),
                model=req.model,
            )

            # Stopping criteria
            if token == tokenizer.eos_token_id:
                completion_reason = ChatCompletionReason.STOP
                break

            # Check grammar termination
            if getattr(result, "grammar_terminated", False):
                logger.info("Grammar terminated signal received")
                if use_tool_grammar:
                    completion_reason = ChatCompletionReason.TOOL_CALLS
                else:
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
            "<|im_end|>",       # Qwen, ChatML format
            "<|im_start|>",     # Qwen, ChatML format
            "<|endoftext|>",    # GPT/generic
            "</s>",             # Llama, Mistral
            "<|eot_id|>",       # Llama 3
            "<|end|>",          # Phi
            "<|assistant|>",    # Some chat templates
            "<|user|>",         # Some chat templates
        ]
        for token in SPECIAL_TOKENS_TO_STRIP:
            final_text = final_text.replace(token, "")
        final_text = final_text.strip()

        # Parse tool calls from generated text
        tool_calls = None
        has_tool_calls_json = '"tool_calls"' in final_text and "{" in final_text
        should_attempt_parse = use_tool_grammar or (
            req.tools and req.tool_choice == "auto" and has_tool_calls_json
        )

        if should_attempt_parse and final_text:
            clean_text = final_text.strip()

            # Remove special tokens
            for eos_pattern in ["<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>"]:
                if eos_pattern in clean_text:
                    clean_text = clean_text.split(eos_pattern)[0].strip()
                    break
            
            # Handle Qwen3 <think> tags
            if "</think>" in clean_text:
                clean_text = clean_text.split("</think>")[-1].strip()

            try:
                parsed = json.loads(clean_text)
                if isinstance(parsed, dict) and "tool_calls" in parsed:
                    tool_calls = parsed["tool_calls"]
                    if tool_calls and isinstance(tool_calls, list):
                        completion_reason = ChatCompletionReason.TOOL_CALLS
                        tool_names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
                        logger.info(f"Parsed {len(tool_calls)} tool call(s): {tool_names}")
                    else:
                        tool_calls = None
            except json.JSONDecodeError:
                if use_tool_grammar:
                    logger.error(f"Failed to parse tool call JSON: {clean_text[:200]}")

        # Build metrics
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
                "tps_overall": round((tokens_generated / total_s) if tokens_generated else 0.0, 4),
                "tps_decoding": round((tokens_generated / gen_s) if tokens_generated else 0.0, 4),
            }

        # Build final message
        final_message = ChatMessage(
            role="assistant",
            content=None if tool_calls else final_text,
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

    # =========================================================================
    # Chat Completions (with optional MCP tool execution loop)
    # =========================================================================

    async def chat_completions(
        self,
        req: ChatRequestModel,
        execute_tools: bool = True,
        max_tool_rounds: Optional[int] = None,
    ) -> ChatResponseModel:
        """
        Handles chat completion request (non-streaming).
        
        If MCP is enabled and execute_tools=True, will automatically execute
        tool calls and feed results back to the model.
        
        Args:
            req: Chat completion request
            execute_tools: Whether to automatically execute tool calls
            max_tool_rounds: Maximum tool execution rounds (defaults to config value)
        """
        # Use default max_tool_rounds if not explicitly provided
        max_rounds = max_tool_rounds or DEFAULT_MAX_TOOL_ROUNDS
        
        logger.debug(f"chat_completions called: model={req.model}, execute_tools={execute_tools}, max_rounds={max_rounds}")
        
        # Check if MCP tool injection is needed
        working_req = req
        mcp_enabled = (
            self.mcp_provider is not None 
            and getattr(self.mcp_provider, 'enabled', False)
        )
        
        if mcp_enabled and not req.tools:  
            mcp_tools = self.mcp_provider.get_tools()
            if mcp_tools:
                logger.info(f"MCP tools auto-injected: {len(mcp_tools)} tools (tool_choice=auto, like Ollama)")
                working_req = req.model_copy(update={
                    "tools": mcp_tools,
                    "tool_choice": "auto",  
                })
            else:
                logger.debug("MCP enabled but no tools available")

        # Tool execution loop
        current_messages = list(working_req.messages)
        tool_round = 0
        response = None
        loop_start_time = time.perf_counter()

        while tool_round < max_rounds:
            tool_round += 1
            round_start_time = time.perf_counter()
            
            _log_tool_execution(
                "debug",
                f"Starting tool execution round {tool_round}/{max_rounds}",
                round_num=tool_round,
            )

            # Generate response - use model_copy to preserve all fields correctly
            loop_req = working_req.model_copy(update={
                "messages": current_messages,
                "stream": False,
            })
            response = await self._generate_single_completion(loop_req)

            # Check if we need to execute tools
            choice = response.choices[0]
            should_execute = (
                execute_tools
                and mcp_enabled
                and choice.finish_reason == ChatCompletionReason.TOOL_CALLS
                and choice.message
                and choice.message.tool_calls
            )

            if should_execute:
                tool_calls = choice.message.tool_calls
                
                _log_tool_execution(
                    "info",
                    f"Executing {len(tool_calls)} tool call(s) in round {tool_round}",
                    round_num=tool_round,
                )

                # Add assistant message with tool calls to conversation
                current_messages.append(choice.message)

                # Execute tools and add results
                tool_results = await self._execute_tool_calls(tool_calls, round_num=tool_round)
                current_messages.extend(tool_results)
                
                if tool_results:
                    # Check if any tool returned an error
                    has_errors = any(
                        result.content and (
                            result.content.startswith("Error") or 
                            "failed" in result.content.lower() or
                            "timeout" in result.content.lower()
                        )
                        for result in tool_results
                    )
                    
                    # Use appropriate guidance message based on tool execution result
                    if has_errors:
                        guidance = TOOL_EXECUTION_GUIDANCE_ERROR
                    else:
                        guidance = TOOL_EXECUTION_GUIDANCE_SUCCESS
                    
                    current_messages.append(ChatMessage(
                        role="user",
                        content=guidance
                    ))
                
                round_duration_ms = (time.perf_counter() - round_start_time) * 1000.0
                _log_tool_execution(
                    "debug",
                    f"Completed tool execution round {tool_round}",
                    round_num=tool_round,
                    duration_ms=round_duration_ms,
                )
                
                # Continue loop for next response
                continue

            # No tool calls or execution disabled - return response
            total_duration_ms = (time.perf_counter() - loop_start_time) * 1000.0
            _log_tool_execution(
                "info",
                f"Chat completion finished after {tool_round} round(s)",
                round_num=tool_round,
                duration_ms=total_duration_ms,
            )
            return response

        # Max rounds reached
        total_duration_ms = (time.perf_counter() - loop_start_time) * 1000.0
        _log_tool_execution(
            "warning",
            f"Max tool rounds ({max_rounds}) reached",
            round_num=tool_round,
            duration_ms=total_duration_ms,
            error_type="max_rounds_exceeded",
        )
        return response

    async def _generate_single_completion(self, req: ChatRequestModel) -> ChatResponseModel:
        """Generate a single completion (accumulates stream into response)."""
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
                    ) if req.logprobs else None,
                )
            ],
            usage=usage,
            created=int(time.time()),
            model=req.model,
            metrics=metrics_dict,
        )

    def resolve_request(self, nonce: str, result: Any):
        self.adapter.resolve_token(nonce, result)
