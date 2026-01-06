import asyncio
import time
import uuid
import json
import mlx.core as mx
import numpy as np
from typing import Optional, Any, List, Union, Dict


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
    """Inference manager for dnet with MCP and LangChain-style tool integration."""

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
        """
        Generator for chat completion chunks.
        """
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

                # Check if we should use structured output for tools
                available_tools = self._bound_tools or req.tools or []
                use_structured_tools = bool(available_tools) and not req.structured_outputs

                if use_structured_tools:
                    # Structured output will handle tools - no need for text prompt
                    logger.debug(f"🛠️ Using structured output for {len(available_tools)} tools")
                elif available_tools:
                    # Fallback to text-based tool prompting
                    logger.debug(f"🛠️ Using text-based tool prompting for {len(available_tools)} tools")
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
            logger.warning(f"⚠️ Failed to apply chat template: {e}, using fallback")

            # Check if we should use structured output for tools
            available_tools = self._bound_tools or req.tools or []
            use_structured_tools = bool(available_tools) and not req.structured_outputs

            if use_structured_tools:
                # Structured output will handle tools - simple fallback
                prompt_text = "\n".join(m.content or "" for m in req.messages) + "\nAssistant:"
                logger.debug(f"📝 Structured fallback prompt created, length: {len(prompt_text)}")
            else:
                # Fallback with tool prompting if needed
                prompt_parts = []
                if available_tools:
                    logger.debug("📝 Adding tools to fallback prompt")
                    tool_system_msg = self._create_langchain_tool_prompt(available_tools)
                    prompt_parts.append(f"System: {tool_system_msg}")

                prompt_parts.extend(m.content or "" for m in req.messages)
                prompt_parts.append("Assistant:")
                prompt_text = "\n".join(prompt_parts)
                logger.debug(f"📝 Fallback prompt created, length: {len(prompt_text)}")

        logger.debug("🔢 Encoding prompt to tokens...")
        prompt_tokens = tokenizer.encode(prompt_text)
        prompt_array = mx.array(prompt_tokens)
        logger.info(f"✅ Prompt encoded: {len(prompt_tokens)} tokens")

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

        # Check if we should use structured output for tool calls
        available_tools = self._bound_tools or req.tools or []
        use_structured_tools = bool(available_tools) and not req.structured_outputs

        if use_structured_tools:
            # Create structured output schema for tool calls
            tool_names = []
            for tool in available_tools:
                if tool.get("type") == "function" and "function" in tool:
                    tool_names.append(tool["function"]["name"])

            tool_call_schema = self._create_tool_call_schema(tool_names)
            req.structured_outputs = StructuredOutputsParams(json=tool_call_schema)
            logger.debug(f"🔧 Using structured output for tool calls with schema: {tool_names}")

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
        initial_chunk = ChatResponseModel(
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
        yield initial_chunk

        logger.info(f"🔄 Starting inference loop: max_tokens={req.max_tokens}")
        y = prompt_array
        generated_tokens = 0
        for i in range(req.max_tokens):
            logger.debug(f"🔄 Token {i+1}/{req.max_tokens}: preparing data")
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
            chunk = ChatResponseModel(
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

            try:
                chunk_json = chunk.model_dump_json(exclude_none=True)
                logger.debug(f"📦 Chunk JSON created: {len(chunk_json)} bytes")
                yield chunk
                logger.debug("✅ Chunk yielded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to serialize chunk: {e}")
                logger.debug(f"   Chunk data: {chunk}")
                raise

            # stopping criteria
            if token == tokenizer.eos_token_id:
                logger.debug(f"🛑 EOS token detected ({tokenizer.eos_token_id}), stopping generation")
                completion_reason = ChatCompletionReason.STOP
                break

            y = mx.array([token], dtype=mx.int32)
            generated_tokens += 1
            logger.debug(f"🔄 Continuing with token {generated_tokens} generated so far")

        logger.debug("🏁 Finalizing detokenizer...")
        detokenizer.finalize()
        final_text = detokenizer.text
        logger.info(f"📝 Final text generated: {len(final_text)} chars")
        logger.debug(f"📝 Final text preview: {final_text[:200]}...")

        # Strip special tokens from output
        # mlx-lm's NaiveStreamingDetokenizer calls tokenizer.decode() without skip_special_tokens=True
        # (see: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tokenizer_utils.py)
        # So we strip them manually as a post-processing step
        logger.debug("🧹 Stripping special tokens...")
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
        original_length = len(final_text)
        for token in SPECIAL_TOKENS_TO_STRIP:
            final_text = final_text.replace(token, "")
        final_text = final_text.strip()
        logger.debug(f"🧹 Special tokens stripped: {original_length} -> {len(final_text)} chars")

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

        # Process tool calls from structured output (no parsing needed!)
        logger.debug(f"🔍 Processing tool calls from structured output")
        tool_calls = None
        final_content = final_text
        available_tools = self._bound_tools or req.tools or []

        if available_tools and req.structured_outputs:
            logger.info(f"🛠️ Tools available with structured output, checking for tool calls")
            try:
                # Parse the structured output directly (no text parsing!)
                structured_output = json.loads(final_text.strip())
                logger.debug(f"📋 Structured output: {structured_output}")

                if structured_output.get("response_type") == "tool_calls":
                    tool_calls_data = structured_output.get("tool_calls", [])
                    if tool_calls_data:
                        logger.info(f"✅ Found {len(tool_calls_data)} tool calls in structured output")
                        # Convert structured data directly to ToolCall objects
                        tool_calls = []
                        for call_data in tool_calls_data:
                            tool_call = ToolCall(
                                name=call_data["name"],
                                args=call_data["arguments"],
                                id=call_data.get("id", f"call_{uuid.uuid4().hex[:8]}")
                            )
                            tool_calls.append(tool_call)

                        final_content = structured_output.get("reasoning", "Using tools...")
                        logger.debug(f"📝 Tool calls: {[tc.name for tc in tool_calls]}")

                elif structured_output.get("response_type") == "text_response":
                    logger.debug("📝 Structured output indicates direct text response")
                    final_content = structured_output["text_response"]

            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse structured output as JSON: {e}")
                logger.debug(f"   Raw output: {final_text[:500]}")
                final_content = final_text
            except Exception as e:
                logger.error(f"❌ Failed to process structured tool output: {e}")
                final_content = final_text
        elif available_tools:
            # Fallback to text parsing for backward compatibility
            logger.warning("⚠️ Tools available but no structured output - falling back to text parsing")
            try:
                parsed_calls = self._parse_tool_calls_langchain_style(final_text)
                if parsed_calls:
                    tool_calls = self._convert_to_tool_call_objects(parsed_calls)
                    final_content = self._format_tool_call_response(final_text, tool_calls)
            except Exception as e:
                logger.error(f"❌ Fallback tool parsing failed: {e}")
                final_content = final_text
        else:
            logger.debug("📝 No tools available, using direct response")

        logger.debug("📝 Creating final message and chunk")
        final_message = ChatMessage(
            role="assistant",
            content=final_content,
            tool_calls=tool_calls,
        )
        logger.debug(f"📝 Final message created: content={len(final_content)} chars, tool_calls={len(tool_calls) if tool_calls else 0}")

        # Final chunk
        final_chunk = ChatResponseModel(
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

        try:
            final_chunk_json = final_chunk.model_dump_json(exclude_none=True)
            logger.info(f"🏁 generate_stream COMPLETE: final chunk {len(final_chunk_json)} bytes")
            logger.debug(f"📊 Generation stats: prompt_tokens={len(prompt_tokens)}, completion_tokens={len(tokens)}, total={len(prompt_tokens) + len(tokens)}")
            yield final_chunk
            logger.debug("✅ Final chunk yielded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to serialize final chunk: {e}")
            logger.debug(f"   Final chunk data: {final_chunk}")
            raise

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

        # Process structured output for tool calls or regular responses
        tool_calls = None
        final_content = full_content
        available_tools = self._bound_tools or req.tools or []

        if req.structured_outputs and req.structured_outputs.json:
            # Clean up structured output responses - remove end tokens
            full_content = full_content.strip()
            for token in ["<|im_end|>", "<|endoftext|>", "</s>"]:
                if token in full_content:
                    full_content = full_content.split(token)[0].strip()

            try:
                structured_output = json.loads(full_content)
                logger.debug(f"📋 Structured output in chat_completions: {structured_output}")

                if available_tools and structured_output.get("response_type") == "tool_calls":
                    tool_calls_data = structured_output.get("tool_calls", [])
                    if tool_calls_data:
                        tool_calls = []
                        for call_data in tool_calls_data:
                            tool_call = ToolCall(
                                name=call_data["name"],
                                args=call_data["arguments"],
                                id=call_data.get("id", f"call_{uuid.uuid4().hex[:8]}")
                            )
                            tool_calls.append(tool_call)
                        final_content = structured_output.get("reasoning", "Using tools...")

                elif structured_output.get("response_type") == "text_response":
                    final_content = structured_output["text_response"]

            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse structured output in chat_completions: {e}")
                final_content = full_content
        elif available_tools:
            # Fallback to text parsing for backward compatibility
            logger.warning("⚠️ Tools available but no structured output - falling back to text parsing")
            parsed_calls = self._parse_tool_calls_langchain_style(full_content)
            if parsed_calls:
                tool_calls = self._convert_to_tool_call_objects(parsed_calls)
                final_content = self._format_tool_call_response(full_content, tool_calls)

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



    def _create_tool_call_schema(self, tool_names: List[str]) -> Dict[str, Any]:
        """Create JSON schema for structured tool call output (eliminates text parsing)."""
        return {
            "type": "object",
            "properties": {
                "response_type": {
                    "type": "string",
                    "enum": ["tool_calls", "text_response"],
                    "description": "Whether this response contains tool calls or direct text"
                },
                "tool_calls": {
                    "type": "array",
                    "description": "Tool calls to execute",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "enum": tool_names,
                                "description": "Name of the tool to call"
                            },
                            "arguments": {
                                "type": "object",
                                "description": "Arguments for the tool call"
                            },
                            "id": {
                                "type": "string",
                                "description": "Unique identifier for this tool call"
                            }
                        },
                        "required": ["name", "arguments"]
                    }
                },
                "text_response": {
                    "type": "string",
                    "description": "Direct text response when no tools are needed"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Optional reasoning about the decision"
                }
            },
            "required": ["response_type"],
            "allOf": [
                {
                    "if": {"properties": {"response_type": {"const": "tool_calls"}}},
                    "then": {"required": ["tool_calls"]}
                },
                {
                    "if": {"properties": {"response_type": {"const": "text_response"}}},
                    "then": {"required": ["text_response"]}
                }
            ]
        }

    def _create_langchain_tool_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """Create LangChain-style tool prompt (no grammar constraints, just instructions)."""
        all_tools = list(tools)  # Start with provided tools

        # Add tools from ToolRegistry if available (but avoid duplicates with already bound tools)
        if TOOL_REGISTRY_AVAILABLE and self._tool_registry:
            try:
                registry_tools = self._tool_registry.get_tools_json()
                # Filter out duplicates by name - only add tools not already in all_tools
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

                # Let full description pass (no shortening)
                summary = f"- {name}: {desc}"
                tool_summaries.append(summary)

        # Debug: Log first few tool names
        logger.debug(f"🛠️ Tool prompt includes {len(all_tools)} tools. First 5: {[t.get('function', {}).get('name', 'unknown') for t in all_tools[:5]]}")

        tools_section = "\n".join(tool_summaries)

        prompt = f"""

You have access to the following tools:
{tools_section}

To use a tool, respond with ONLY a JSON object containing tool calls:
{{"tool_calls": [{{"id": "call_1", "type": "function", "function": {{"name": "tool_name", "arguments": "{{\\"param\\": \\"value\\"}}"}}}}]}}

For general conversation or when no tools are needed, respond with plain text.

Important: Only output JSON when you actually want to call tools. For normal responses, just write naturally."""

        logger.debug(f"🛠️ Created tool prompt ({len(prompt)} chars) with {len(all_tools)} tools")
        return prompt

    def _parse_tool_calls_langchain_style(
        self, content: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Parse tool calls using LangChain-style robust parsing (no grammar required)."""
        if not content or not content.strip():
            return None

        logger.debug(f"🔍 Parsing tool calls from content ({len(content)} chars): {content[:200]}...")

        # Remove common prefixes/suffixes that models sometimes add
        clean_content = content.strip()

        # Remove think tags (from proposed code) - make more robust
        import re
        think_removed = re.sub(r'<think>.*?</think>', '', clean_content, flags=re.DOTALL | re.IGNORECASE).strip()
        if len(think_removed) != len(clean_content):
            logger.debug(f"🧹 Removed think tags: {len(clean_content)} -> {len(think_removed)} chars")
        clean_content = think_removed

        # Remove special tokens (from proposed code)
        special_tokens = ['<|im_end|>', '<|im_start|>', '<|endoftext|>', '</s>', '<|eot_id|>', '<|end|>']
        for token in special_tokens:
            clean_content = clean_content.replace(token, '').strip()

        prefixes_to_remove = ["Assistant:", "AI:", "Response:"]
        for prefix in prefixes_to_remove:
            if clean_content.startswith(prefix):
                clean_content = clean_content[len(prefix) :].strip()

        logger.debug(f"🧹 After cleaning ({len(clean_content)} chars): {clean_content[:300]}...")

        # Try direct JSON parsing first
        try:
            data = json.loads(clean_content)
            if isinstance(data, dict) and "tool_calls" in data:
                calls = data["tool_calls"]
                if isinstance(calls, list) and calls:
                    logger.debug(f"✅ Found {len(calls)} tool calls via direct JSON parsing")
                    return calls
        except json.JSONDecodeError:
            logger.debug("❌ Direct JSON parsing failed, trying fallback methods")

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
                logger.debug(f"✅ Found single tool call: {data['function'].get('name', 'unknown')}")
                return [tool_call]
        except (json.JSONDecodeError, KeyError):
            logger.debug("❌ Single tool call parsing also failed")

        logger.debug("❌ No tool calls found in content")
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


    async def _execute_tool_calls_async(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute tool calls using ToolRegistry's batch API."""
        if not TOOL_REGISTRY_AVAILABLE or not self._tool_registry:
            logger.error("ToolRegistry not available for tool execution")
            return [
                {
                    "tool_call_id": "error",
                    "content": "ToolRegistry not available",
                    "success": False,
                }
            ]

        try:
            logger.info(f"🔨 Executing {len(tool_calls)} tool calls via ToolRegistry batch API")
            # Use ToolRegistry's built-in batch execution method
            tool_responses = self._tool_registry.execute_tool_calls(tool_calls)

            # Convert ToolRegistry response format to expected format
            results = []
            for tool_call_id, result in tool_responses.items():
                results.append({
                    "tool_call_id": tool_call_id,
                    "content": str(result),
                    "success": True
                })

            successful_count = len([r for r in results if r["success"]])
            logger.info(f"📊 Tool execution complete: {successful_count}/{len(results)} successful")
            return results

        except Exception as e:
            logger.error(f"❌ ToolRegistry batch execution failed: {e}")
            return [
                {
                    "tool_call_id": "error",
                    "content": f"Tool execution failed: {e}",
                    "success": False,
                }
            ]

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
        user_query = req.messages[-1].content[:100] if req.messages else 'empty'
        logger.info(f"🚀 Starting AGENT EXECUTION FLOW for: '{user_query}'")
        logger.debug(f"📋 Agent request: tools={len(req.tools) if req.tools else 0}, messages={len(req.messages)}")

        # Step 1: Generate initial response (may contain tool calls)
        logger.info("📝 Step 1: Generating initial response with potential tool calls")
        initial_response = await self._generate_single_completion(req)

        choice = initial_response.choices[0]
        if not choice.message or not choice.message.tool_calls:
            # No tool calls, return normal response
            logger.info("📝 No tool calls generated, returning normal response")
            return initial_response

        tool_calls = choice.message.tool_calls
        logger.info(f"⚙️ Step 2: Found {len(tool_calls)} tool calls to execute")

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

        logger.info(f"🔨 Step 2b: Executing {len(tool_calls_dict)} tool calls")
        tool_results = await self._execute_tool_calls_async(tool_calls_dict)
        successful_results = sum(1 for r in tool_results if r.get('success'))
        logger.info(f"📊 Tool execution complete: {successful_results}/{len(tool_results)} successful")

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
            content="Based on the tool results above, provide a comprehensive and well-structured answer to my original question. Extract and organize the key information from the tool outputs.",
        )
        new_messages.append(guidance_msg)

        # Step 4: Generate final synthesized response
        logger.info(f"🎯 Step 4: Generating final response with {len(new_messages)} messages")
        logger.debug(f"📋 Final request: {len(new_messages)} messages, no tools")

        final_req = req.model_copy(
            update={
                "messages": new_messages,
                "tools": None,  # Don't include tools in final generation
            }
        )

        final_response = await self._generate_single_completion(final_req)

        # Mark as tool-synthesized response
        final_response.choices[0].finish_reason = ChatCompletionReason.STOP
        final_content = final_response.choices[0].message.content if final_response.choices[0].message else ""
        logger.info(f"🎉 AGENT COMPLETE: Generated {len(final_content)} char synthesized response")
        logger.debug(f"📄 Final answer preview: {final_content[:200]}...")

        return final_response

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
            from pydantic import BaseModel, Field

            class WeatherResponse(BaseModel):
                temperature: float = Field(description="Temperature in Fahrenheit")
                wind_direction: str = Field(description="Wind direction")

            # Create structured output wrapper
            structured_llm = inference_manager.with_structured_output(WeatherResponse)

            # Agent will call WeatherResponse tool with structured data when ready to respond
        """
        return StructuredOutputInferenceManager(self, schema)

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
            # Disable namespacing to keep tool names simple for model understanding
            await self._tool_registry.register_from_mcp_async(transport, with_namespace=False)

            # Get registered tools and bind them (like MCP client does)
            registry_tools = self._tool_registry.get_tools_json()
            logger.info(f"ToolRegistry returned {len(registry_tools)} tools for {server_name}")

            # Debug: Log actual tool names
            tool_names = [t.get('function', {}).get('name', 'unknown') for t in registry_tools[:5]]
            logger.debug(f"📋 Sample tool names: {tool_names}")

            # Bind tools to inference manager (now that streaming errors are fixed)
            for tool in registry_tools:
                if tool not in self._bound_tools:
                    self._bound_tools.append(tool)
            logger.info(f"✅ Bound {len(registry_tools)} tools from {server_name} to inference manager")


            logger.info(f"Registered and bound {len(registry_tools)} tools from {server_name} MCP server")

            logger.info(f"MCP server '{server_name}' (HTTP) registered successfully via ToolRegistry")
            return True

        except Exception as e:
            logger.error(f"Failed to register MCP server '{server_name}': {e}")
            return False








    def get_all_tools_openai_format(self) -> List[Dict[str, Any]]:
        """Get all tools in OpenAI-compatible format."""
        tools = list(self._bound_tools)


        return tools

    def resolve_request(self, nonce: str, result: Any):
        """Resolve a pending request with the given result.

        Called by gRPC servicer when a token is received from a shard.
        """
        self.adapter.resolve_token(nonce, result)

    async def generate_with_structured_tools(self, req: ChatRequestModel) -> ChatResponseModel:
        """
        Generate response using structured output for tool calls (experimental).

        This method forces structured output for tool calls, eliminating text parsing.
        Use this to test the grammar-based tool call approach.
        """
        if not req.tools and not self._bound_tools:
            # No tools, just do normal generation
            return await self.chat_completions(req)

        # Force structured output for tools
        original_structured = req.structured_outputs
        try:
            available_tools = self._bound_tools or req.tools or []
            tool_names = []
            for tool in available_tools:
                if tool.get("type") == "function" and "function" in tool:
                    tool_names.append(tool["function"]["name"])

            tool_call_schema = self._create_tool_call_schema(tool_names)
            req.structured_outputs = StructuredOutputsParams(json=tool_call_schema)

            logger.info(f"🔧 Using structured tool output for tools: {tool_names}")
            return await self.chat_completions(req)

        finally:
            req.structured_outputs = original_structured


class StructuredOutputInferenceManager:
    """
    structured output wrapper for InferenceManager.
    """

    def __init__(self, inference_manager: "InferenceManager", schema: Any):
        self.inference_manager = inference_manager
        self.schema = schema
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
        # Temporarily bind the structured output tool
        original_tools = (
            self.inference_manager._bound_tools.copy()
            if hasattr(self.inference_manager, "_bound_tools")
            else []
        )

        try:
            if hasattr(self.inference_manager, "_bound_tools"):
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
        if hasattr(self.inference_manager, "bind_tools"):
            self.inference_manager.bind_tools(tools)
        return self

    def __getattr__(self, name):
        """Delegate other methods to the underlying inference manager."""
        return getattr(self.inference_manager, name)

    def with_structured_output(self, schema: Any) -> "StructuredOutputInferenceManager":
         return StructuredOutputInferenceManager(self.inference_manager, schema)
