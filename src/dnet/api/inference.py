import asyncio
import time
import uuid
import json
import mlx.core as mx
import numpy as np
from typing import Optional, Any, List, Union, Dict

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

from .tool_manager import ToolManager
from .structured_output import StructuredOutputManager


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
        self._tool_manager = ToolManager()

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
        logger.info(f"🚀 generate_stream START: model={req.model}")
        logger.debug(f"   Request details: messages={len(req.messages) if req.messages else 0}, tools={len(req.tools) if req.tools else 0}")

        if not self.model_manager.tokenizer:
            logger.error("❌ generate_stream FAILED: No tokenizer available")
            raise RuntimeError(
                "Inference manager not ready (ring not connected or tokenizer not loaded)"
            )

        tokenizer = self.model_manager.tokenizer
        logger.debug("✅ Tokenizer ready, proceeding with generation")

        try:
            if (
                hasattr(tokenizer, "chat_template")
                and tokenizer.chat_template is not None
            ):
                # Convert messages to dict format
                message_dicts = []

                # Add tool system message if tools are available (LangChain-style)
                # Use bound tools (LangChain approach) or request tools (backward compatibility)
                available_tools = self._tool_manager.get_bound_tools() or req.tools or []
                logger.debug(f"🛠️ Tool check: bound_tools={len(self._tool_manager.get_bound_tools())}, req_tools={len(req.tools) if req.tools else 0}, available={len(available_tools)}")
                if available_tools:
                    logger.debug("📝 Generating tool system message...")
                    tool_system_msg = self._tool_manager.create_langchain_tool_prompt(
                        available_tools
                    )
                    message_dicts.append({"role": "system", "content": tool_system_msg})
                    logger.debug(f"✅ Tool system message added ({len(tool_system_msg)} chars)")

                for m in req.messages:
                    msg_dict = {"role": m.role, "content": m.content or ""}
                    message_dicts.append(msg_dict)

                prompt_text = tokenizer.apply_chat_template(
                    message_dicts,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                logger.debug(f"📝 Chat template applied, prompt length: {len(prompt_text)}")
            else:
                prompt_text = (
                    "\n".join(m.content or "" for m in req.messages) + "\nAssistant:"
                )
                logger.debug("📝 Using fallback prompt format")
        except Exception as e:
            logger.warning(f"⚠️ Failed to apply chat template: {e}, using fallback")
            prompt_parts = []

            # Add tool system message if tools are provided (LangChain-style)
            if req.tools:
                logger.debug("📝 Adding tools to fallback prompt")
                tool_system_msg = self._tool_manager.create_langchain_tool_prompt(req.tools)
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

        logger.debug("🔄 Resetting cache and starting inference")
        await self.adapter.reset_cache()

        logger.debug("📤 Yielding initial chunk with assistant role")
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
        logger.debug(f"📦 Initial chunk created: {len(initial_chunk.model_dump_json())} bytes")
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
            logger.debug(f"📊 Token data prepared: {len(tok_bytes)} bytes")

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

            logger.debug("📤 Sending tokens to shard...")
            # Send tokens to first shard
            await self.adapter.send_tokens(
                tokens=tok_bytes,
                nonce=nonce,
                callback_addr=self._api_callback_addr,
                logprobs=req.logprobs if req.logprobs else False,
                top_logprobs=req.top_logprobs if req.top_logprobs else 0,
                decoding_config=decoding_config,
            )
            logger.debug("⏳ Awaiting token response...")
            result = await self.adapter.await_token(nonce, timeout_s=300.0)
            token = int(result.token_id)
            logger.debug(f"✅ Token received: {token}")

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

            logger.debug(f"📦 Creating chunk: delta='{delta_text[:50]}...', token={token}")
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

        # Parse tool calls if tools were available
        logger.debug(f"🔍 Starting tool call processing for final text")
        tool_calls = None
        final_content = final_text
        available_tools = self._tool_manager.get_bound_tools() or req.tools or []
        logger.debug(f"🛠️ Tool availability: bound={len(self._tool_manager.get_bound_tools())}, req={len(req.tools) if req.tools else 0}, available={len(available_tools)}")

        if available_tools:
            logger.info(f"🛠️ Tools available ({len(available_tools)}), attempting to parse tool calls")
            try:
                parsed_calls = self._tool_manager.parse_tool_calls_langchain_style(final_text)
                logger.debug(f"🔍 Parse result: {len(parsed_calls) if parsed_calls else 0} tool calls")
                if parsed_calls:
                    logger.info(f"✅ Found {len(parsed_calls)} tool calls in response")
                    # Convert parsed dicts to ToolCall objects
                    tool_calls = self._tool_manager.convert_to_tool_call_objects(parsed_calls)
                    final_content = self._tool_manager.format_tool_call_response(final_text, tool_calls)
                    logger.debug(f"📝 Formatted response content (tool calls present)")
                    logger.debug(f"📋 Tool calls: {[tc.name for tc in tool_calls] if tool_calls else []}")
                else:
                    logger.debug("📝 No tool calls found, using original content")
                    final_content = final_text
            except Exception as e:
                logger.error(f"❌ Tool parsing failed: {e}")
                logger.debug("Tool parsing error details:", exc_info=True)
                final_content = final_text
        else:
            logger.debug("📝 No tools available, skipping tool parsing")

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

        # Clean up structured output responses - remove end tokens
        if req.structured_outputs and req.structured_outputs.json:
            full_content = full_content.strip()
            for token in ["<|im_end|>", "<|endoftext|>", "</s>"]:
                if token in full_content:
                    full_content = full_content.split(token)[0].strip()

        # Parse tool calls if tools were available (LangChain-style, no grammar)
        tool_calls = None
        final_content = full_content
        available_tools = self._tool_manager.get_bound_tools() or req.tools or []
        if available_tools:
            parsed_calls = self._tool_manager.parse_tool_calls_langchain_style(full_content)
            if parsed_calls:
                # Convert parsed dicts to ToolCall objects
                tool_calls = self._tool_manager.convert_to_tool_call_objects(parsed_calls)
                final_content = self._tool_manager.format_tool_call_response(
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

    async def register_mcp_stdio(
        self,
        server_name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> bool:
        return await self._tool_manager.register_mcp_stdio(server_name, command, args, env)

    async def register_mcp_http(
        self, server_name: str, url: str, headers: Optional[Dict[str, str]] = None
    ) -> bool:
        return await self._tool_manager.register_mcp_http(server_name, url, headers)

    async def register_mcp_sse(
        self, server_name: str, url: str, headers: Optional[Dict[str, str]] = None
    ) -> bool:
        return await self._tool_manager.register_mcp_sse(server_name, url, headers)

    async def register_mcp_preset(
        self, preset_name: str, env: Optional[Dict[str, str]] = None
    ) -> bool:
        return await self._tool_manager.register_mcp_preset(preset_name, env)

    async def register_mcp_servers(self, config: Dict[str, Dict[str, Any]]) -> bool:
        return await self._tool_manager.register_mcp_servers(config)

    def bind_tools(self, tools: List[Any]) -> "InferenceManager":
        self._tool_manager.bind_tools(tools)
        return self

    def get_registered_tools(self) -> List[str]:
        return self._tool_manager.get_all_tools_openai_format()

    def with_structured_output(self, schema: Any):
        return StructuredOutputManager.create_structured_wrapper(self, schema)

    def resolve_request(self, nonce: str, result: Any):
        """Resolve a pending request with the given result.

        Called by gRPC servicer when a token is received from a shard.
        """
        self.adapter.resolve_token(nonce, result)
