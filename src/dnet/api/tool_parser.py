import json
import re
import uuid
from typing import Optional, List, Dict, Any
from .models import ToolCall


class ToolParser:
    def create_langchain_tool_prompt(self, tools: List[Dict[str, Any]]) -> str:
        all_tools = list(tools)
        if not all_tools:
            return ""

        tool_summaries = []
        for tool in all_tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                name = func.get("name", "unknown")
                desc = func.get("description", "")

                short_desc = desc.split(".")[0][:100] if desc else ""
                summary = f"- {name}: {short_desc}"
                tool_summaries.append(summary)

        tools_section = "\n".join(tool_summaries)

        prompt = """

You have access to the following tools:
{tools_section}

To use a tool, respond with ONLY a JSON object containing tool calls:
{{"tool_calls": [{{"id": "call_1", "type": "function", "function": {{"name": "tool_name", "arguments": "{{\\"param\\": \\"value\\"}}"}}}}]}}}

For general conversation or when no tools are needed, respond with plain text.

Important: Only output JSON when you actually want to call tools. For normal responses, just write naturally.""".format(tools_section=tools_section)

        return prompt

    def parse_tool_calls_langchain_style(self, content: str) -> Optional[List[Dict[str, Any]]]:
        if not content or not content.strip():
            return None

        clean_content = content.strip()
        original_length = len(clean_content)

        think_removed = re.sub(r'<think>.*?</think>', '', clean_content, flags=re.DOTALL | re.IGNORECASE).strip()
        if len(think_removed) != len(clean_content):
            pass
        clean_content = think_removed

        special_tokens = ['<|im_end|>', '<|im_start|>', '<|endoftext|>', '</s>', '<|eot_id|>', '<|end|>']
        for token in special_tokens:
            clean_content = clean_content.replace(token, '').strip()

        prefixes_to_remove = ["Assistant:", "AI:", "Response:"]
        for prefix in prefixes_to_remove:
            if clean_content.startswith(prefix):
                clean_content = clean_content[len(prefix) :].strip()

        try:
            data = json.loads(clean_content)
            if isinstance(data, dict) and "tool_calls" in data:
                calls = data["tool_calls"]
                if isinstance(calls, list) and calls:
                    return calls
        except json.JSONDecodeError:
            pass

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

        try:
            data = json.loads(clean_content)
            if isinstance(data, dict) and "function" in data:
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
        candidates = []

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

    def convert_to_tool_call_objects(self, parsed_calls: List[Dict[str, Any]]) -> List[ToolCall]:
        tool_calls = []
        for call in parsed_calls:
            try:
                if "function" in call:
                    func = call.get("function", {})
                    name = func.get("name", "")
                    args_raw = func.get("arguments", "{}")

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

            except Exception:
                continue

        return tool_calls

    def format_tool_call_response(self, content: str, tool_calls: Optional[List[ToolCall]]) -> str:
        if tool_calls:
            return ""
        return content
