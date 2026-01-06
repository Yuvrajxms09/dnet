from typing import Any, List, Dict, Optional
import json

from .models import ChatRequestModel, ChatResponseModel, ChatMessage


class StructuredOutputManager:
    def __init__(self, inference_manager, schema: Any):
        self.inference_manager = inference_manager
        self.schema = schema
        self._structured_output_tool = self._schema_to_tool(schema)

    def _schema_to_tool(self, schema: Any) -> Dict[str, Any]:
        if hasattr(schema, "model_json_schema"):
            json_schema = schema.model_json_schema()
            return {
                "type": "function",
                "function": {
                    "name": schema.__name__,
                    "description": getattr(schema, "__doc__", "").strip() or "Structured response",
                    "parameters": json_schema,
                },
            }
        elif isinstance(schema, dict):
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
        original_tools = (
            self.inference_manager._bound_tools.copy()
            if hasattr(self.inference_manager, "_bound_tools")
            else []
        )

        try:
            if hasattr(self.inference_manager, "_bound_tools"):
                self.inference_manager._bound_tools.append(self._structured_output_tool)
            else:
                if not req.tools:
                    req.tools = []
                req.tools.append(self._structured_output_tool)

            if hasattr(req, "tool_choice"):
                req.tool_choice = "any"

            response = await self.inference_manager.chat_completions(req)

            if response.choices and response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    tool_name = (
                        tool_call.name
                        if hasattr(tool_call, "name")
                        else tool_call.get("function", {}).get("name", "")
                    )
                    if tool_name == self._structured_output_tool["function"]["name"]:
                        if hasattr(tool_call, "args"):
                            structured_data = tool_call.args
                        else:
                            args_str = tool_call.get("function", {}).get("arguments", "{}")
                            try:
                                structured_data = (
                                    json.loads(args_str)
                                    if isinstance(args_str, str)
                                    else args_str
                                )
                            except:
                                structured_data = {}

                        response.choices[0].message.content = str(structured_data)
                        response.structured_output = structured_data
                        break

            return response

        finally:
            if hasattr(self.inference_manager, "_bound_tools"):
                self.inference_manager._bound_tools = original_tools

    def bind_tools(self, tools: List[Any]) -> "StructuredOutputManager":
        if hasattr(self.inference_manager, "bind_tools"):
            self.inference_manager.bind_tools(tools)
        return self

    def __getattr__(self, name):
        return getattr(self.inference_manager, name)

    @classmethod
    def create_structured_wrapper(cls, inference_manager, schema: Any) -> "StructuredOutputManager":
        return cls(inference_manager, schema)
