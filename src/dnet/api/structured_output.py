import json
from typing import Any, Dict, List, TYPE_CHECKING

from .models import ChatRequestModel, ChatResponseModel, ChatMessage

if TYPE_CHECKING:
    from .inference import InferenceManager

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
