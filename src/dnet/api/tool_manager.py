from typing import List, Dict, Any, Optional

try:
    from .mcp_client import MCPToolClient
    MCP_CLIENT_AVAILABLE = True
except ImportError:
    MCP_CLIENT_AVAILABLE = False
    MCPToolClient = None

try:
    from toolregistry import ToolRegistry
    TOOL_REGISTRY_AVAILABLE = True
except ImportError:
    TOOL_REGISTRY_AVAILABLE = False
    ToolRegistry = None

from .tool_parser import ToolParser
from .tool_executor import ToolExecutor
from .mcp_transports import (
    MCPStdioTransport,
    MCPHttpTransport,
    MCPSseTransport,
    MCPPresetTransport,
    MCPMultiServerTransport,
)


class ToolManager:
    def __init__(self):
        self._mcp_client: Optional[MCPToolClient] = None
        if MCP_CLIENT_AVAILABLE:
            self._mcp_client = MCPToolClient()

        self._tool_registry: Optional[ToolRegistry] = None
        if TOOL_REGISTRY_AVAILABLE:
            self._tool_registry = ToolRegistry()

        self._parser = ToolParser()
        self._executor = ToolExecutor(self._mcp_client, self._tool_registry)

        self._stdio_transport = MCPStdioTransport(self._mcp_client)
        self._http_transport = MCPHttpTransport(self._tool_registry)
        self._sse_transport = MCPSseTransport(self._mcp_client)
        self._preset_transport = MCPPresetTransport(self._stdio_transport)
        self._multi_transport = MCPMultiServerTransport(self._mcp_client)

        self._bound_tools: List[Dict[str, Any]] = []

    async def register_mcp_stdio(
        self,
        server_name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> bool:
        success = await self._stdio_transport.register(server_name, command, args, env)
        if success:
            self._sync_mcp_tools_to_bound()
        return success

    async def register_mcp_http(
        self, server_name: str, url: str, headers: Optional[Dict[str, str]] = None
    ) -> bool:
        success = await self._http_transport.register(server_name, url, headers)
        if success:
            self._sync_registry_tools_to_bound()
        return success

    async def register_mcp_sse(
        self, server_name: str, url: str, headers: Optional[Dict[str, str]] = None
    ) -> bool:
        success = await self._sse_transport.register(server_name, url, headers)
        if success:
            self._sync_mcp_tools_to_bound()
        return success

    async def register_mcp_preset(
        self, preset_name: str, env: Optional[Dict[str, str]] = None
    ) -> bool:
        success = await self._preset_transport.register(preset_name, env)
        if success:
            self._sync_mcp_tools_to_bound()
        return success

    async def register_mcp_servers(self, config: Dict[str, Dict[str, Any]]) -> bool:
        success = await self._multi_transport.register_servers(config)
        if success:
            self._sync_mcp_tools_to_bound()
        return success

    def bind_tools(self, tools: List[Any]) -> "ToolManager":
        bound_tools = []
        for tool in tools:
            tool_def = self._convert_tool_to_definition(tool)
            if tool_def:
                bound_tools.append(tool_def)

        self._bound_tools = bound_tools
        return self

    def get_bound_tools(self) -> List[Dict[str, Any]]:
        return self._bound_tools.copy()

    def get_all_tools_openai_format(self) -> List[Dict[str, Any]]:
        tools = list(self._bound_tools)

        if MCP_CLIENT_AVAILABLE and self._mcp_client:
            mcp_tools = self._mcp_client.get_tools_openai_format()
            for tool in mcp_tools:
                if tool not in tools:
                    tools.append(tool)

        return tools

    def create_langchain_tool_prompt(self, available_tools: List[Dict[str, Any]]) -> str:
        return self._parser.create_langchain_tool_prompt(available_tools)

    def parse_tool_calls_langchain_style(self, content: str) -> Optional[List[Dict[str, Any]]]:
        return self._parser.parse_tool_calls_langchain_style(content)

    def convert_to_tool_call_objects(self, parsed_calls: List[Dict[str, Any]]) -> List[Any]:
        from .models import ToolCall
        return self._parser.convert_to_tool_call_objects(parsed_calls)

    def format_tool_call_response(self, content: str, tool_calls: Optional[List[Any]]) -> str:
        return self._parser.format_tool_call_response(content, tool_calls)

    async def execute_tool_calls_async(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return await self._executor.execute_tool_calls_async(tool_calls)

    def _sync_mcp_tools_to_bound(self):
        if MCP_CLIENT_AVAILABLE and self._mcp_client:
            mcp_tools = self._mcp_client.get_tools_openai_format()
            for tool in mcp_tools:
                if tool not in self._bound_tools:
                    self._bound_tools.append(tool)

    def _sync_registry_tools_to_bound(self):
        if TOOL_REGISTRY_AVAILABLE and self._tool_registry:
            try:
                registry_tools = self._tool_registry.get_tools_json()
                for tool in registry_tools:
                    if tool not in self._bound_tools:
                        self._bound_tools.append(tool)
            except Exception:
                pass

    def _convert_tool_to_definition(self, tool: Any) -> Optional[Dict[str, Any]]:
        if isinstance(tool, dict) and tool.get("type") == "function":
            return tool

        if hasattr(tool, "__annotations__") and hasattr(tool, "model_json_schema"):
            try:
                schema = tool.model_json_schema()
                return {
                    "type": "function",
                    "function": {
                        "name": getattr(tool, "__name__", tool.__class__.__name__.lower()),
                        "description": getattr(tool, "__doc__", "").strip(),
                        "parameters": schema,
                    },
                }
            except Exception:
                pass

        if callable(tool) and hasattr(tool, "__name__"):
            import inspect

            try:
                sig = inspect.signature(tool)
                params = {}
                for name, param in sig.parameters.items():
                    if name == "self":
                        continue
                    param_def = {"type": "string"}
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
            except Exception:
                pass

        return None
