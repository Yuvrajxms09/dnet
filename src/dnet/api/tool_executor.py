import asyncio
import json
import uuid
from typing import List, Dict, Any, Optional

try:
    from toolregistry import ToolRegistry
    TOOL_REGISTRY_AVAILABLE = True
except ImportError:
    TOOL_REGISTRY_AVAILABLE = False
    ToolRegistry = None

try:
    from .mcp_client import MCPToolClient
    MCP_CLIENT_AVAILABLE = True
except ImportError:
    MCP_CLIENT_AVAILABLE = False
    MCPToolClient = None


class ToolExecutor:
    def __init__(self, mcp_client: Optional[MCPToolClient] = None, tool_registry: Optional[ToolRegistry] = None):
        self._mcp_client = mcp_client
        self._tool_registry = tool_registry

    async def execute_tool_calls_async(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not TOOL_REGISTRY_AVAILABLE or not self._tool_registry:
            return [
                {
                    "tool_call_id": "error",
                    "content": "ToolRegistry not available",
                    "success": False,
                }
            ]

        try:
            tool_responses = self._tool_registry.execute_tool_calls(tool_calls)

            results = []
            for tool_call_id, result in tool_responses.items():
                results.append({
                    "tool_call_id": tool_call_id,
                    "content": str(result),
                    "success": True
                })

            return results

        except Exception as e:
            return [
                {
                    "tool_call_id": "error",
                    "content": f"Tool execution failed: {e}",
                    "success": False,
                }
            ]

    async def execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self._has_execution_capability():
            return []

        results = []
        for tc in tool_calls:
            tool_id = tc.get("id", f"call_{uuid.uuid4().hex[:8]}")
            func = tc.get("function", {})
            tool_name = func.get("name", "")

            args_raw = func.get("arguments", "{}")
            if isinstance(args_raw, str):
                try:
                    arguments = json.loads(args_raw)
                except json.JSONDecodeError:
                    arguments = {}
            else:
                arguments = args_raw

            try:
                result = await self._execute_single_tool(tool_name, arguments)
                result_str = str(result)
                results.append(
                    {"tool_call_id": tool_id, "content": result_str, "success": True}
                )

            except Exception as e:
                error_msg = f"Tool execution failed: {e}"
                results.append(
                    {"tool_call_id": tool_id, "content": error_msg, "success": False}
                )

        return results

    async def _execute_single_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        if MCP_CLIENT_AVAILABLE and self._mcp_client and tool_name in self._mcp_client.get_tool_names():
            try:
                return await asyncio.create_task(
                    self._mcp_client.execute_tool(tool_name, arguments)
                )
            except RuntimeError:
                return await asyncio.run(
                    self._mcp_client.execute_tool(tool_name, arguments)
                )

        elif TOOL_REGISTRY_AVAILABLE and self._tool_registry:
            return self._tool_registry.invoke(tool_name, **arguments)
        else:
            raise ValueError(f"Tool '{tool_name}' not found in any backend")

    def _has_execution_capability(self) -> bool:
        has_mcp = MCP_CLIENT_AVAILABLE and self._mcp_client
        has_registry = TOOL_REGISTRY_AVAILABLE and self._tool_registry
        return has_mcp or has_registry
