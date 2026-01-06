import os
from typing import List, Dict, Any, Optional

try:
    from .mcp_client import MCPToolClient, MCP_SERVER_PRESETS
    MCP_CLIENT_AVAILABLE = True
except ImportError:
    MCP_CLIENT_AVAILABLE = False
    MCPToolClient = None
    MCP_SERVER_PRESETS = {}

try:
    from toolregistry import ToolRegistry
    TOOL_REGISTRY_AVAILABLE = True
except ImportError:
    TOOL_REGISTRY_AVAILABLE = False
    ToolRegistry = None


class MCPStdioTransport:
    def __init__(self, mcp_client: Optional[MCPToolClient] = None):
        self._mcp_client = mcp_client

    async def register(
        self,
        server_name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> bool:
        if not MCP_CLIENT_AVAILABLE:
            return False

        try:
            config = {
                server_name: {
                    "command": command,
                    "args": args or [],
                    "transport": "stdio",
                }
            }
            if env:
                config[server_name]["env"] = env

            if self._mcp_client is None:
                self._mcp_client = MCPToolClient(server_configs=config)
            else:
                self._mcp_client._server_configs.update(config)

            await self._mcp_client.load_tools()
            return True

        except Exception:
            return False


class MCPHttpTransport:
    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        self._tool_registry = tool_registry

    async def register(
        self,
        server_name: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> bool:
        if not TOOL_REGISTRY_AVAILABLE:
            return False

        try:
            if self._tool_registry is None:
                self._tool_registry = ToolRegistry()

            if headers:
                try:
                    from fastmcp.client.transports import StreamableHttpTransport
                    transport = StreamableHttpTransport(url=url, headers=headers)
                except ImportError:
                    transport = url
            else:
                transport = url

            await self._tool_registry.register_from_mcp_async(transport, with_namespace=False)
            return True

        except Exception:
            return False


class MCPSseTransport:
    def __init__(self, mcp_client: Optional[MCPToolClient] = None):
        self._mcp_client = mcp_client

    async def register(
        self,
        server_name: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> bool:
        if not MCP_CLIENT_AVAILABLE:
            return False

        try:
            config = {
                server_name: {
                    "url": url,
                    "transport": "sse",
                }
            }
            if headers:
                config[server_name]["headers"] = headers

            if self._mcp_client is None:
                self._mcp_client = MCPToolClient(server_configs=config)
            else:
                self._mcp_client._server_configs.update(config)

            await self._mcp_client.load_tools()
            return True

        except Exception:
            return False


class MCPPresetTransport:
    def __init__(self, stdio_transport: MCPStdioTransport):
        self._stdio_transport = stdio_transport

    async def register(
        self,
        preset_name: str,
        env: Optional[Dict[str, str]] = None,
    ) -> bool:
        if not MCP_CLIENT_AVAILABLE:
            return False

        if preset_name not in MCP_SERVER_PRESETS:
            return False

        preset = MCP_SERVER_PRESETS[preset_name]

        final_env = {}
        if preset.get("env_key"):
            env_value = (env or {}).get(preset["env_key"]) or os.getenv(preset["env_key"])
            if env_value:
                final_env[preset["env_key"]] = env_value

        return await self._stdio_transport.register(
            server_name=preset_name,
            command=preset["command"],
            args=preset["args"],
            env=final_env,
        )


class MCPMultiServerTransport:
    def __init__(self, mcp_client: Optional[MCPToolClient] = None):
        self._mcp_client = mcp_client

    async def register_servers(self, config: Dict[str, Dict[str, Any]]) -> bool:
        if not MCP_CLIENT_AVAILABLE:
            return False

        try:
            self._mcp_client = MCPToolClient(server_configs=config)
            await self._mcp_client.load_tools()
            return True

        except Exception:
            return False
