import json
import asyncio
from typing import Optional, Dict, Any, List
from dnet.utils.logger import logger

try:
    from fastmcp import Client
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    Client = None


class MCPToolProvider:
    def __init__(
        self, 
        servers: Optional[Dict[str, Dict[str, Any]]] = None,
        max_retries: int = 3,
        timeout_seconds: float = 30.0,
    ):
        self.servers = servers or {}
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._clients: Dict[str, Any] = {}
        self._initialized = False
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

    @property
    def enabled(self) -> bool:
        return FASTMCP_AVAILABLE and bool(self.servers)

    async def initialize(self) -> None:
        if not FASTMCP_AVAILABLE:
            logger.warning("fastmcp not installed, MCP tools disabled")
            return

        if not self.servers:
            logger.debug("No MCP servers configured")
            return

        logger.info(f"Initializing MCP connections to {len(self.servers)} servers...")

        for server_name, config in self.servers.items():
            try:
                await self._connect_server(server_name, config)
            except Exception as e:
                logger.warning(f"Failed to connect to MCP server '{server_name}': {e}")

        self._initialized = True
        logger.info(f"MCP initialized: {len(self._tools)} tools from {len(self._clients)} servers")

    async def _connect_server(self, server_name: str, config: Dict[str, Any]) -> None:
        mcp_config = {"mcpServers": {server_name: config}}
        client = Client(mcp_config)
        self._clients[server_name] = client

        async with client:
            tools_list = await client.list_tools()

            for tool in tools_list:
                tool_name = tool.name
                params = {}
                if hasattr(tool, 'inputSchema') and tool.inputSchema:
                    params = tool.inputSchema
                elif hasattr(tool, 'parameters') and tool.parameters:
                    params = tool.parameters

                self._tools[tool_name] = {
                    "server": server_name,
                    "description": tool.description or f"Tool from {server_name}",
                    "parameters": params,
                }
                logger.debug(f"Discovered tool: {tool_name} from {server_name}")

        logger.info(f"Connected to '{server_name}': {len(tools_list)} tools")

    def get_tools(self) -> List[Dict[str, Any]]:
        tools = []
        for name, info in self._tools.items():
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["description"],
                    "parameters": info["parameters"],
                }
            })
        return tools

    def get_tool_names(self) -> List[str]:
        return list(self._tools.keys())

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name not in self._tools:
            return f"Error: Tool '{tool_name}' not found. Available: {list(self._tools.keys())}"

        tool_info = self._tools[tool_name]
        server_name = tool_info["server"]

        if server_name not in self._clients:
            return f"Error: Server '{server_name}' not connected"

        client = self._clients[server_name]
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Executing MCP tool: {tool_name} on {server_name} (attempt {attempt + 1}/{self.max_retries + 1})")
                logger.debug(f"Tool arguments: {arguments}")

                async with client:
                    result = await asyncio.wait_for(
                        client.call_tool(tool_name, arguments),
                        timeout=self.timeout_seconds
                    )

                    if result and result.content:
                        text = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
                        logger.info(f"Tool {tool_name} succeeded: {len(text)} chars")
                        return text

                    logger.warning(f"Tool {tool_name} returned empty result")
                    return ""

            except asyncio.TimeoutError:
                last_error = f"Tool execution timed out after {self.timeout_seconds}s"
                logger.warning(f"MCP tool {tool_name} timeout (attempt {attempt + 1}/{self.max_retries + 1})")
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"MCP tool {tool_name} failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                
                if "not found" in str(e).lower() or "invalid" in str(e).lower():
                    break

            if attempt < self.max_retries:
                wait_time = 2 ** attempt
                logger.debug(f"Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

        error_msg = f"Tool execution failed after {self.max_retries + 1} attempts: {last_error}"
        logger.error(f"MCP tool {tool_name} failed: {error_msg}")
        return f"Error: {error_msg}"

    async def shutdown(self) -> None:
        self._clients.clear()
        self._tools.clear()
        self._initialized = False
        logger.debug("MCP tool provider shut down")


def load_mcp_config(config_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    import os

    env_config = os.environ.get("DNET_MCP_CONFIG")
    if env_config:
        try:
            return json.loads(env_config)
        except json.JSONDecodeError:
            logger.warning("Failed to parse DNET_MCP_CONFIG env var")

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path) as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    try:
                        import yaml
                        return yaml.safe_load(f) or {}
                    except ImportError:
                        logger.warning("PyYAML not installed, skipping YAML config")
                else:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load MCP config from {config_path}: {e}")

    default_paths = [
        os.path.expanduser("~/.dnet/mcp_config.json"),
        os.path.expanduser("~/.config/dnet/mcp.json"),
        "./mcp_config.json",
    ]

    for path in default_paths:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    config = json.load(f)
                    logger.info(f"Loaded MCP config from {path}")
                    return config
            except Exception:
                continue

    return {}

