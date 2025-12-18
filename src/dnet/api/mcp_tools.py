"""Minimal MCP Tool Provider for dnet.

Connects to external MCP servers (GitHub, HuggingFace, Exa, etc.)
and exposes their tools to dnet's inference pipeline.

Design:
- Uses fastmcp Client for MCP protocol
- Auto-discovers tools from servers
- Converts to OpenAI format (works with existing Outlines integration)
- Simple execute() method for tool calls

Usage:
    provider = MCPToolProvider({"github": {"url": "https://..."}})
    await provider.initialize()
    tools = provider.get_tools()  # OpenAI format
    result = await provider.execute("search_repos", {"query": "user:xyz"})
"""

import json
from typing import Optional, Dict, Any, List
from dnet.utils.logger import logger

# Optional import - MCP tools work only if fastmcp is installed
try:
    from fastmcp import Client
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    Client = None


class MCPToolProvider:
    """Minimal MCP tool provider using fastmcp."""

    def __init__(self, servers: Optional[Dict[str, Dict[str, Any]]] = None):
        """Initialize MCP tool provider.

        Args:
            servers: Dict of server configs. Example:
                {
                    "github": {
                        "url": "https://api.githubcopilot.com/mcp/",
                        "headers": {"Authorization": "Bearer xxx"}
                    },
                    "filesystem": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "~"]
                    }
                }
        """
        self.servers = servers or {}
        self._tools: Dict[str, Dict[str, Any]] = {}  # tool_name -> {server, schema}
        self._clients: Dict[str, Any] = {}  # server_name -> Client
        self._initialized = False

    @property
    def enabled(self) -> bool:
        """Check if MCP tools are enabled and available."""
        return FASTMCP_AVAILABLE and bool(self.servers)

    async def initialize(self) -> None:
        """Connect to MCP servers and discover tools."""
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
        """Connect to a single MCP server and discover its tools."""
        # Build fastmcp config format
        mcp_config = {"mcpServers": {server_name: config}}

        client = Client(mcp_config)
        self._clients[server_name] = client

        # Discover tools
        async with client:
            tools_list = await client.list_tools()

            for tool in tools_list:
                tool_name = tool.name

                # Get parameters schema
                params = {}
                if hasattr(tool, 'inputSchema') and tool.inputSchema:
                    params = tool.inputSchema
                elif hasattr(tool, 'parameters') and tool.parameters:
                    params = tool.parameters

                # Store tool info
                self._tools[tool_name] = {
                    "server": server_name,
                    "description": tool.description or f"Tool from {server_name}",
                    "parameters": params,
                }

                logger.debug(f"Discovered tool: {tool_name} from {server_name}")

        logger.info(f"Connected to '{server_name}': {len(tools_list)} tools")

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get all tools in OpenAI function calling format."""
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
        """Get list of available tool names."""
        return list(self._tools.keys())

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool and return the result as string.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool

        Returns:
            Tool result as string
        """
        if tool_name not in self._tools:
            return f"Error: Tool '{tool_name}' not found. Available: {list(self._tools.keys())}"

        tool_info = self._tools[tool_name]
        server_name = tool_info["server"]

        if server_name not in self._clients:
            return f"Error: Server '{server_name}' not connected"

        client = self._clients[server_name]

        try:
            logger.info(f"Executing MCP tool: {tool_name} on {server_name}")
            logger.debug(f"Tool arguments: {arguments}")

            async with client:
                result = await client.call_tool(tool_name, arguments)

                if result and result.content:
                    # Extract text content from result
                    text = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
                    
                    # Truncate very long results
                    max_len = 8000
                    if len(text) > max_len:
                        text = text[:max_len] + f"\n\n[Truncated from {len(text)} chars]"
                    
                    logger.info(f"Tool {tool_name} returned {len(text)} chars")
                    return text

                logger.warning(f"Tool {tool_name} returned empty result")
                return ""

        except Exception as e:
            error_msg = f"Tool execution failed: {e}"
            logger.error(f"MCP tool {tool_name} failed: {e}")
            return error_msg

    async def shutdown(self) -> None:
        """Clean up MCP connections."""
        self._clients.clear()
        self._tools.clear()
        self._initialized = False
        logger.debug("MCP tool provider shut down")


def load_mcp_config(config_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Load MCP server configuration from file or environment.

    Args:
        config_path: Path to JSON/YAML config file. If None, checks env vars.

    Returns:
        Dict of server configurations
    """
    import os

    # Check environment variable first
    env_config = os.environ.get("DNET_MCP_CONFIG")
    if env_config:
        try:
            return json.loads(env_config)
        except json.JSONDecodeError:
            logger.warning("Failed to parse DNET_MCP_CONFIG env var")

    # Check for config file
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

    # Check default locations
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

