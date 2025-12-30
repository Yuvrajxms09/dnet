"""MCP Client for integrating external MCP tools with dnet.

This module provides proper MCP integration using the official langchain-mcp-adapters
package, following the API documented at:
https://github.com/langchain-ai/langchain-mcp-adapters

MCP Transport Types:
- stdio: Spawns a subprocess (most common for local MCP servers)
- http: HTTP transport (for remote MCP servers) - preferred over SSE
- sse: Server-Sent Events over HTTP (legacy, still supported)

Example Usage (following official langchain-mcp-adapters patterns):

    # Using MultiServerMCPClient (recommended)
    from langchain_mcp_adapters.client import MultiServerMCPClient
    
    client = MultiServerMCPClient({
        "math": {
            "command": "python",
            "args": ["/path/to/math_server.py"],
            "transport": "stdio",
        },
        "weather": {
            "url": "http://localhost:8000/mcp",
            "transport": "http",
        }
    })
    tools = await client.get_tools()
    
    # Using with MCP SDK directly
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from langchain_mcp_adapters.tools import load_mcp_tools
    
    server_params = StdioServerParameters(
        command="python",
        args=["/path/to/server.py"],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field

from dnet.utils.logger import logger


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection.
    
    Matches the config format used by langchain-mcp-adapters MultiServerMCPClient:
    
    For stdio transport:
        {"command": "python", "args": ["/path/to/server.py"], "transport": "stdio", "env": {...}}
    
    For http transport:
        {"url": "http://localhost:8000/mcp", "transport": "http", "headers": {...}}
    
    For sse transport (legacy):
        {"url": "http://localhost:8000/mcp/sse", "transport": "sse", "headers": {...}}
    """
    name: str
    transport: str  # "stdio", "http", or "sse"
    command: Optional[str] = None  # For stdio transport
    args: Optional[List[str]] = None  # For stdio transport
    env: Optional[Dict[str, str]] = None  # Environment variables for stdio
    url: Optional[str] = None  # For http/sse transport
    headers: Optional[Dict[str, str]] = None  # For http/sse transport
    
    def to_langchain_config(self) -> Dict[str, Any]:
        """Convert to langchain-mcp-adapters config format."""
        config: Dict[str, Any] = {"transport": self.transport}
        
        if self.transport == "stdio":
            config["command"] = self.command
            config["args"] = self.args or []
            if self.env:
                config["env"] = self.env
        else:  # http or sse
            config["url"] = self.url
            if self.headers:
                config["headers"] = self.headers
        
        return config


@dataclass
class MCPTool:
    """Represents an MCP tool definition."""
    name: str
    description: str
    parameters: Dict[str, Any]
    server_name: str
    _langchain_tool: Optional[Any] = None  # Reference to LangChain tool for execution
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


class MCPToolClient:
    """
    Client for connecting to MCP servers and retrieving/executing tools.
    
    Uses langchain-mcp-adapters MultiServerMCPClient as the primary implementation,
    with fallback to direct MCP SDK usage.
    
    Example:
        client = MCPToolClient()
        
        # Configure servers
        client.add_stdio_server(
            "math",
            command="python",
            args=["/path/to/math_server.py"]
        )
        client.add_http_server(
            "weather",
            url="http://localhost:8000/mcp"
        )
        
        # Load tools (connects to all servers)
        await client.load_tools()
        
        # Get tools in OpenAI format
        tools = client.get_tools_openai_format()
    """
    
    def __init__(self, server_configs: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize MCP client.
        
        Args:
            server_configs: Optional dict of server configurations in langchain-mcp-adapters format:
                {
                    "server_name": {
                        "command": "...",
                        "args": [...],
                        "transport": "stdio",
                    },
                    "other_server": {
                        "url": "http://...",
                        "transport": "http",
                    }
                }
        """
        self._server_configs: Dict[str, Dict[str, Any]] = server_configs or {}
        self._tools: Dict[str, MCPTool] = {}
        self._langchain_tools: List[Any] = []  # LangChain BaseTool instances
        self._mcp_client = None
        self._connected: bool = False
        
        # Check available libraries
        self._langchain_available = self._check_langchain_adapters()
        self._mcp_sdk_available = self._check_mcp_sdk()
    
    def _check_langchain_adapters(self) -> bool:
        """Check if langchain-mcp-adapters is available."""
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            from langchain_mcp_adapters.tools import load_mcp_tools
            logger.debug("✅ langchain-mcp-adapters available")
            return True
        except ImportError:
            logger.debug("langchain-mcp-adapters not available")
            return False
    
    def _check_mcp_sdk(self) -> bool:
        """Check if MCP SDK is available."""
        try:
            from mcp import ClientSession, StdioServerParameters
            logger.debug("✅ MCP SDK available")
            return True
        except ImportError:
            logger.debug("MCP SDK not available")
            return False
    
    def add_stdio_server(
        self,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None
    ) -> 'MCPToolClient':
        """
        Add a stdio transport MCP server configuration.
        
        Args:
            name: Unique server name
            command: Command to run (e.g., "python", "npx", "node")
            args: Command arguments
            env: Environment variables
            
        Returns:
            self for method chaining
        """
        self._server_configs[name] = {
            "command": command,
            "args": args or [],
            "transport": "stdio",
        }
        if env:
            self._server_configs[name]["env"] = env
        
        logger.debug(f"📝 Added stdio server config: {name}")
        return self
    
    def add_http_server(
        self,
        name: str,
        url: str,
        headers: Optional[Dict[str, str]] = None
    ) -> 'MCPToolClient':
        """
        Add an HTTP transport MCP server configuration.
        
        This is the preferred transport for remote MCP servers.
        
        Args:
            name: Unique server name
            url: Server URL (e.g., "http://localhost:8000/mcp")
            headers: Optional HTTP headers (e.g., for authentication)
            
        Returns:
            self for method chaining
        """
        self._server_configs[name] = {
            "url": url,
            "transport": "http",
        }
        if headers:
            self._server_configs[name]["headers"] = headers
        
        logger.debug(f"📝 Added http server config: {name}")
        return self
    
    def add_sse_server(
        self,
        name: str,
        url: str,
        headers: Optional[Dict[str, str]] = None
    ) -> 'MCPToolClient':
        """
        Add an SSE transport MCP server configuration.
        
        Note: HTTP transport is now preferred over SSE for remote servers.
        
        Args:
            name: Unique server name
            url: SSE endpoint URL
            headers: Optional HTTP headers
            
        Returns:
            self for method chaining
        """
        self._server_configs[name] = {
            "url": url,
            "transport": "sse",
        }
        if headers:
            self._server_configs[name]["headers"] = headers
        
        logger.debug(f"📝 Added sse server config: {name}")
        return self
    
    async def load_tools(self) -> List[Any]:
        """
        Connect to all configured servers and load tools.
        
        Returns:
            List of LangChain-compatible tools
        """
        if not self._server_configs:
            logger.warning("⚠️ No MCP servers configured")
            return []
        
        if self._langchain_available:
            return await self._load_tools_langchain()
        elif self._mcp_sdk_available:
            return await self._load_tools_direct()
        else:
            logger.error("❌ No MCP libraries available. Install: pip install langchain-mcp-adapters")
            return []
    
    async def _load_tools_langchain(self) -> List[Any]:
        """Load tools using langchain-mcp-adapters MultiServerMCPClient."""
        from langchain_mcp_adapters.client import MultiServerMCPClient
        
        logger.info(f"🔌 Connecting to {len(self._server_configs)} MCP server(s)...")
        
        try:
            self._mcp_client = MultiServerMCPClient(self._server_configs)
            
            # Get tools - this connects to all servers
            self._langchain_tools = await self._mcp_client.get_tools()
            
            # Register tools
            for tool in self._langchain_tools:
                # Extract schema from LangChain tool
                parameters = {}
                if hasattr(tool, 'args_schema') and tool.args_schema:
                    try:
                        parameters = tool.args_schema.model_json_schema()
                    except Exception:
                        parameters = {}
                
                mcp_tool = MCPTool(
                    name=tool.name,
                    description=getattr(tool, 'description', ''),
                    parameters=parameters,
                    server_name="multi",  # MultiServerMCPClient doesn't expose server per tool
                    _langchain_tool=tool
                )
                self._tools[tool.name] = mcp_tool
                logger.debug(f"   📋 Loaded tool: {tool.name}")
            
            self._connected = True
            logger.info(f"✅ Loaded {len(self._langchain_tools)} tools from MCP servers")
            return self._langchain_tools
            
        except Exception as e:
            logger.error(f"❌ Failed to load MCP tools: {e}")
            raise
    
    async def _load_tools_direct(self) -> List[Any]:
        """Load tools using MCP SDK directly (fallback)."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        
        all_tools = []
        
        for name, config in self._server_configs.items():
            transport = config.get("transport", "stdio")
            
            if transport == "stdio":
                try:
                    env = os.environ.copy()
                    if config.get("env"):
                        env.update(config["env"])
                    
                    server_params = StdioServerParameters(
                        command=config["command"],
                        args=config.get("args", []),
                        env=env
                    )
                    
                    async with stdio_client(server_params) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            
                            # Try to use langchain adapter if available
                            try:
                                from langchain_mcp_adapters.tools import load_mcp_tools
                                tools = await load_mcp_tools(session)
                                all_tools.extend(tools)
                                
                                for tool in tools:
                                    mcp_tool = MCPTool(
                                        name=tool.name,
                                        description=getattr(tool, 'description', ''),
                                        parameters={},
                                        server_name=name,
                                        _langchain_tool=tool
                                    )
                                    self._tools[tool.name] = mcp_tool
                                    
                            except ImportError:
                                # Fall back to raw tool list
                                response = await session.list_tools()
                                for tool in response.tools:
                                    mcp_tool = MCPTool(
                                        name=tool.name,
                                        description=tool.description or "",
                                        parameters=getattr(tool, 'inputSchema', {}),
                                        server_name=name
                                    )
                                    self._tools[tool.name] = mcp_tool
                            
                            logger.info(f"✅ Connected to '{name}'")
                            
                except Exception as e:
                    logger.error(f"❌ Failed to connect to '{name}': {e}")
            else:
                logger.warning(f"⚠️ Direct SDK doesn't support '{transport}' transport yet")
        
        self._langchain_tools = all_tools
        self._connected = True
        return all_tools
    
    def get_tools(self) -> List[MCPTool]:
        """Get all registered tools as MCPTool objects."""
        return list(self._tools.values())
    
    def get_langchain_tools(self) -> List[Any]:
        """Get LangChain-compatible tool objects."""
        return self._langchain_tools
    
    def get_tools_openai_format(self) -> List[Dict[str, Any]]:
        """Get all tools in OpenAI-compatible format."""
        return [tool.to_openai_format() for tool in self._tools.values()]
    
    def get_tool_names(self) -> List[str]:
        """Get names of all registered tools."""
        return list(self._tools.keys())
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool by name with given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            
        Returns:
            Tool execution result
        """
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not found. Available: {list(self._tools.keys())}")
        
        mcp_tool = self._tools[tool_name]
        logger.info(f"🔧 Executing tool: {tool_name}")
        logger.debug(f"   Arguments: {arguments}")
        
        try:
            if mcp_tool._langchain_tool is not None:
                # Use LangChain tool's invoke method
                tool = mcp_tool._langchain_tool
                if hasattr(tool, 'ainvoke'):
                    result = await tool.ainvoke(arguments)
                elif hasattr(tool, 'invoke'):
                    result = tool.invoke(arguments)
                else:
                    result = tool(arguments)
            else:
                raise NotImplementedError(
                    f"Tool '{tool_name}' doesn't have an executor. "
                    "Load tools with load_mcp_tools() for execution support."
                )
            
            logger.info(f"✅ Tool '{tool_name}' executed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Tool execution failed: {e}")
            raise
    
    async def __aenter__(self):
        """Async context manager entry - loads tools."""
        await self.load_tools()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Cleanup if needed
        pass


# Predefined MCP server configurations for popular services
MCP_SERVER_PRESETS: Dict[str, Dict[str, Any]] = {
    # Exa - Remote HTTP (recommended, no local install needed)
    "exa-http": {
        "transport": "http",
        "url_template": "https://mcp.exa.ai/mcp?exaApiKey={EXA_API_KEY}&tools=web_search_exa,get_code_context_exa",
        "env_key": "EXA_API_KEY",
        "description": "Exa AI search (remote HTTP - recommended)"
    },
    # Exa - Local stdio (requires npx)
    "exa": {
        "command": "npx",
        "args": ["-y", "exa-mcp-server"],
        "transport": "stdio",
        "env_key": "EXA_API_KEY",
        "description": "Exa AI search engine (local stdio)"
    },
    "github": {
        "command": "npx", 
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "transport": "stdio",
        "env_key": "GITHUB_TOKEN",
        "description": "GitHub API integration"
    },
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem"],
        "transport": "stdio",
        "description": "Local filesystem access"
    },
    "brave-search": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        "transport": "stdio",
        "env_key": "BRAVE_API_KEY",
        "description": "Brave Search API"
    },
    "fetch": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-fetch"],
        "transport": "stdio",
        "description": "HTTP fetch requests"
    }
}


def create_mcp_client_from_config(config: Dict[str, Dict[str, Any]]) -> MCPToolClient:
    """
    Create an MCP client from a configuration dict.
    
    This matches the langchain-mcp-adapters MultiServerMCPClient config format.
    
    Args:
        config: Server configurations in langchain-mcp-adapters format
        
    Returns:
        Configured MCPToolClient (not yet connected, call load_tools())
        
    Example:
        client = create_mcp_client_from_config({
            "math": {
                "command": "python",
                "args": ["/path/to/math_server.py"],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/mcp",
                "transport": "http",
            }
        })
        await client.load_tools()
    """
    return MCPToolClient(server_configs=config)


async def create_mcp_client_from_presets(
    presets: List[str],
    additional_env: Optional[Dict[str, str]] = None
) -> MCPToolClient:
    """
    Create an MCP client with predefined server configurations.
    
    Args:
        presets: List of preset names (e.g., ["exa-http", "github"])
        additional_env: Additional environment variables
        
    Returns:
        Configured and connected MCPToolClient
        
    Example:
        client = await create_mcp_client_from_presets(
            presets=["exa-http", "github"],
            additional_env={"EXA_API_KEY": "...", "GITHUB_TOKEN": "..."}
        )
    """
    configs: Dict[str, Dict[str, Any]] = {}
    
    for preset_name in presets:
        if preset_name not in MCP_SERVER_PRESETS:
            logger.warning(f"⚠️ Unknown MCP preset: {preset_name}")
            continue
        
        preset = MCP_SERVER_PRESETS[preset_name]
        transport = preset.get("transport", "stdio")
        
        # Handle HTTP transport (remote servers)
        if transport == "http":
            url_template = preset.get("url_template", "")
            
            # Get API key if needed
            if preset.get("env_key"):
                env_value = (additional_env or {}).get(preset["env_key"]) or os.getenv(preset["env_key"])
                if env_value:
                    # Substitute API key in URL template
                    url = url_template.replace(f"{{{preset['env_key']}}}", env_value)
                    configs[preset_name] = {
                        "url": url,
                        "transport": "http",
                    }
                else:
                    logger.warning(f"⚠️ {preset['env_key']} not set for {preset_name}")
            else:
                configs[preset_name] = {
                    "url": url_template,
                    "transport": "http",
                }
        
        # Handle stdio transport (local servers)
        else:
            config: Dict[str, Any] = {
                "command": preset["command"],
                "args": preset["args"],
                "transport": "stdio",
            }
            
            # Build environment
            if preset.get("env_key"):
                env_value = (additional_env or {}).get(preset["env_key"]) or os.getenv(preset["env_key"])
                if env_value:
                    config["env"] = {preset["env_key"]: env_value}
                else:
                    logger.warning(f"⚠️ {preset['env_key']} not set for {preset_name}")
            
            configs[preset_name] = config
    
    client = MCPToolClient(server_configs=configs)
    await client.load_tools()
    return client

