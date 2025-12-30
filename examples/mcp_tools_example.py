#!/usr/bin/env python3
"""
Example: Using MCP Tools with dnet InferenceManager

This example demonstrates the CORRECT way to register and use MCP tools
with dnet's InferenceManager, following the official langchain-mcp-adapters API:
https://github.com/langchain-ai/langchain-mcp-adapters

Key Concepts:
- MCP (Model Context Protocol) uses stdio or HTTP transport, NOT plain HTTP URLs
- stdio transport spawns a subprocess that communicates via stdin/stdout
- http transport uses HTTP for remote servers (preferred over SSE)
- Tools are registered dynamically from MCP servers

Available Presets:
- "exa": Exa AI search engine (requires EXA_API_KEY)
- "github": GitHub API (requires GITHUB_TOKEN)
- "brave-search": Brave Search (requires BRAVE_API_KEY)
- "filesystem": Local filesystem access
- "fetch": HTTP fetch requests

Usage:
    # Set your API keys
    export EXA_API_KEY="your-exa-api-key"
    export GITHUB_TOKEN="your-github-token"
    
    # Run this example
    python examples/mcp_tools_example.py
"""

import asyncio
import os
from typing import Dict, Optional


async def example_using_langchain_mcp_adapters_directly():
    """Example 1: Using langchain-mcp-adapters directly (Official Pattern)
    
    This shows the official way from the langchain-mcp-adapters README.
    """
    print("\n" + "="*60)
    print("Example 1: Using langchain-mcp-adapters Directly")
    print("="*60)
    
    print("""
    # Official langchain-mcp-adapters pattern:
    
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain.agents import create_agent
    
    # Create a simple math server (math_server.py):
    '''
    from mcp.server.fastmcp import FastMCP
    
    mcp = FastMCP("Math")
    
    @mcp.tool()
    def add(a: int, b: int) -> int:
        \"\"\"Add two numbers\"\"\"
        return a + b
    
    @mcp.tool()
    def multiply(a: int, b: int) -> int:
        \"\"\"Multiply two numbers\"\"\"
        return a * b
    
    if __name__ == "__main__":
        mcp.run(transport="stdio")
    '''
    
    # Configure client with multiple servers:
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
    
    # Load tools
    tools = await client.get_tools()
    
    # Create agent
    agent = create_agent("openai:gpt-4.1", tools)
    response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
    """)


async def example_using_mcp_sdk_directly():
    """Example 2: Using MCP SDK directly with load_mcp_tools"""
    print("\n" + "="*60)
    print("Example 2: Using MCP SDK with load_mcp_tools")
    print("="*60)
    
    print("""
    # Using MCP SDK directly with langchain_mcp_adapters.tools.load_mcp_tools:
    
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from langchain_mcp_adapters.tools import load_mcp_tools
    
    server_params = StdioServerParameters(
        command="python",
        args=["/path/to/math_server.py"],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            
            # Get tools (returns LangChain-compatible tools)
            tools = await load_mcp_tools(session)
            
            # Use with agent
            from langchain.agents import create_agent
            agent = create_agent("openai:gpt-4.1", tools)
            response = await agent.ainvoke({"messages": "what's 3 + 5?"})
    """)


async def example_dnet_preset_registration():
    """Example 3: Using dnet's Preset Registration (Easiest Approach)"""
    print("\n" + "="*60)
    print("Example 3: dnet Preset Registration")
    print("="*60)
    
    from dnet.api.mcp_client import MCP_SERVER_PRESETS
    
    print("\nAvailable presets:")
    for name, config in MCP_SERVER_PRESETS.items():
        env_info = f" (requires {config['env_key']})" if config.get("env_key") else ""
        print(f"  - {name}: {config.get('description', 'No description')}{env_info}")
    
    print("""
    
    # In your dnet InferenceManager:
    
    # Option 1: Use a preset
    await inference_manager.register_mcp_preset("exa", env={"EXA_API_KEY": "..."})
    
    # Option 2: Register stdio server manually
    await inference_manager.register_mcp_stdio(
        server_name="my-server",
        command="python",
        args=["/path/to/server.py"],
        env={"MY_API_KEY": "..."}
    )
    
    # Option 3: Register HTTP server
    await inference_manager.register_mcp_http(
        server_name="weather",
        url="http://localhost:8000/mcp",
        headers={"Authorization": "Bearer token"}
    )
    
    # Option 4: Register multiple servers at once (langchain-mcp-adapters format)
    await inference_manager.register_mcp_servers({
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
    """)


async def example_http_transport():
    """Example 4: HTTP Transport (Preferred for Remote Servers)"""
    print("\n" + "="*60)
    print("Example 4: HTTP Transport for Remote MCP Servers")
    print("="*60)
    
    print("""
    # HTTP transport is now preferred over SSE for remote servers.
    
    # Server (using FastMCP):
    from mcp.server.fastmcp import FastMCP
    
    mcp = FastMCP("Weather")
    
    @mcp.tool()
    async def get_weather(location: str) -> str:
        \"\"\"Get weather for location.\"\"\"
        return f"It's sunny in {location}"
    
    if __name__ == "__main__":
        mcp.run(transport="http")  # Runs on port 8000 by default
    
    # Client:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    
    client = MultiServerMCPClient({
        "weather": {
            "url": "http://localhost:8000/mcp",
            "transport": "http",
        }
    })
    
    # With authentication headers:
    client = MultiServerMCPClient({
        "weather": {
            "url": "http://localhost:8000/mcp",
            "transport": "http",
            "headers": {
                "Authorization": "Bearer YOUR_TOKEN",
                "X-Custom-Header": "custom-value"
            }
        }
    })
    """)


async def example_with_langgraph():
    """Example 5: Using with LangGraph StateGraph"""
    print("\n" + "="*60)
    print("Example 5: Using MCP Tools with LangGraph StateGraph")
    print("="*60)
    
    print("""
    # Full LangGraph integration example:
    
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langgraph.graph import StateGraph, MessagesState, START
    from langgraph.prebuilt import ToolNode, tools_condition
    from langchain.chat_models import init_chat_model
    
    model = init_chat_model("openai:gpt-4.1")
    
    client = MultiServerMCPClient({
        "math": {
            "command": "python",
            "args": ["./math_server.py"],
            "transport": "stdio",
        },
        "weather": {
            "url": "http://localhost:8000/mcp",
            "transport": "http",
        }
    })
    
    tools = await client.get_tools()
    
    def call_model(state: MessagesState):
        response = model.bind_tools(tools).invoke(state["messages"])
        return {"messages": response}
    
    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_node(ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)
    builder.add_edge("tools", "call_model")
    
    graph = builder.compile()
    
    math_response = await graph.ainvoke({"messages": "what's (3 + 5) x 12?"})
    weather_response = await graph.ainvoke({"messages": "what is the weather in nyc?"})
    """)


async def example_tool_execution_flow():
    """Example 6: Tool Execution Flow in dnet"""
    print("\n" + "="*60)
    print("Example 6: Tool Execution Flow in dnet")
    print("="*60)
    
    print("""
    # When using execute_tools_and_continue():
    
    # 1. User sends a message that might need tools
    # 2. LLM generates response with tool_calls
    # 3. System executes the tools via MCP
    # 4. Results are fed back to LLM
    # 5. LLM synthesizes final response
    
    from dnet.api.models import ChatRequestModel, ChatMessage
    
    request = ChatRequestModel(
        model="qwen3-8b",
        messages=[
            ChatMessage(role="user", content="Search for latest AI news")
        ],
        tools=inference_manager.get_all_tools_openai_format()
    )
    
    # This handles the full agent loop
    response = await inference_manager.execute_tools_and_continue(request)
    
    print(response.choices[0].message.content)
    """)


async def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("MCP Tools Integration Examples for dnet")
    print("="*60)
    print("""
    Based on the official langchain-mcp-adapters library:
    https://github.com/langchain-ai/langchain-mcp-adapters
    
    MCP Transport Types:
    
    ✅ stdio - Spawns subprocess (for local servers)
       {"command": "python", "args": ["server.py"], "transport": "stdio"}
    
    ✅ http - HTTP transport (preferred for remote servers)
       {"url": "http://localhost:8000/mcp", "transport": "http"}
    
    ✅ sse - Server-Sent Events (legacy, still supported)
       {"url": "http://localhost:8000/mcp/sse", "transport": "sse"}
    
    ❌ WRONG: Plain HTTP URLs like "https://exa-mcp.com"
       MCP does NOT work this way!
    """)
    
    await example_using_langchain_mcp_adapters_directly()
    await example_using_mcp_sdk_directly()
    await example_dnet_preset_registration()
    await example_http_transport()
    await example_with_langgraph()
    await example_tool_execution_flow()
    
    print("\n" + "="*60)
    print("Summary: Getting Started")
    print("="*60)
    print("""
    1. Install dependencies:
       pip install langchain-mcp-adapters mcp
    
    2. For stdio servers (like Exa), install Node.js and npx:
       npm install -g npx
    
    3. Set environment variables:
       export EXA_API_KEY="your-key"
       export GITHUB_TOKEN="your-token"
    
    4. In your code:
       # Simple preset
       await inference_manager.register_mcp_preset("exa")
       
       # Or full config
       await inference_manager.register_mcp_servers({
           "exa": {
               "command": "npx",
               "args": ["-y", "@anthropic-ai/exa-mcp-server"],
               "transport": "stdio",
               "env": {"EXA_API_KEY": os.getenv("EXA_API_KEY")}
           }
       })
    
    For more info, see:
    - https://github.com/langchain-ai/langchain-mcp-adapters
    - dnet/src/dnet/api/mcp_client.py
    - dnet/src/dnet/api/inference.py
    """)


if __name__ == "__main__":
    asyncio.run(main())
