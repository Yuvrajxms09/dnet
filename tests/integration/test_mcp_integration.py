"""Integration tests for MCP server.

These tests validate that MCP tools work end-to-end through HTTP endpoint.

Usage (with servers running):
    uv run pytest tests/integration/test_mcp_integration.py -v -x

Usage (standalone - starts servers automatically):
    uv run pytest tests/integration/test_mcp_integration.py -v -x --start-servers

Usage (in CI - expects servers started externally):
    uv run pytest tests/integration/test_mcp_integration.py -m integration -v -x
"""

import json
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Any, Generator

import pytest
import requests

from dnet.api.catalog import get_ci_test_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_HTTP_PORT = 8080
API_GRPC_PORT = 58080
SHARD_HTTP_PORT = 8081
SHARD_GRPC_PORT = 58081
BASE_URL = f"http://localhost:{API_HTTP_PORT}"
MCP_URL = f"{BASE_URL}/mcp"

HEALTH_CHECK_TIMEOUT = 60
MODEL_LOAD_TIMEOUT = 300
INFERENCE_TIMEOUT = 120


def wait_for_health(url: str, timeout: float = HEALTH_CHECK_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{url}/health", timeout=2)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


@pytest.fixture(scope="module")
def servers(start_servers_flag) -> Generator[None, None, None]:
    procs: list[subprocess.Popen] = []

    if start_servers_flag:
        shard_cmd = [
            sys.executable,
            "-m",
            "cli.shard",
            "--http-port",
            str(SHARD_HTTP_PORT),
            "--grpc-port",
            str(SHARD_GRPC_PORT),
        ]
        shard_proc = subprocess.Popen(
            shard_cmd,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            env={**os.environ, "PYTHONPATH": "src"},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        procs.append(shard_proc)

        if not wait_for_health(f"http://localhost:{SHARD_HTTP_PORT}", timeout=30):
            shard_proc.terminate()
            try:
                shard_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                shard_proc.kill()
                shard_proc.wait()
            pytest.skip(f"Shard server not healthy at port {SHARD_HTTP_PORT}")

        api_cmd = [
            sys.executable,
            "-m",
            "cli.api",
            "--http-port",
            str(API_HTTP_PORT),
            "--grpc-port",
            str(API_GRPC_PORT),
        ]
        api_proc = subprocess.Popen(
            api_cmd,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            env={**os.environ, "PYTHONPATH": "src"},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        procs.append(api_proc)

    if not wait_for_health(BASE_URL):
        for p in procs:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        pytest.skip(f"Server not healthy at {BASE_URL}/health")

    yield

    for p in procs:
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def mcp_call_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        from fastmcp.client import Client
        from fastmcp.client.transports import HTTPTransport
    except ImportError:
        pytest.skip("fastmcp not available")

    async def _call():
        transport = HTTPTransport(MCP_URL)
        async with Client(transport=transport) as client:
            return await client.call_tool(name=tool_name, arguments=arguments)

    import asyncio

    return asyncio.run(_call())


def prepare_and_load_model_mcp(model_id: str) -> None:
    result = mcp_call_tool("load_model", {"model": model_id})
    assert result.data is not None
    assert "loaded successfully" in result.data.lower()


def unload_model_mcp() -> None:
    try:
        mcp_call_tool("unload_model", {})
    except Exception as e:
        logger.warning(f"Failed to unload model via MCP (best effort): {e}")


CI_TEST_MODELS = get_ci_test_models()


@pytest.mark.integration
def test_mcp_health_check(servers):
    resp = requests.get(f"{MCP_URL}/mcp-health", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "dnet-mcp"


@pytest.mark.integration
def test_mcp_list_models(servers):
    result = mcp_call_tool("list_models", {})
    assert result.data is not None
    assert isinstance(result.data, str)
    data = json.loads(result.data)
    assert "object" in data
    assert data["object"] == "list"
    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0


@pytest.mark.integration
def test_mcp_get_status_no_model(servers):
    result = mcp_call_tool("get_status", {})
    assert result.data is not None
    assert isinstance(result.data, str)
    data = json.loads(result.data)
    assert "model_loaded" in data
    assert "shards_discovered" in data


@pytest.mark.integration
@pytest.mark.parametrize(
    "model",
    CI_TEST_MODELS[:1],
    ids=[m["alias"] for m in CI_TEST_MODELS[:1]],
)
def test_mcp_load_and_chat(servers, model: dict[str, Any]):
    model_id = model["id"]
    try:
        prepare_and_load_model_mcp(model_id)

        result = mcp_call_tool(
            "chat_completion",
            {
                "messages": [
                    {"role": "user", "content": "What is 2+2? Reply briefly."}
                ],
                "max_tokens": 50,
                "temperature": 0.1,
            },
        )
        assert result.data is not None
        assert isinstance(result.data, str)
        assert len(result.data) > 0
        assert result.data.strip()

    finally:
        unload_model_mcp()


@pytest.mark.integration
def test_mcp_get_cluster_details(servers):
    result = mcp_call_tool("get_cluster_details", {})
    assert result.data is not None
    assert isinstance(result.data, str)
    data = json.loads(result.data)
    assert "devices" in data
    assert "topology" in data
