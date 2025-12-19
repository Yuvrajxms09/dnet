"""Tests: MCP handler tools and resources (chat, load/unload, status, models)."""

import json
import pytest

try:
    from fastmcp.client import Client
    from fastmcp.client.transports import FastMCPTransport
except ImportError:
    pytest.skip("fastmcp not available", allow_module_level=True)

from dnet_p2p import DnetDeviceProperties
from dnet.core.types.topology import TopologyInfo, LayerAssignment
from dnet.api.mcp_handler import create_mcp_server

from tests.fakes import (
    FakeClusterManager,
    FakeInferenceManager,
    FakeModelManager,
    FakeProps,
)

pytestmark = [pytest.mark.api, pytest.mark.mcp]


def _create_mcp_server(cm, im, mm):
    return create_mcp_server(im, mm, cm)


@pytest.fixture
async def mcp_client():
    cm = FakeClusterManager({})
    im = FakeInferenceManager()
    mm = FakeModelManager(current_model_id=None)
    server = _create_mcp_server(cm, im, mm)
    async with Client(transport=server) as client:
        yield client, cm, im, mm


async def test_list_models(mcp_client):
    client, cm, im, mm = mcp_client
    result = await client.call_tool(name="list_models", arguments={})
    assert result.data is not None
    assert isinstance(result.data, str)
    data = json.loads(result.data)
    assert "object" in data
    assert data["object"] == "list"
    assert "data" in data
    assert isinstance(data["data"], list)


async def test_get_status_no_model(mcp_client):
    client, cm, im, mm = mcp_client
    result = await client.call_tool(name="get_status", arguments={})
    assert result.data is not None
    assert isinstance(result.data, str)
    data = json.loads(result.data)
    assert "model_loaded" in data
    assert data["model_loaded"] is None
    assert "shards_discovered" in data
    assert data["shards_discovered"] == 0


async def test_get_cluster_details_empty(mcp_client):
    client, cm, im, mm = mcp_client
    result = await client.call_tool(name="get_cluster_details", arguments={})
    assert result.data is not None
    assert isinstance(result.data, str)
    data = json.loads(result.data)
    assert "devices" in data
    assert "topology" in data
    assert data["devices"] == {}
    assert data["topology"] is None


async def test_chat_completion_no_model_raises_error(mcp_client):
    client, cm, im, mm = mcp_client
    with pytest.raises(Exception) as exc_info:
        await client.call_tool(
            name="chat_completion",
            arguments={"messages": [{"role": "user", "content": "Hello"}]},
        )
    assert exc_info.value is not None


async def test_chat_completion_with_model_success(mcp_client):
    client, cm, im, mm = mcp_client
    mm.current_model_id = "test-model"
    result = await client.call_tool(
        name="chat_completion",
        arguments={"messages": [{"role": "user", "content": "Hello"}]},
    )
    assert result.data is not None
    assert isinstance(result.data, str)
    assert result.data == "ok"


async def test_unload_model_no_model_loaded(mcp_client):
    client, cm, im, mm = mcp_client
    result = await client.call_tool(name="unload_model", arguments={})
    assert result.data is not None
    assert isinstance(result.data, str)
    assert "No model is currently loaded" in result.data


async def test_unload_model_success(mcp_client):
    client, cm, im, mm = mcp_client
    mm.current_model_id = "test-model"
    shards = {
        "S1": FakeProps("S1", "127.0.0.1", 8001, is_manager=False),
    }
    cm.shards = shards
    mm.unload_success = True
    result = await client.call_tool(name="unload_model", arguments={})
    assert result.data is not None
    assert isinstance(result.data, str)
    assert "unloaded successfully" in result.data
    assert cm.current_topology is None


async def test_unload_model_failure_raises_error(mcp_client):
    client, cm, im, mm = mcp_client
    mm.current_model_id = "test-model"
    shards = {
        "S1": FakeProps("S1", "127.0.0.1", 8001, is_manager=False),
    }
    cm.shards = shards
    mm.unload_success = False
    with pytest.raises(Exception) as exc_info:
        await client.call_tool("unload_model", {})
    assert exc_info.value is not None


async def test_load_model_bootstrap_no_topology(monkeypatch):
    cm = FakeClusterManager({})
    im = FakeInferenceManager()
    mm = FakeModelManager(current_model_id=None)
    server = _create_mcp_server(cm, im, mm)

    monkeypatch.setattr(
        "dnet.api.load_helpers.get_model_config_json",
        lambda m: {"hidden_size": 8, "num_hidden_layers": 4},
        raising=True,
    )

    async def _prof(model_id, emb, maxb, batches):
        return {}

    monkeypatch.setattr(cm, "profile_cluster", _prof, raising=True)

    async with Client(transport=server) as client:
        with pytest.raises(Exception) as exc_info:
            await client.call_tool(
                name="load_model",
                arguments={"model": "m", "kv_bits": "8bit", "seq_len": 64},
            )
        assert exc_info.value is not None


async def test_load_model_bootstrap_success_connects(monkeypatch):
    cm = FakeClusterManager({})
    im = FakeInferenceManager(grpc_port=55555)
    mm = FakeModelManager(current_model_id=None, load_success=True)
    server = _create_mcp_server(cm, im, mm)

    monkeypatch.setattr(
        "dnet.api.load_helpers.get_model_config_json",
        lambda m: {"hidden_size": 16, "num_hidden_layers": 6},
        raising=True,
    )

    async def _prof(model_id, emb, maxb, batches):
        return {"S": object()}

    monkeypatch.setattr(cm, "profile_cluster", _prof, raising=True)

    from tests.fakes import FakeModelProfile as _MP2

    monkeypatch.setattr(
        "dnet.api.load_helpers.profile_model",
        lambda repo_id, batch_sizes, sequence_length: _MP2(),
        raising=True,
    )

    async def _solve(profiles, model_profile, model_name, num_layers, kv_bits):
        dev = DnetDeviceProperties(
            is_manager=False,
            is_busy=False,
            instance="S1",
            server_port=8001,
            shard_port=9001,
            local_ip="10.0.0.1",
        )
        return TopologyInfo(
            model=model_name,
            kv_bits=kv_bits,
            num_layers=int(num_layers),
            devices=[dev],
            assignments=[
                LayerAssignment(
                    instance="S1",
                    layers=[[0]],
                    next_instance=None,
                    window_size=1,
                    residency_size=1,
                )
            ],
            solution=None,
        )

    monkeypatch.setattr(cm, "solve_topology", _solve, raising=True)

    async with Client(transport=server) as client:
        result = await client.call_tool(
            name="load_model",
            arguments={"model": "m", "kv_bits": "8bit", "seq_len": 64},
        )
        assert result.data is not None
        assert isinstance(result.data, str)
        assert "loaded successfully" in result.data.lower()
        assert (
            im.connected is not None
            and im.connected[0] == "10.0.0.1"
            and im.connected[1] == 9001
        )


async def test_load_model_existing_topology_success(mcp_client):
    client, cm, im, mm = mcp_client
    mm.load_success = True
    dev = DnetDeviceProperties(
        is_manager=False,
        is_busy=False,
        instance="S1",
        server_port=8001,
        shard_port=9001,
        local_ip="10.0.0.1",
    )
    cm.current_topology = TopologyInfo(
        model="m",
        kv_bits="8bit",
        num_layers=2,
        devices=[dev],
        assignments=[
            LayerAssignment(
                instance="S1",
                layers=[[0, 1]],
                next_instance=None,
                window_size=1,
                residency_size=1,
            )
        ],
        solution=None,
    )
    result = await client.call_tool(
        name="load_model", arguments={"model": "m", "kv_bits": "8bit", "seq_len": 64}
    )
    assert result.data is not None
    assert "loaded successfully" in result.data.lower()


async def test_load_model_failure_raises_error(mcp_client):
    client, cm, im, mm = mcp_client
    mm.load_success = False
    dev = DnetDeviceProperties(
        is_manager=False,
        is_busy=False,
        instance="S1",
        server_port=8001,
        shard_port=9001,
        local_ip="10.0.0.1",
    )
    cm.current_topology = TopologyInfo(
        model="m",
        kv_bits="8bit",
        num_layers=2,
        devices=[dev],
        assignments=[
            LayerAssignment(
                instance="S1",
                layers=[[0, 1]],
                next_instance=None,
                window_size=1,
                residency_size=1,
            )
        ],
        solution=None,
    )
    with pytest.raises(Exception) as exc_info:
        await client.call_tool(
            "load_model", {"model": "m", "kv_bits": "8bit", "seq_len": 64}
        )
    assert exc_info.value is not None


async def test_get_status_with_model(mcp_client):
    client, cm, im, mm = mcp_client
    mm.current_model_id = "test-model"
    result = await client.call_tool(name="get_status", arguments={})
    assert result.data is not None
    data = json.loads(result.data)
    assert data["model_loaded"] == "test-model"


async def test_get_cluster_details_with_shards(mcp_client):
    client, cm, im, mm = mcp_client
    shards = {
        "S1": FakeProps("S1", "127.0.0.1", 8001, is_manager=False),
        "S2": FakeProps("S2", "127.0.0.2", 8002, is_manager=True),
    }
    cm.shards = shards
    result = await client.call_tool(name="get_cluster_details", arguments={})
    assert result.data is not None
    data = json.loads(result.data)
    assert "devices" in data
    assert "S1" in data["devices"]
    assert "S2" in data["devices"]


async def test_chat_completion_validation_error(mcp_client):
    client, cm, im, mm = mcp_client
    mm.current_model_id = "test-model"
    with pytest.raises(Exception):
        await client.call_tool(
            name="chat_completion",
            arguments={"messages": "invalid"},
        )

