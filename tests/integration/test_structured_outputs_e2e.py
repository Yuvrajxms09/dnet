"""End-to-end tests for structured outputs functionality.

These tests validate that structured outputs work against a running server.
Tests assume server is already running and accessible.

Usage:
    # Test against localhost
    uv run pytest tests/integration/test_structured_outputs_e2e.py -v

    # Test against remote server
    BASE_URL=http://62.210.150.5:8080 uv run pytest tests/integration/test_structured_outputs_e2e.py -v
"""

import json
import os
import time
from typing import Any

import pytest
import requests

# Server configuration - can be overridden via environment
API_HTTP_PORT = int(os.getenv("API_HTTP_PORT", "8080"))
BASE_URL = os.getenv("BASE_URL", f"http://62.210.150.5:{API_HTTP_PORT}")

# Timeouts
HEALTH_CHECK_TIMEOUT = 30  # seconds to wait for server health
INFERENCE_TIMEOUT = 60  # seconds for inference requests


def wait_for_health(url: str, timeout: float = HEALTH_CHECK_TIMEOUT) -> bool:
    """Wait for server health endpoint to respond."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                return True
        except (requests.RequestException, requests.ConnectionError):
            time.sleep(0.5)
    return False


@pytest.mark.parametrize(
    "schema,prompt",
    [
        (
            {
                "type": "object",
                "properties": {"answer": {"type": "string"}, "count": {"type": "integer"}},
                "required": ["answer"]
            },
            "Give me a simple response with a count"
        ),
        (
            {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name"]
            },
            "Create a profile for a person"
        ),
        (
            {
                "type": "object",
                "properties": {"items": {"type": "array", "items": {"type": "string"}}},
                "required": ["items"]
            },
            "List three fruits"
        )
    ]
)
def test_structured_outputs_end_to_end(schema: dict[str, Any], prompt: str) -> None:
    """Test that structured outputs produce valid JSON conforming to schema."""
    if not wait_for_health(BASE_URL):
        pytest.skip(f"Server not responding at {BASE_URL}/health")

    payload = {
        "model": "Qwen/Qwen3-4B-MLX-4bit",
        "messages": [{"role": "user", "content": prompt}],
        "structured_outputs": {"json": schema},
        "max_tokens": 500,
        "temperature": 0.1
    }

    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json=payload,
        timeout=INFERENCE_TIMEOUT
    )
    assert response.status_code == 200, f"Request failed: {response.text}"

    result = response.json()
    assert "choices" in result, f"Response missing 'choices': {result}"
    assert len(result["choices"]) > 0, f"No choices in response: {result}"

    choice = result["choices"][0]
    assert "message" in choice, f"Choice missing 'message': {choice}"
    content = choice["message"].get("content", "")

    # Verify response is valid JSON
    parsed = json.loads(content)
    assert isinstance(parsed, dict), f"Response is not a JSON object: {content}"

    # Verify it matches the schema requirements
    for required_field in schema.get("required", []):
        assert required_field in parsed, f"Required field '{required_field}' missing from response"

    # Basic type checking for properties
    properties = schema.get("properties", {})
    for field_name, field_schema in properties.items():
        if field_name in parsed:
            field_type = field_schema.get("type")
            if field_type == "string":
                assert isinstance(parsed[field_name], str), f"Field '{field_name}' should be string"
            elif field_type == "integer":
                assert isinstance(parsed[field_name], int), f"Field '{field_name}' should be integer"
            elif field_type == "array":
                assert isinstance(parsed[field_name], list), f"Field '{field_name}' should be array"
