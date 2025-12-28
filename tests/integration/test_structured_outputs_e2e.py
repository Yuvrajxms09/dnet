import json
import time
import requests
import pytest

API_HTTP_PORT = 8080
BASE_URL = f"http://localhost:{API_HTTP_PORT}"


def wait_for_health(url: str, timeout: float = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=2)
            response.raise_for_status()
            return True
        except (requests.RequestException, requests.ConnectionError):
            time.sleep(0.5)
    return False


def test_structured_outputs_end_to_end():
    if not wait_for_health(f"{BASE_URL}/health"):
        pytest.skip("Server not responding")

    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}, "count": {"type": "integer"}},
        "required": ["answer"]
    }
    payload = {
        "model": "Qwen/Qwen3-4B-MLX-4bit",
        "messages": [{"role": "user", "content": "Give me a simple response"}],
        "structured_outputs": {"json": schema},
        "max_tokens": 30
    }

    response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=60)
    assert response.status_code == 200

    result = response.json()
    content = result["choices"][0]["message"]["content"]

    parsed = json.loads(content)
    assert isinstance(parsed, dict)
    assert "answer" in parsed
    if "count" in parsed:
        assert isinstance(parsed["count"], int)
