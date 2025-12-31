#!/bin/bash
# Run E1 compression test - following dnet integration test pattern

set -e

# Server configuration (same as integration tests)
API_HTTP_PORT=8080
API_GRPC_PORT=58080
SHARD_HTTP_PORT=8081
SHARD_GRPC_PORT=58081
BASE_URL="http://localhost:$API_HTTP_PORT"

# Timeouts
HEALTH_CHECK_TIMEOUT=60
MODEL_LOAD_TIMEOUT=300
INFERENCE_TIMEOUT=120

RESULTS_DIR="e1_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

wait_for_health() {
    local url=$1
    local timeout=${2:-$HEALTH_CHECK_TIMEOUT}
    local deadline=$(($(date +%s) + timeout))

    echo "Waiting for health check at $url/health (timeout: ${timeout}s)"

    while [ $(date +%s) -lt $deadline ]; do
        if curl -s -f "$url/health" > /dev/null 2>&1; then
            echo "✓ Server healthy at $url"
            return 0
        fi
        sleep 1
    done

    echo "✗ Server not healthy at $url after ${timeout}s"
    return 1
}

run_test() {
    local name=$1
    local config=$2

    echo "=== Running $name test with $config ==="

    # Setup config
    cp "$config" .env || { echo "Config file $config not found"; exit 1; }

    TEST_DIR="$RESULTS_DIR/$name"
    mkdir -p "$TEST_DIR"

    # Start shard first (following integration test pattern)
    echo "Starting shard..."
    uv run dnet-shard --http-port $SHARD_HTTP_PORT --grpc-port $SHARD_GRPC_PORT \
        > "$TEST_DIR/shard.log" 2>&1 &
    SHARD_PID=$!

    # Wait for shard health
    if ! wait_for_health "http://localhost:$SHARD_HTTP_PORT" 30; then
        echo "Shard failed to start, cleaning up..."
        kill $SHARD_PID 2>/dev/null || true
        sleep 2
        kill -9 $SHARD_PID 2>/dev/null || true
        echo "$name test failed - shard not healthy"
        return 1
    fi

    # Start API
    echo "Starting API..."
    uv run dnet-api --http-port $API_HTTP_PORT --grpc-port $API_GRPC_PORT \
        > "$TEST_DIR/api.log" 2>&1 &
    API_PID=$!

    # Wait for API health
    if ! wait_for_health "$BASE_URL"; then
        echo "API failed to start, cleaning up..."
        kill $API_PID $SHARD_PID 2>/dev/null || true
        sleep 2
        kill -9 $API_PID $SHARD_PID 2>/dev/null || true
        echo "$name test failed - API not healthy"
        return 1
    fi

    # Memory before
    echo "Capturing memory before inference..."
    ps aux | head -1 > "$TEST_DIR/memory_before.txt"
    ps aux | grep -E "(dnet-api|dnet-shard)" | grep -v grep >> "$TEST_DIR/memory_before.txt" || true
    echo "=== VM Stats ===" >> "$TEST_DIR/memory_before.txt"
    vm_stat >> "$TEST_DIR/memory_before.txt" 2>/dev/null || echo "vm_stat not available" >> "$TEST_DIR/memory_before.txt"

    # Prepare and load model (following integration test pattern)
    echo "Preparing topology..."
    if ! curl -s -X POST "$BASE_URL/v1/prepare_topology" \
        -H "Content-Type: application/json" \
        -d '{"model": "mlx-community/Llama-3.2-3B-Instruct-4bit"}' \
        > "$TEST_DIR/prepare_response.json"; then
        echo "Topology preparation failed"
        return 1
    fi

    echo "Loading model..."
    if ! curl -s -X POST "$BASE_URL/v1/load_model" \
        -H "Content-Type: application/json" \
        -d '{"model": "mlx-community/Llama-3.2-3B-Instruct-4bit"}' \
        > "$TEST_DIR/load_response.json"; then
        echo "Model loading failed"
        return 1
    fi

    # Run inference
    echo "Running inference test..."
    START=$(date +%s)
    curl -s -X POST "$BASE_URL/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{"model": "mlx-community/Llama-3.2-3B-Instruct-4bit", "messages": [{"role": "user", "content": "Explain the concept of machine learning in detail."}], "max_tokens": 500}' \
      > "$TEST_DIR/response.json"

    END=$(date +%s)

    # Memory after
    echo "Capturing memory after inference..."
    ps aux | grep -E "(dnet-api|dnet-shard)" | grep -v grep > "$TEST_DIR/memory_after.txt" || true
    echo "=== VM Stats ===" >> "$TEST_DIR/memory_after.txt"
    vm_stat >> "$TEST_DIR/memory_after.txt" 2>/dev/null || echo "vm_stat not available" >> "$TEST_DIR/memory_after.txt"

    # Extract logs
    grep "\[PROFILE\]" "$TEST_DIR"/*.log > "$TEST_DIR/profile.txt" 2>/dev/null || true

    # Summary
    cat > "$TEST_DIR/summary.txt" << EOF
Test: $name
Config: $config
Duration: $((END - START))s
API_PID: $API_PID
Shard_PID: $SHARD_PID
EOF

    # Cleanup (following integration test pattern)
    echo "Cleaning up processes..."
    kill $API_PID 2>/dev/null || true
    kill $SHARD_PID 2>/dev/null || true

    # Wait for graceful shutdown
    sleep 5

    # Force kill if still running
    kill -9 $API_PID $SHARD_PID 2>/dev/null || true

    echo "✓ $name test completed"
}

# Run tests
run_test "baseline" "baseline.config"
run_test "compressed" "compressed.config"

# Generate summary
echo "Tests completed. Results in $RESULTS_DIR"
echo ""
echo "To analyze:"
echo "grep MATERIALIZE $RESULTS_DIR/*/profile.txt"
echo "cat $RESULTS_DIR/*/memory_*.txt"
