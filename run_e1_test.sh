#!/bin/bash
# Run E1 compression test

set -e

RESULTS_DIR="e1_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

run_test() {
    local name=$1
    local config=$2

    echo "Running $name test with $config"

    # Setup config
    cp "$config" .env || { echo "Config file $config not found"; exit 1; }

    TEST_DIR="$RESULTS_DIR/$name"
    mkdir -p "$TEST_DIR"

    # Start services
    uv run dnet-api --http-port 8080 --grpc-port 58080 > "$TEST_DIR/api.log" 2>&1 &
    API_PID=$!
    uv run dnet-shard --http-port 8081 --grpc-port 58081 > "$TEST_DIR/shard.log" 2>&1 &
    SHARD_PID=$!

    sleep 10

    # Memory before
    ps aux | head -1 > "$TEST_DIR/memory_before.txt"
    ps aux | grep -E "(dnet-api|dnet-shard)" | grep -v grep >> "$TEST_DIR/memory_before.txt" || true
    # macOS memory info
    echo "=== VM Stats ===" >> "$TEST_DIR/memory_before.txt"
    vm_stat >> "$TEST_DIR/memory_before.txt" 2>/dev/null || echo "vm_stat not available" >> "$TEST_DIR/memory_before.txt"

    # Run inference
    START=$(date +%s)
    curl -s -X POST http://localhost:8080/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "mlx-community/Llama-3.2-3B-Instruct-4bit", "messages": [{"role": "user", "content": "Explain the concept of machine learning in detail."}], "max_tokens": 500}' \
      > "$TEST_DIR/response.json"

    END=$(date +%s)

    # Memory after
    ps aux | grep -E "(dnet-api|dnet-shard)" | grep -v grep > "$TEST_DIR/memory_after.txt" || true
    # macOS memory info
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

    # Cleanup
    kill $API_PID $SHARD_PID 2>/dev/null || true
    sleep 3
    kill -9 $API_PID $SHARD_PID 2>/dev/null || true

    echo "$name test completed"
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
