#!/bin/bash
# Run E4: KV cache quantization test - assumes API/shard are running and model is loaded

set -e

BASE_URL="http://localhost:8080"
RESULTS_DIR="e4_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

run_test() {
    local name=$1
    local config=$2

    echo "=== Running $name test ==="

    # Setup config
    cp "$config" .env || { echo "Config file $config not found"; exit 1; }
    echo "Config loaded:"
    cat .env

    TEST_DIR="$RESULTS_DIR/$name"
    mkdir -p "$TEST_DIR"

    # Check if services are running
    if ! curl -s -f "$BASE_URL/health" > /dev/null; then
        echo "ERROR: API not running at $BASE_URL"
        return 1
    fi

    # Memory before
    echo "Capturing memory before inference..."
    ps aux | head -1 > "$TEST_DIR/memory_before.txt"
    ps aux | grep -E "(dnet-api|dnet-shard)" | grep -v grep >> "$TEST_DIR/memory_before.txt" || true
    vm_stat >> "$TEST_DIR/memory_before.txt" 2>/dev/null || echo "vm_stat not available" >> "$TEST_DIR/memory_before.txt"

    # Run inference
    echo "Running inference..."
    START=$(date +%s)
    curl -s -X POST "$BASE_URL/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{"model": "mlx-community/Llama-3.3-70B-Instruct-4bit", "messages": [{"role": "user", "content": "Explain machine learning"}], "max_tokens": 500}' \
      > "$TEST_DIR/response.json"

    END=$(date +%s)

    # Memory after
    echo "Capturing memory after inference..."
    ps aux | grep -E "(dnet-api|dnet-shard)" | grep -v grep > "$TEST_DIR/memory_after.txt" || true
    vm_stat >> "$TEST_DIR/memory_after.txt" 2>/dev/null || echo "vm_stat not available" >> "$TEST_DIR/memory_after.txt"

    # Extract logs from running processes (this assumes logs are being written)
    # Note: This won't work well since logs go to the running processes
    echo "Note: Logs are written to running API/shard processes"

    # Summary
    cat > "$TEST_DIR/summary.txt" << EOF
Test: $name
Config: $config
Duration: $((END - START))s
Timestamp: $(date)
EOF

    echo "✓ $name test completed - results in $TEST_DIR"
}

# Run E4 tests: KV cache quantization impact
run_test "kv_fp16" "kv_fp16.config"
run_test "kv_quantized" "kv_quantized.config"

echo ""
echo "Tests completed. Results in $RESULTS_DIR"
echo ""
echo "To analyze:"
echo "cat $RESULTS_DIR/*/memory_*.txt"
echo "Check API/shard logs for:"
echo "  [PROFILE] entries (weight loading)"
echo "  [STAGE_MEMORY] entries (stage-wise memory breakdown)"
echo "  [COMM_BUDGET] entries (inter-stage communication)"
echo "  [STAGE_PEAK_MEMORY] entries (peak memory per stage)"
echo ""
echo "Compare KV cache memory impact:"
echo "  Look at kv_mb in [STAGE_MEMORY] for fp16 vs 8bit quantization"
echo "  kv_fp16 should use ~2x more memory than kv_quantized for same sequence length"
echo "  Check if KV cache is a significant contributor to peak stage memory"
