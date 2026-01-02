#!/bin/bash
# Run E3: Embedding/LM head memory behavior test - assumes API/shard are running and model is loaded

set -e

BASE_URL="http://localhost:8080"
RESULTS_DIR="e3_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Run budget calculator for baseline
echo "=== Running Memory Budget Calculator ==="
uv run python3 scripts/memory_budget.py \
  --seq-len 2048 \
  --pools 512 \
  > "$RESULTS_DIR/budget_baseline.txt"

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
      -d '{"model": "Qwen/Qwen3-32B-MLX-bf16", "messages": [{"role": "user", "content": "hi there"}], "max_tokens": 50}' \
      > "$TEST_DIR/response.json"

    END=$(date +%s)

    # Memory after
    echo "Capturing memory after inference..."
    ps aux | grep -E "(dnet-api|dnet-shard)" | grep -v grep > "$TEST_DIR/memory_after.txt" || true
    vm_stat >> "$TEST_DIR/memory_after.txt" 2>/dev/null || echo "vm_stat not available" >> "$TEST_DIR/memory_after.txt"

    # Extract memory snapshots from logs
    echo "Extracting memory snapshots from logs..."
    for log_file in ~/.dria/dnet/dnet-shard-*.log; do
        if [ -f "$log_file" ]; then
            grep "\[MEMORY_SNAPSHOT\]" "$log_file" > "$TEST_DIR/memory_snapshots.txt" 2>/dev/null || true
        fi
    done

    # Analyze snapshots vs budget
    if [ -f "$TEST_DIR/memory_snapshots.txt" ] && [ -f "$RESULTS_DIR/budget_baseline.txt" ]; then
        echo "Analyzing memory snapshots vs budget..." > "$TEST_DIR/analysis.txt"
        echo "Expected budget:" >> "$TEST_DIR/analysis.txt"
        tail -n 10 "$RESULTS_DIR/budget_baseline.txt" >> "$TEST_DIR/analysis.txt" 2>/dev/null || true
        echo "" >> "$TEST_DIR/analysis.txt"
        echo "Actual snapshots:" >> "$TEST_DIR/analysis.txt"
        cat "$TEST_DIR/memory_snapshots.txt" >> "$TEST_DIR/analysis.txt"
    fi

    # Summary
    cat > "$TEST_DIR/summary.txt" << EOF
Test: $name
Config: $config
Duration: $((END - START))s
Timestamp: $(date)
EOF

    echo "✓ $name test completed - results in $TEST_DIR"
}

# Run E3 tests: Embedding/LM head placement and quantization
run_test "embedding_default" "embedding_default.config"
run_test "embedding_stressed" "embedding_stressed.config"
run_test "embedding_extreme" "embedding_extreme.config"

echo ""
echo "Tests completed. Results in $RESULTS_DIR"
echo ""
echo "To analyze:"
echo "1. Budget calculator: $RESULTS_DIR/budget_baseline.txt"
echo "2. Memory snapshots: $RESULTS_DIR/*/memory_snapshots.txt"
echo "3. Analysis: $RESULTS_DIR/*/analysis.txt"
echo "4. Check API/shard logs for:"
echo "   [MEMORY_SNAPSHOT] entries (actual memory usage)"
echo "   [PROFILE] entries (weight loading)"
echo "   [STAGE_MEMORY] entries (stage-wise memory breakdown)"
echo "   [COMM_BUDGET] entries (inter-stage communication)"
echo ""
echo "Compare embedding memory behavior under constraints:"
echo "  Check budget for embedding size on start stage (H2 hypothesis)"
echo "  Compare actual snapshots to see if embeddings cause memory spikes"
echo "  Look for gap between expected and actual memory usage"
