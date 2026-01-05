#!/bin/bash
# Test memory monitoring tools on single shard (without compression)
# This validates MEMORY_SNAPSHOT logs, vm_stat monitoring, etc.

set -e

echo "=============================================="
echo "TESTING MEMORY MONITORING TOOLS - SINGLE SHARD"
echo "=============================================="

# Use baseline config (no compression)
export DNET_TRANSPORT_COMPRESS=false

TEST_DIR="test_monitoring_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DIR"

echo "Test directory: $TEST_DIR"
echo ""

# Start memory monitoring in background
echo "Starting memory monitoring..."
MEMORY_LOG="$TEST_DIR/memory_monitoring.log"
MONITOR_PID=""

(
    while true; do
        echo "=== $(date +%s) ===" >> "$MEMORY_LOG"
        ps aux | head -1 >> "$MEMORY_LOG"
        ps aux | grep -E "(dnet-api|dnet-shard)" | grep -v grep >> "$MEMORY_LOG" || true
        echo "--- vm_stat ---" >> "$MEMORY_LOG"
        vm_stat >> "$MEMORY_LOG" 2>/dev/null || echo "vm_stat not available" >> "$MEMORY_LOG"
        echo "" >> "$MEMORY_LOG"
        sleep 1  # More frequent sampling for testing
    done
) &
MONITOR_PID=$!

echo "Memory monitor started (PID: $MONITOR_PID)"
echo ""

# Test inference request
echo "Testing inference request..."
START=$(date +%s)
curl -s -X POST "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}' \
  > "$TEST_DIR/inference_response.json" 2>&1 || echo "Inference request failed"
END=$(date +%s)

# Test memory budget calculator using uv
echo "Testing memory budget calculator..."
if curl -s -f "http://localhost:8080/health" > /dev/null; then
    echo "API is running - testing budget calculator..."
    uv run python3 scripts/memory_budget.py --seq-len 4096 --pools 512 > "../$TEST_DIR/budget_baseline.txt" 2>&1 || echo "Budget calculator failed"
else
    echo "API not running - skipping budget calculator test"
fi
echo "Inference took $((END - START)) seconds"
echo ""

# Stop monitoring
echo "Stopping memory monitoring..."
kill $MONITOR_PID 2>/dev/null || true
wait $MONITOR_PID 2>/dev/null || true

# Extract and analyze logs
echo "Analyzing results..."
echo ""

echo "=== Memory Monitoring Stats ==="
echo "Monitoring duration: $((END - START)) seconds"
echo "Log file size: $(du -h "$MEMORY_LOG" | cut -f1)"
echo ""

echo "=== Peak Memory Usage ==="
grep "dnet-" "$MEMORY_LOG" | grep -o " [0-9]\+ " | sort -n | tail -3
echo ""

echo "=== MEMORY_SNAPSHOT Logs ==="
if ls ~/.dria/dnet/logs/dnet-shard-*.log 1> /dev/null 2>&1; then
    LATEST_LOG=$(ls -t ~/.dria/dnet/logs/dnet-shard-*.log | head -1)
    echo "Latest shard log: $LATEST_LOG"
    echo "MEMORY_SNAPSHOT entries found:"
    grep "\[MEMORY_SNAPSHOT\]" "$LATEST_LOG" | wc -l
    echo ""
    echo "Sample MEMORY_SNAPSHOT:"
    grep "\[MEMORY_SNAPSHOT\]" "$LATEST_LOG" | head -2
else
    echo "No shard logs found"
fi
echo ""

echo "=== COMM_BUDGET Logs ==="
if ls ~/.dria/dnet/logs/dnet-shard-*.log 1> /dev/null 2>&1; then
    echo "COMM_BUDGET entries found:"
    grep "\[COMM_BUDGET\]" "$LATEST_LOG" | wc -l
    echo ""
    echo "Sample COMM_BUDGET:"
    grep "\[COMM_BUDGET\]" "$LATEST_LOG" | head -1
else
    echo "No shard logs found"
fi
echo ""

echo "=============================================="
echo "MONITORING TEST COMPLETE"
echo "=============================================="
echo "Test results in: $TEST_DIR/"
echo ""
echo "Next steps:"
echo "1. Check if MEMORY_SNAPSHOT logs appear"
echo "2. Verify vm_stat monitoring captured data"
echo "3. Confirm budget calculator worked (if API was running)"
echo "4. If all good, proceed to multi-shard compression testing"
