#!/bin/bash
# Local validation test for Issue #73 - Single device setup
# Tests monitoring infrastructure and basic functionality

set -e

BASE_URL="http://localhost:8080"
RESULTS_DIR="local_validation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "=== LOCAL VALIDATION TEST ==="
echo "Testing monitoring infrastructure with single device"
echo "Results: $RESULTS_DIR"

# Check single shard setup
if ! curl -s -f "$BASE_URL/health" > /dev/null; then
    echo "ERROR: API not running at $BASE_URL"
    exit 1
fi

echo "✓ API health check passed"

# Get model info
MODEL_NAME=$(curl -s "$BASE_URL/v1/topology" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('model', 'unknown'))
except:
    print('unknown')
")

if [ "$MODEL_NAME" = "unknown" ]; then
    echo "ERROR: Could not detect model. Load via dnet-tui first."
    exit 1
fi

echo "✓ Detected model: $MODEL_NAME"

# Start memory monitoring (same as real tests)
MEMORY_LOG="$RESULTS_DIR/memory_monitoring.log"
(
    while true; do
        echo "=== $(date +%s) ===" >> "$MEMORY_LOG"
        ps aux | head -1 >> "$MEMORY_LOG"
        ps aux | grep -E "(dnet-api|dnet-shard)" | grep -v grep >> "$MEMORY_LOG" || true
        vm_stat >> "$MEMORY_LOG" 2>/dev/null || echo "vm_stat not available" >> "$MEMORY_LOG"
        echo "" >> "$MEMORY_LOG"
        sleep 1
    done
) &
MONITOR_PID=$!

# Run inference test
echo "Running inference test..."
START=$(date +%s)
curl -s -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello, test message\"}], \"max_tokens\": 10}" \
  > "$RESULTS_DIR/response.json"

END=$(date +%s)
kill $MONITOR_PID 2>/dev/null || true

# Analyze results
echo "Analyzing results..."
echo "Test duration: $((END - START))s" > "$RESULTS_DIR/summary.txt"
echo "Model: $MODEL_NAME" >> "$RESULTS_DIR/summary.txt"
echo "Response: $(cat $RESULTS_DIR/response.json | jq -r '.choices[0].message.content' 2>/dev/null || echo 'parse_error')" >> "$RESULTS_DIR/summary.txt"

# Check monitoring worked
if [ -s "$MEMORY_LOG" ]; then
    echo "✓ Memory monitoring captured data"
    echo "Memory samples: $(grep -c "===" "$MEMORY_LOG")" >> "$RESULTS_DIR/summary.txt"
else
    echo "❌ Memory monitoring failed"
fi

# Extract basic memory stats
if command -v vm_stat >/dev/null 2>&1; then
    echo "✓ vm_stat available"
else
    echo "⚠️ vm_stat not available (expected on some systems)"
fi

echo ""
echo "=== VALIDATION COMPLETE ==="
echo "Check $RESULTS_DIR for results"
echo "If monitoring works here, it should work on remote machines too"
echo ""
echo "Next: Deploy to Scaleway with multi-device setup for actual E1-E3 tests"
