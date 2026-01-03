#!/bin/bash
# Run E1: Inter-stage dtype test - ONE CONFIG PER RUN
# 
# Usage:
#   ./run_e1_test.sh baseline     # Test with fp16 wire (current default)
#   ./run_e1_test.sh compressed   # Test with qsparse8_v1 compression
#
# IMPORTANT: You must restart shards between different configs!
# The config is read at shard startup, not during runtime.
#
# Workflow:
#   1. Copy baseline.config to .env on BOTH shard machines
#   2. Start API and shards, load model via dnet-tui
#   3. ./run_e1_test.sh baseline
#   4. Stop shards and API
#   5. Copy compressed.config to .env on BOTH shard machines
#   6. Start API and shards, load model via dnet-tui
#   7. ./run_e1_test.sh compressed
#   8. Compare e1_baseline_*/analysis.txt vs e1_compressed_*/analysis.txt

set -e

# Parse argument
CONFIG_NAME="${1:-baseline}"

case "$CONFIG_NAME" in
    baseline)
        CONFIG_FILE="baseline.config"
        ;;
    compressed)
        CONFIG_FILE="compressed.config"
        ;;
    *)
        echo "Usage: $0 [baseline|compressed]"
        echo ""
        echo "  baseline   - Test with fp16 wire dtype (default)"
        echo "  compressed - Test with qsparse8_v1 compression"
        echo ""
        echo "IMPORTANT: Restart shards with the matching .env config before running each test!"
        exit 1
        ;;
esac

BASE_URL="http://localhost:8080"
RESULTS_DIR="e1_${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo "E1 Test: $CONFIG_NAME"
echo "Expected config: $CONFIG_FILE"
echo "Results will be in: $RESULTS_DIR"
echo "=============================================="
echo ""

# Verify the config file exists (for documentation purposes)
if [ ! -f "$CONFIG_FILE" ]; then
    echo "WARNING: $CONFIG_FILE not found in current directory"
    echo "Make sure shards were started with the correct .env settings"
fi

# Show expected config for this test
echo "=== Expected Config for $CONFIG_NAME ==="
if [ -f "$CONFIG_FILE" ]; then
    cat "$CONFIG_FILE"
else
    echo "(Config file not found - proceeding anyway)"
fi
echo ""

# Auto-detect model from topology
echo "=== Detecting Model from Running API ==="
MODEL_NAME=$(curl -s "http://localhost:8080/v1/topology" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('model', 'unknown'))
except:
    print('unknown')
")

if [ "$MODEL_NAME" = "unknown" ] || [ -z "$MODEL_NAME" ]; then
    echo "ERROR: Could not detect model from topology."
    echo "Is the model loaded via dnet-tui?"
    echo "Is the API running at $BASE_URL?"
    exit 1
fi

echo "Detected Model: $MODEL_NAME"
echo ""

# Run budget calculator (auto-detects model and topology)
echo "=== Running Memory Budget Calculator ==="
uv run python3 scripts/memory_budget.py \
  --seq-len 2048 \
  --pools 512 \
  | tee "$RESULTS_DIR/budget.txt"
echo ""

# Check if API is reachable
if ! curl -s -f "$BASE_URL/health" > /dev/null; then
    echo "ERROR: API not running at $BASE_URL"
    exit 1
fi

# Memory before inference
echo "=== Capturing Memory Before Inference ==="
ps aux | head -1 > "$RESULTS_DIR/memory_before.txt"
ps aux | grep -E "(dnet-api|dnet-shard|python.*dnet)" | grep -v grep >> "$RESULTS_DIR/memory_before.txt" || true
vm_stat >> "$RESULTS_DIR/memory_before.txt" 2>/dev/null || echo "vm_stat not available" >> "$RESULTS_DIR/memory_before.txt"
echo "Saved to $RESULTS_DIR/memory_before.txt"

# Run inference
echo "=== Running Inference ==="
START=$(date +%s)
curl -s -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"Explain the concept of memory management in operating systems in detail.\"}], \"max_tokens\": 100}" \
  | tee "$RESULTS_DIR/response.json"
echo ""
END=$(date +%s)
DURATION=$((END - START))
echo "Inference completed in ${DURATION}s"
echo ""

# Memory after inference
echo "=== Capturing Memory After Inference ==="
ps aux | grep -E "(dnet-api|dnet-shard|python.*dnet)" | grep -v grep > "$RESULTS_DIR/memory_after.txt" || true
vm_stat >> "$RESULTS_DIR/memory_after.txt" 2>/dev/null || echo "vm_stat not available" >> "$RESULTS_DIR/memory_after.txt"
echo "Saved to $RESULTS_DIR/memory_after.txt"

# Extract memory snapshots from logs
echo "=== Extracting Memory Snapshots from Logs ==="
> "$RESULTS_DIR/memory_snapshots.txt"  # Clear/create file

for log_file in ~/.dria/dnet/logs/dnet-shard-*.log; do
    if [ -f "$log_file" ]; then
        echo "--- From $(basename $log_file) ---" >> "$RESULTS_DIR/memory_snapshots.txt"
        grep "\[MEMORY_SNAPSHOT\]" "$log_file" >> "$RESULTS_DIR/memory_snapshots.txt" 2>/dev/null || echo "(no snapshots)" >> "$RESULTS_DIR/memory_snapshots.txt"
        echo "" >> "$RESULTS_DIR/memory_snapshots.txt"
    fi
done

# Extract stage memory breakdown
> "$RESULTS_DIR/stage_memory.txt"
for log_file in ~/.dria/dnet/logs/dnet-shard-*.log; do
    if [ -f "$log_file" ]; then
        echo "--- From $(basename $log_file) ---" >> "$RESULTS_DIR/stage_memory.txt"
        grep "\[STAGE_MEMORY\]" "$log_file" >> "$RESULTS_DIR/stage_memory.txt" 2>/dev/null || echo "(no stage memory logs)" >> "$RESULTS_DIR/stage_memory.txt"
        echo "" >> "$RESULTS_DIR/stage_memory.txt"
    fi
done

# Extract communication budget
> "$RESULTS_DIR/comm_budget.txt"
for log_file in ~/.dria/dnet/logs/dnet-shard-*.log; do
    if [ -f "$log_file" ]; then
        echo "--- From $(basename $log_file) ---" >> "$RESULTS_DIR/comm_budget.txt"
        grep "\[COMM_BUDGET\]" "$log_file" >> "$RESULTS_DIR/comm_budget.txt" 2>/dev/null || echo "(no comm budget logs)" >> "$RESULTS_DIR/comm_budget.txt"
        echo "" >> "$RESULTS_DIR/comm_budget.txt"
    fi
done

# Extract active config (to verify settings are applied)
echo "=== Verifying Active Config ==="
> "$RESULTS_DIR/active_config.txt"
for log_file in ~/.dria/dnet/logs/dnet-shard-*.log; do
    if [ -f "$log_file" ]; then
        echo "--- From $(basename $log_file) ---" >> "$RESULTS_DIR/active_config.txt"
        grep "\[CONFIG\]" "$log_file" | tail -1 >> "$RESULTS_DIR/active_config.txt" 2>/dev/null || echo "(no config log)" >> "$RESULTS_DIR/active_config.txt"
        echo "" >> "$RESULTS_DIR/active_config.txt"
    fi
done
echo "Active config from shards:"
cat "$RESULTS_DIR/active_config.txt"
echo ""

# Verify config matches expected
echo "=== Config Verification ==="
if [ "$CONFIG_NAME" = "baseline" ]; then
    EXPECTED_COMPRESS="False"
    EXPECTED_WIRE="fp16"
else
    EXPECTED_COMPRESS="True"
    EXPECTED_WIRE="qsparse8_v1"
fi

if grep -q "compress=$EXPECTED_COMPRESS" "$RESULTS_DIR/active_config.txt" 2>/dev/null; then
    echo "✓ Compression setting matches expected ($EXPECTED_COMPRESS)"
else
    echo "⚠ WARNING: Compression setting may not match expected ($EXPECTED_COMPRESS)"
    echo "  Check active_config.txt for actual values"
fi
echo ""

echo "Extracted log entries to:"
echo "  - $RESULTS_DIR/memory_snapshots.txt"
echo "  - $RESULTS_DIR/stage_memory.txt"
echo "  - $RESULTS_DIR/comm_budget.txt"
echo ""

# Create analysis summary
echo "=== Creating Analysis Summary ==="
cat > "$RESULTS_DIR/analysis.txt" << EOF
============================================
E1 TEST ANALYSIS: $CONFIG_NAME
============================================
Test: $CONFIG_NAME
Config: $CONFIG_FILE
Model: $MODEL_NAME
Duration: ${DURATION}s
Timestamp: $(date)
============================================

THEORETICAL BUDGET (from memory_budget.py):
-------------------------------------------
EOF
cat "$RESULTS_DIR/budget.txt" >> "$RESULTS_DIR/analysis.txt" 2>/dev/null || echo "(budget not available)" >> "$RESULTS_DIR/analysis.txt"

cat >> "$RESULTS_DIR/analysis.txt" << EOF

ACTUAL MEMORY SNAPSHOTS (from shard logs):
------------------------------------------
EOF
cat "$RESULTS_DIR/memory_snapshots.txt" >> "$RESULTS_DIR/analysis.txt"

cat >> "$RESULTS_DIR/analysis.txt" << EOF

STAGE MEMORY BREAKDOWN:
-----------------------
EOF
cat "$RESULTS_DIR/stage_memory.txt" >> "$RESULTS_DIR/analysis.txt"

cat >> "$RESULTS_DIR/analysis.txt" << EOF

COMMUNICATION BUDGET (bytes per activation):
--------------------------------------------
EOF
cat "$RESULTS_DIR/comm_budget.txt" >> "$RESULTS_DIR/analysis.txt"

echo "Analysis saved to $RESULTS_DIR/analysis.txt"
echo ""

# Final summary
echo "=============================================="
echo "✓ E1 Test ($CONFIG_NAME) Completed"
echo "=============================================="
echo ""
echo "Results in: $RESULTS_DIR/"
echo ""
echo "Key files to examine:"
echo "  1. budget.txt          - Theoretical memory per stage"
echo "  2. memory_snapshots.txt - Actual MLX memory usage"
echo "  3. stage_memory.txt    - Component breakdown (weights/pools/activations)"
echo "  4. analysis.txt        - Combined summary"
echo ""
echo "To compare theoretical vs actual, look for the GAP:"
echo "  - If Stage X budget says 20GB but snapshot shows 28GB"
echo "  - That 8GB gap points to hidden overhead (H1/H4)"
echo ""
if [ "$CONFIG_NAME" = "baseline" ]; then
    echo "NEXT STEP: To test compressed config:"
    echo "  1. Stop shards and API"
    echo "  2. Copy compressed.config to .env on both shard machines"
    echo "  3. Restart shards and API, load model"
    echo "  4. Run: ./run_e1_test.sh compressed"
    echo "  5. Compare e1_baseline_*/analysis.txt vs e1_compressed_*/analysis.txt"
fi
