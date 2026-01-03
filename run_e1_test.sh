#!/bin/bash
# Run E1: Inter-stage dtype test - ONE CONFIG PER RUN
#
# Usage: SHARD1_IP=x.x.x.x SHARD2_IP=y.y.y.y ./run_e1_test.sh [baseline|compressed]
#
# IMPORTANT: Restart shards between configs. Config is read at startup.

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
    q8)
        CONFIG_FILE="q8.config"
        ;;
    *)
        echo "Usage: $0 [baseline|compressed|q8]"
        echo ""
        echo "  baseline   - Test with fp16 wire dtype (default)"
        echo "  compressed - Test with sparse fp16 compression"
        echo "  q8         - Test with Q8 quantization + compression (like DLlama Q80)"
        echo ""
        echo "IMPORTANT: Restart shards with the matching .env config before running each test!"
        exit 1
        ;;
esac

BASE_URL="http://localhost:8080"
RESULTS_DIR="e1_${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "=== E1 Test: $CONFIG_NAME (Results: $RESULTS_DIR) ==="

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

# Get model name from environment or use default
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-32B-MLX-8bit}"
echo "Using Model: $MODEL_NAME"
echo ""

# Generate deterministic manual topology
if [ -z "$SHARD1_IP" ] || [ -z "$SHARD2_IP" ]; then
    echo "ERROR: SHARD1_IP and SHARD2_IP required"
    exit 1
fi

TOPOLOGY_JSON=$(uv run python3 scripts/generate_manual_topology.py "$MODEL_NAME" "$SHARD1_IP" "$SHARD2_IP" 2>/dev/null)
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to generate topology"
    exit 1
fi

# Check if API is reachable
if ! curl -s -f "$BASE_URL/health" > /dev/null; then
    echo "ERROR: API not running at $BASE_URL"
    exit 1
fi

# Prepare manual topology (deterministic layer assignment)
echo "=== Preparing Manual Topology ==="
echo "$TOPOLOGY_JSON" | curl -s -X POST "$BASE_URL/v1/prepare_topology_manual" \
  -H "Content-Type: application/json" \
  -d @- \
  > "$RESULTS_DIR/topology_response.json"

if [ $? -ne 0 ] || ! grep -q "assignments" "$RESULTS_DIR/topology_response.json"; then
    echo "ERROR: Failed to prepare manual topology"
    cat "$RESULTS_DIR/topology_response.json"
    exit 1
fi

echo "Manual topology prepared successfully"
echo ""

# Load model with prepared topology
echo "=== Loading Model with Manual Topology ==="
echo "$TOPOLOGY_JSON" | curl -s -X POST "$BASE_URL/v1/load_model" \
  -H "Content-Type: application/json" \
  -d @- \
  > "$RESULTS_DIR/load_model_response.json"

if [ $? -ne 0 ] || ! grep -q "success.*true" "$RESULTS_DIR/load_model_response.json"; then
    echo "ERROR: Failed to load model"
    cat "$RESULTS_DIR/load_model_response.json"
    exit 1
fi

echo "Model loaded successfully with manual topology"
echo ""

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
echo "✓ E1 Test ($CONFIG_NAME) completed: $RESULTS_DIR"
echo "Key files: budget.txt, memory_snapshots.txt, analysis.txt"
