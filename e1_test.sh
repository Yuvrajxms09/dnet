#!/bin/bash
# Run E1: Inter-stage dtype compression test - assumes API/shard are running and model is loaded

set -e

BASE_URL="http://localhost:8080"

# Check if argument provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 [baseline|compressed]"
    echo ""
    echo "Run workflow:"
    echo "1. Start services with baseline config: cp baseline.config .env && ./dnet-api & ./dnet-shard shard-1 & ./dnet-shard shard-2 &"
    echo "2. Load model via dnet-tui"
    echo "3. Run: ./run_e1_test.sh baseline"
    echo "4. Stop services: pkill -f 'dnet-api' && pkill -f 'dnet-shard'"
    echo "5. Start services with compressed config: cp compressed.config .env && ./dnet-api & ./dnet-shard shard-1 & ./dnet-shard shard-2 &"
    echo "6. Load model via dnet-tui"
    echo "7. Run: ./run_e1_test.sh compressed"
    echo "8. Compare: ./compare_e1_results.sh e1_results_*/baseline_fp16 e1_results_*/compressed_qsparse8"
    exit 1
fi

TEST_TYPE=$1
RESULTS_DIR="e1_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

case $TEST_TYPE in
    "baseline")
        TEST_NAME="baseline_fp16"
        EXPECTED_COMPRESS="false"
        ;;
    "compressed")
        TEST_NAME="compressed_qsparse8"
        EXPECTED_COMPRESS="true"
        ;;
    *)
        echo "ERROR: Invalid test type. Use 'baseline' or 'compressed'"
        exit 1
        ;;
esac

TEST_DIR="$RESULTS_DIR/$TEST_NAME"
mkdir -p "$TEST_DIR"

echo "=== Running E1 $TEST_TYPE Test ==="
echo "Results will be saved to: $TEST_DIR"

# Check if services are running
if ! curl -s -f "$BASE_URL/health" > /dev/null; then
    echo "ERROR: API not running at $BASE_URL"
    echo "Make sure dnet-api and dnet-shard services are running"
    exit 1
fi

# Auto-detect model from topology
MODEL_NAME=$(curl -s "http://localhost:8080/v1/topology" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('model', 'unknown'))
except:
    print('unknown')
")

if [ "$MODEL_NAME" = "unknown" ] || [ -z "$MODEL_NAME" ]; then
    echo "ERROR: Could not detect model from topology. Is the model loaded via dnet-tui?"
    exit 1
fi

echo "✓ Detected Model: $MODEL_NAME"

# Verify config matches test type
echo "Verifying config matches test type..."
CURRENT_COMPRESS=$(grep "^DNET_TRANSPORT_COMPRESS=" .env | cut -d'=' -f2 || echo "not_set")
if [ "$CURRENT_COMPRESS" != "$EXPECTED_COMPRESS" ]; then
    echo "ERROR: Config mismatch!"
    echo "  Expected DNET_TRANSPORT_COMPRESS=$EXPECTED_COMPRESS for $TEST_TYPE test"
    echo "  Current: DNET_TRANSPORT_COMPRESS=$CURRENT_COMPRESS"
    echo "  Make sure you set the correct .env config before starting services"
    exit 1
fi

echo "✓ Config verified: DNET_TRANSPORT_COMPRESS=$CURRENT_COMPRESS"

# Run budget calculator first (auto-detects model and topology from loaded system)
echo "Running memory budget calculator..."
uv run python3 scripts/memory_budget.py \
  --seq-len 4096 \
  --pools 512 \
  > "$RESULTS_DIR/budget_baseline.txt"

# Start external memory monitoring
echo "Starting external memory monitoring..."
MEMORY_LOG="$TEST_DIR/memory_monitoring.log"
MONITOR_PID=""

# Start background memory monitoring (samples every 0.5 seconds during inference)
(
    while true; do
        echo "=== $(date +%s) ===" >> "$MEMORY_LOG"
        ps aux | head -1 >> "$MEMORY_LOG"
        ps aux | grep -E "(dnet-api|dnet-shard)" | grep -v grep >> "$MEMORY_LOG" || true
        echo "--- vm_stat ---" >> "$MEMORY_LOG"
        vm_stat >> "$MEMORY_LOG" 2>/dev/null || echo "vm_stat not available" >> "$MEMORY_LOG"
        echo "" >> "$MEMORY_LOG"
        sleep 0.5
    done
) &
MONITOR_PID=$!

# Run inference
echo "Running inference test..."
START=$(date +%s)
curl -s -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"Write a detailed analysis of quantum computing.\"}], \"max_tokens\": 120}" \
  | tee "$TEST_DIR/response.json"  # Print to console AND save to file

END=$(date +%s)

# Stop memory monitoring
echo "Stopping memory monitoring..."
kill $MONITOR_PID 2>/dev/null || true
wait $MONITOR_PID 2>/dev/null || true

# Extract peak memory from monitoring log
echo "Analyzing peak memory from monitoring..." > "$TEST_DIR/peak_memory_analysis.txt"
echo "Total monitoring duration: $((END - START)) seconds" >> "$TEST_DIR/peak_memory_analysis.txt"

# Extract RSS values for dnet processes
grep "dnet-" "$MEMORY_LOG" | grep -o " [0-9]\+ " | sort -n | tail -5 >> "$TEST_DIR/peak_memory_analysis.txt"
echo "Peak RSS values (KB):" >> "$TEST_DIR/peak_memory_analysis.txt"
grep "dnet-" "$MEMORY_LOG" | grep -o " [0-9]\+ " | sort -nr | head -3 >> "$TEST_DIR/peak_memory_analysis.txt"

# Extract communication budget and other metrics from logs
echo "Extracting communication and memory metrics from logs..."
for log_file in ~/.dria/dnet/logs/dnet-shard-*.log; do
    if [ -f "$log_file" ]; then
        # Extract COMM_BUDGET entries (bytes/token metrics)
        grep "\[COMM_BUDGET\]" "$log_file" >> "$TEST_DIR/comm_budget.txt" 2>/dev/null || true

        # Extract MEMORY_SNAPSHOT entries (stage-wise memory analysis)
        grep "\[MEMORY_SNAPSHOT\]" "$log_file" >> "$TEST_DIR/memory_snapshots.txt" 2>/dev/null || true

        # Extract compute method calls for debugging
        grep "Runtime\.compute.*called" "$log_file" >> "$TEST_DIR/compute_calls.txt" 2>/dev/null || true

        # Extract CONFIG entries to verify settings were applied
        grep "\[CONFIG\]" "$log_file" >> "$TEST_DIR/config_log.txt" 2>/dev/null || true
    fi
done

# Analyze communication costs
if [ -f "$TEST_DIR/comm_budget.txt" ]; then
    echo "Communication Budget Analysis:" > "$TEST_DIR/comm_analysis.txt"
    echo "Total communication entries: $(wc -l < "$TEST_DIR/comm_budget.txt")" >> "$TEST_DIR/comm_analysis.txt"
    echo "" >> "$TEST_DIR/comm_analysis.txt"

    # Extract bytes_per_token values
    grep "bytes_per_token" "$TEST_DIR/comm_budget.txt" | \
    sed 's/.*bytes_per_token=\([0-9.]\+\).*/\1/' | \
    sort -n > "$TEST_DIR/bytes_per_token_values.txt"

    if [ -s "$TEST_DIR/bytes_per_token_values.txt" ]; then
        echo "Bytes per token statistics:" >> "$TEST_DIR/comm_analysis.txt"
        echo "Min: $(head -1 "$TEST_DIR/bytes_per_token_values.txt")" >> "$TEST_DIR/comm_analysis.txt"
        echo "Max: $(tail -1 "$TEST_DIR/bytes_per_token_values.txt")" >> "$TEST_DIR/comm_analysis.txt"
        echo "Avg: $(awk '{sum+=$1} END {print sum/NR}' "$TEST_DIR/bytes_per_token_values.txt")" >> "$TEST_DIR/comm_analysis.txt"
    fi

    echo "" >> "$TEST_DIR/comm_analysis.txt"
    echo "Sample entries:" >> "$TEST_DIR/comm_analysis.txt"
    head -5 "$TEST_DIR/comm_budget.txt" >> "$TEST_DIR/comm_analysis.txt"
fi

# Compare with budget
if [ -f "$RESULTS_DIR/budget_baseline.txt" ]; then
    echo "Budget vs Actual Comparison:" > "$TEST_DIR/budget_comparison.txt"
    echo "Expected from budget calculator:" >> "$TEST_DIR/budget_comparison.txt"
    tail -n 10 "$RESULTS_DIR/budget_baseline.txt" >> "$TEST_DIR/budget_comparison.txt" 2>/dev/null || true
    echo "" >> "$TEST_DIR/budget_comparison.txt"

    if [ -f "$TEST_DIR/peak_memory_analysis.txt" ]; then
        echo "Actual peak memory from monitoring:" >> "$TEST_DIR/budget_comparison.txt"
        cat "$TEST_DIR/peak_memory_analysis.txt" >> "$TEST_DIR/budget_comparison.txt"
    fi

    if [ -f "$TEST_DIR/comm_analysis.txt" ]; then
        echo "" >> "$TEST_DIR/budget_comparison.txt"
        echo "Communication costs:" >> "$TEST_DIR/budget_comparison.txt"
        grep "Avg:" "$TEST_DIR/comm_analysis.txt" >> "$TEST_DIR/budget_comparison.txt" 2>/dev/null || true
    fi

    if [ -f "$TEST_DIR/memory_snapshots.txt" ]; then
        echo "" >> "$TEST_DIR/budget_comparison.txt"
        echo "Stage-wise memory snapshots:" >> "$TEST_DIR/budget_comparison.txt"
        echo "Total snapshots: $(wc -l < "$TEST_DIR/memory_snapshots.txt")" >> "$TEST_DIR/budget_comparison.txt"
        # Extract peak memory values from snapshots
        grep "total=" "$TEST_DIR/memory_snapshots.txt" | sed 's/.*total=\([0-9.]\+\)MB.*/\1/' | sort -n | tail -1 | xargs -I {} echo "Peak stage memory: {} MB" >> "$TEST_DIR/budget_comparison.txt" 2>/dev/null || true
    fi

    if [ -f "$TEST_DIR/compute_calls.txt" ]; then
        echo "" >> "$TEST_DIR/budget_comparison.txt"
        echo "Compute method calls:" >> "$TEST_DIR/budget_comparison.txt"
        echo "Total compute calls: $(wc -l < "$TEST_DIR/compute_calls.txt")" >> "$TEST_DIR/budget_comparison.txt"
    fi
fi

# Summary
cat > "$TEST_DIR/summary.txt" << EOF
Test: $TEST_NAME
Config: $TEST_TYPE
Duration: $((END - START))s
Timestamp: $(date)
Inference: $(grep -c "choices" "$TEST_DIR/response.json" 2>/dev/null || echo "unknown") completions
Model: $MODEL_NAME
EOF

echo ""
echo "✓ $TEST_NAME test completed successfully!"
echo "Results saved to: $TEST_DIR"
echo ""
echo "Files created:"
echo "  response.json           - Inference API response"
echo "  memory_monitoring.log   - External memory monitoring (every 0.5s)"
echo "  peak_memory_analysis.txt - Peak memory statistics from monitoring"
echo "  comm_budget.txt         - Communication costs (bytes/token)"
echo "  comm_analysis.txt       - Communication statistics"
echo "  memory_snapshots.txt    - Stage-wise memory snapshots"
echo "  compute_calls.txt       - Compute method call tracking"
echo "  config_log.txt          - Runtime configuration verification"
echo "  budget_comparison.txt   - Budget vs actual comparison"
echo "  summary.txt             - Test summary"