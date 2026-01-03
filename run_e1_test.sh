#!/bin/bash
# Run E1: Inter-stage dtype compression test - assumes API/shard are running and model is loaded

set -e

BASE_URL="http://localhost:8080"
RESULTS_DIR="e1_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

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

echo "=== Detected Model: $MODEL_NAME ==="

# Run budget calculator first (auto-detects model and topology from loaded system)
echo "=== Running Memory Budget Calculator ==="
uv run python3 scripts/memory_budget.py \
  --seq-len 2048 \
  --pools 512 \
  > "$RESULTS_DIR/budget_baseline.txt"

run_test() {
    local name=$1
    local config=$2

    echo "=== Running $name test ==="

    # Stop services to allow config reload
    echo "Stopping services for config change..."
    pkill -f "dnet-api" || true
    pkill -f "dnet-shard" || true
    sleep 2

    # Setup config
    cp "$config" .env || { echo "Config file $config not found"; exit 1; }
    echo "Config loaded:"
    cat .env

    TEST_DIR="$RESULTS_DIR/$name"
    mkdir -p "$TEST_DIR"

    # Start services with new config
    echo "Starting services with new config..."
    # Note: You'll need to start your shards and API manually here
    # For example:
    # ./dnet-api &  # Start API
    # ./dnet-shard shard-1 &  # Start shard 1
    # ./dnet-shard shard-2 &  # Start shard 2
    echo "Please start your dnet-api and dnet-shard processes now..."
    echo "Press Enter when services are running and model is loaded..."
    read -r

    # Check if services are running
    if ! curl -s -f "$BASE_URL/health" > /dev/null; then
        echo "ERROR: API not running at $BASE_URL"
        return 1
    fi

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
    echo "Running inference..."
    START=$(date +%s)
    curl -s -X POST "$BASE_URL/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\": \"$MODEL_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"hi there\"}], \"max_tokens\": 50}" \
      > "$TEST_DIR/response.json"

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

            # Extract any remaining MEMORY_SNAPSHOT entries (if they exist)
            grep "\[MEMORY_SNAPSHOT\]" "$log_file" >> "$TEST_DIR/memory_snapshots.txt" 2>/dev/null || true

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
    fi

           # Summary
           cat > "$TEST_DIR/summary.txt" << EOF
Test: $name
Config: $config
Duration: $((END - START))s
Timestamp: $(date)
Inference: $(grep -c "choices" "$TEST_DIR/response.json" 2>/dev/null || echo "unknown") completions
EOF

           echo "✓ $name test completed - results in $TEST_DIR"
}

# Run E1 tests: fp16 wire vs qsparse8_v1 compression
echo ""
echo "IMPORTANT: Tests must be run sequentially with service restarts between them."
echo "The script will stop services, change config, then wait for you to restart services."
echo ""

echo "=== Test 1: Baseline (fp16 wire, no compression) ==="
run_test "baseline_fp16" "baseline.config"

echo ""
echo "=== Test 1 Complete ==="
echo "Results saved to: $RESULTS_DIR/baseline_fp16/"
echo ""
echo "=== Next: Test 2 (qsparse8_v1 compression) ==="
echo "The script will now stop services and setup compressed config."
echo "After services restart with new config, run:"
echo "  ./run_e1_test.sh compressed"
echo ""

# Check if argument provided for single test
if [ $# -eq 1 ]; then
    case $1 in
        "baseline")
            run_test "baseline_fp16" "baseline.config"
            ;;
        "compressed")
            run_test "compressed_qsparse8" "compressed.config"
            ;;
        *)
            echo "Usage: $0 [baseline|compressed]"
            echo "Run without args for full workflow instructions"
            exit 1
            ;;
    esac
else
    echo ""
    echo "Run with argument for single test:"
    echo "  ./run_e1_test.sh baseline    # Run only baseline test"
    echo "  ./run_e1_test.sh compressed  # Run only compressed test"
fi

echo ""
echo "Files created per test:"
echo "  response.json          - Inference API response"
echo "  memory_monitoring.log  - External memory monitoring (every 0.5s)"
echo "  peak_memory_analysis.txt - Peak memory statistics from monitoring"
echo "  comm_budget.txt        - Communication costs (bytes/token)"
echo "  comm_analysis.txt      - Communication statistics"
echo "  config_log.txt         - Runtime configuration verification"
echo "  budget_comparison.txt  - Budget vs actual comparison"
echo "  summary.txt            - Test summary"
echo ""
echo "To run complete E1 test suite:"
echo "1. ./run_e1_test.sh baseline    # Run baseline test"
echo "2. Restart dnet-api and dnet-shard services"
echo "3. ./run_e1_test.sh compressed  # Run compressed test"
echo "4. ./compare_e1_results.sh e1_results_*/baseline_fp16 e1_results_*/compressed_qsparse8"
echo ""
echo "Key metrics for H1 hypothesis:"
echo "  - Lower peak memory in compressed vs baseline = wire format matters"
echo "  - Lower bytes/token in compressed = compression working"
echo "  - Same memory = wire format not bottleneck (investigate H2/H3/H4)"