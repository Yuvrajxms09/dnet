
#!/bin/bash
# Compare two E1 test results
#
# Usage:
#   ./compare_e1_results.sh <baseline_results_dir> <compressed_results_dir>
#
# Where each dir contains: e1_results_YYYYMMDD_HHMMSS/[baseline_fp16|compressed_qsparse8]/

set -e

if [ $# -ne 2 ]; then
    echo "Usage: $0 <baseline_results_dir> <compressed_results_dir>"
    echo ""
    echo "Example:"
    echo "  $0 e1_results_20260103_120000 e1_results_20260103_130000"
    echo ""
    echo "Each directory should contain test results in format:"
    echo "  e1_results_YYYYMMDD_HHMMSS/baseline_fp16/"
    echo "  e1_results_YYYYMMDD_HHMMSS/compressed_qsparse8/"
    echo ""
    echo "Available result directories:"
    ls -d e1_results_*/ 2>/dev/null || echo "  (none found)"
    exit 1
fi

BASELINE_DIR="$1"
COMPRESSED_DIR="$2"

# Find the actual test subdirectories
DIR1=$(find "$BASELINE_DIR" -name "baseline_fp16" -type d | head -1)
DIR2=$(find "$COMPRESSED_DIR" -name "compressed_qsparse8" -type d | head -1)

if [ -z "$DIR1" ] || [ -z "$DIR2" ]; then
    echo "ERROR: Could not find test directories"
    echo "Expected: baseline_fp16/ and compressed_qsparse8/ subdirectories"
    exit 1
fi

if [ ! -d "$DIR1" ] || [ ! -d "$DIR2" ]; then
    echo "ERROR: Test directories not found"
    echo "DIR1: $DIR1"
    echo "DIR2: $DIR2"
    exit 1
fi

echo "=============================================="
echo "E1 RESULTS COMPARISON"
echo "=============================================="
echo "Baseline:   $DIR1"
echo "Compressed: $DIR2"
echo ""

echo "=== Configuration Verification ==="
echo "--- $DIR1 ---"
cat "$DIR1/config_log.txt" 2>/dev/null || echo "(no config log)"
echo ""
echo "--- $DIR2 ---"
cat "$DIR2/config_log.txt" 2>/dev/null || echo "(no config log)"
echo ""

echo "=== Peak Memory from External Monitoring ==="
echo "--- $DIR1 (peak RSS in KB) ---"
grep "Peak RSS" "$DIR1/peak_memory_analysis.txt" 2>/dev/null || echo "(no peak analysis)"
echo ""
echo "--- $DIR2 (peak RSS in KB) ---"
grep "Peak RSS" "$DIR2/peak_memory_analysis.txt" 2>/dev/null || echo "(no peak analysis)"
echo ""

echo "=== Communication Budget Comparison ==="
echo "--- $DIR1 (bytes/token statistics) ---"
if [ -f "$DIR1/comm_analysis.txt" ]; then
    grep -E "(Min:|Max:|Avg:)" "$DIR1/comm_analysis.txt" 2>/dev/null || echo "(no stats)"
else
    echo "(no comm analysis)"
fi
echo ""
echo "--- $DIR2 (bytes/token statistics) ---"
if [ -f "$DIR2/comm_analysis.txt" ]; then
    grep -E "(Min:|Max:|Avg:)" "$DIR2/comm_analysis.txt" 2>/dev/null || echo "(no stats)"
else
    echo "(no comm analysis)"
fi
echo ""

echo "=== Memory Budget vs Actual ==="
echo "--- $DIR1 ---"
if [ -f "$DIR1/budget_comparison.txt" ]; then
    tail -10 "$DIR1/budget_comparison.txt" 2>/dev/null || echo "(no budget comparison)"
else
    echo "(no budget comparison)"
fi
echo ""
echo "--- $DIR2 ---"
if [ -f "$DIR2/budget_comparison.txt" ]; then
    tail -10 "$DIR2/budget_comparison.txt" 2>/dev/null || echo "(no budget comparison)"
else
    echo "(no budget comparison)"
fi
echo ""

echo "=== Stage-wise Memory Analysis ==="
echo "--- $DIR1 (stage memory snapshots) ---"
if [ -f "$DIR1/memory_snapshots.txt" ]; then
    echo "Total snapshots: $(wc -l < "$DIR1/memory_snapshots.txt")"
    grep "total=" "$DIR1/memory_snapshots.txt" | sed 's/.*total=\([0-9.]\+\)MB.*/\1/' | sort -n | tail -1 | xargs -I {} echo "Peak stage memory: {} MB" 2>/dev/null || echo "(no peak data)"
else
    echo "(no memory snapshots)"
fi
echo ""
echo "--- $DIR2 (stage memory snapshots) ---"
if [ -f "$DIR2/memory_snapshots.txt" ]; then
    echo "Total snapshots: $(wc -l < "$DIR2/memory_snapshots.txt")"
    grep "total=" "$DIR2/memory_snapshots.txt" | sed 's/.*total=\([0-9.]\+\)MB.*/\1/' | sort -n | tail -1 | xargs -I {} echo "Peak stage memory: {} MB" 2>/dev/null || echo "(no peak data)"
else
    echo "(no memory snapshots)"
fi
echo ""

echo "=============================================="
echo "INTERPRETATION"
echo "=============================================="
echo ""
echo "E1 Hypothesis H1: Inter-stage fp16 doubles activation traffic vs qsparse8_v1"
echo ""
echo "Expected results:"
echo "  - Lower peak memory in qsparse8_v1 → Wire format matters (H1 confirmed)"
echo "  - Lower bytes/token in qsparse8_v1 → Compression working"
echo "  - Same memory between variants → Wire format not bottleneck (investigate H2/H3/H4)"
echo ""
echo "Key files for analysis:"
echo "  Peak memory: */peak_memory_analysis.txt"
echo "  Communication: */comm_analysis.txt"
echo "  Stage memory: */memory_snapshots.txt"
echo "  Budget vs actual: */budget_comparison.txt"
echo "  Raw monitoring: */memory_monitoring.log"
echo ""
echo "For detailed analysis, compare:"
echo "  diff $DIR1/budget_comparison.txt $DIR2/budget_comparison.txt"
echo ""
echo "=============================================="
echo "USAGE SUMMARY"
echo "=============================================="
echo "1. Set baseline config: cp baseline.config .env"
echo "2. Start services: ./dnet-api & ./dnet-shard shard-1 & ./dnet-shard shard-2 &"
echo "3. Load model via dnet-tui"
echo "4. Run baseline test: ./run_e1_test.sh baseline"
echo "5. Stop services: pkill -f 'dnet-api' && pkill -f 'dnet-shard'"
echo "6. Set compressed config: cp compressed.config .env"
echo "7. Restart services with new config"
echo "8. Load model again via dnet-tui"
echo "9. Run compressed test: ./run_e1_test.sh compressed"
echo "10. Compare results: ./compare_e1_results.sh <baseline_dir> <compressed_dir>"
