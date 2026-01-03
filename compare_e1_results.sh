#!/bin/bash
# Compare two E1 test results
#
# Usage:
#   ./compare_e1_results.sh e1_baseline_20260103_120000 e1_compressed_20260103_130000

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <baseline_results_dir> <comparison_results_dir>"
    echo ""
    echo "Example:"
    echo "  $0 e1_baseline_20260103_120000 e1_compressed_20260103_130000"
    echo ""
    echo "Available result directories:"
    ls -d e1_*/ 2>/dev/null || echo "  (none found)"
    exit 1
fi

DIR1="$1"
DIR2="$2"

if [ ! -d "$DIR1" ] || [ ! -d "$DIR2" ]; then
    echo "ERROR: One or both directories not found"
    exit 1
fi

echo "=============================================="
echo "E1 RESULTS COMPARISON"
echo "=============================================="
echo "Baseline:   $DIR1"
echo "Comparison: $DIR2"
echo ""

# Extract test types
TYPE1=$(basename "$DIR1" | sed 's/e1_\([^_]*\)_.*$/\1/')
TYPE2=$(basename "$DIR2" | sed 's/e1_\([^_]*\)_.*$/\1/')

echo "Test Types: $TYPE1 vs $TYPE2"
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
echo "  Budget vs actual: */budget_comparison.txt"
echo "  Raw monitoring: */memory_monitoring.log"
echo ""
echo "For detailed analysis, compare:"
echo "  diff $DIR1/budget_comparison.txt $DIR2/budget_comparison.txt"

