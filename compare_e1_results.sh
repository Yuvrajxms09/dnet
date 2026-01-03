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

echo "=== Active Config Comparison ==="
echo "--- $DIR1 ---"
cat "$DIR1/active_config.txt" 2>/dev/null || echo "(not found)"
echo ""
echo "--- $DIR2 ---"
cat "$DIR2/active_config.txt" 2>/dev/null || echo "(not found)"
echo ""

echo "=== Memory Budget (Theoretical) ==="
echo "Both should show same theoretical budget (model/topology unchanged)"
echo ""

echo "=== Memory Snapshots Comparison ==="
echo "--- $DIR1 (peak memory) ---"
grep "peak=" "$DIR1/memory_snapshots.txt" 2>/dev/null | tail -5 || echo "(no snapshots)"
echo ""
echo "--- $DIR2 (peak memory) ---"
grep "peak=" "$DIR2/memory_snapshots.txt" 2>/dev/null | tail -5 || echo "(no snapshots)"
echo ""

echo "=== Stage Memory Comparison ==="
echo "--- $DIR1 (total memory per stage) ---"
grep "total=" "$DIR1/stage_memory.txt" 2>/dev/null | tail -5 || echo "(no stage memory)"
echo ""
echo "--- $DIR2 (total memory per stage) ---"
grep "total=" "$DIR2/stage_memory.txt" 2>/dev/null | tail -5 || echo "(no stage memory)"
echo ""

echo "=== Communication Budget Comparison ==="
echo "--- $DIR1 (bytes per activation) ---"
grep "bytes=" "$DIR1/comm_budget.txt" 2>/dev/null | tail -3 || echo "(no comm budget)"
echo ""
echo "--- $DIR2 (bytes per activation) ---"
grep "bytes=" "$DIR2/comm_budget.txt" 2>/dev/null | tail -3 || echo "(no comm budget)"
echo ""

echo "=============================================="
echo "INTERPRETATION"
echo "=============================================="
echo ""
echo "If $DIR2 shows:"
echo "  - Lower peak memory → H1 confirmed (compression helps)"
echo "  - Lower bytes per activation → Wire compression working"
echo "  - Same memory as $DIR1 → H1 not the cause, investigate H2/H3/H4"
echo ""
echo "For detailed analysis, compare:"
echo "  diff $DIR1/analysis.txt $DIR2/analysis.txt"

