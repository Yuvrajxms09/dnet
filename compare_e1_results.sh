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
echo "Interpretation:"
echo "  baseline vs compressed/sparse: Tests if fp16 sparse compression helps"
echo "  baseline vs q8: Tests if Q8 quantization (like DLlama Q80) closes the gap"
echo ""
echo "Expected results:"
echo "  - Lower peak memory → Wire format matters (H1 confirmed)"
echo "  - Lower bytes per activation → Compression working"
echo "  - Same memory → Wire format not the bottleneck (investigate H2/H3/H4)"
echo "  - Q8 shows bigger improvement than sparse → True quantization needed"
echo ""
echo "For detailed analysis, compare:"
echo "  diff $DIR1/analysis.txt $DIR2/analysis.txt"

