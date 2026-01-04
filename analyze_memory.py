#!/usr/bin/env python3
"""
Analyze E1 test results and provide clear memory usage comparison.
Usage: python3 analyze_memory.py <baseline_dir> <compressed_dir>
"""

import sys
import os
import re
from pathlib import Path

def parse_vm_stat_log(log_path):
    """Parse vm_stat log and extract memory metrics over time."""
    pages_active = []
    pages_wired = []

    with open(log_path, 'r') as f:
        for line in f:
            if 'Pages active:' in line:
                # Extract number after "Pages active:"
                match = re.search(r'Pages active:\s*(\d+)', line)
                if match:
                    pages_active.append(int(match.group(1)))
            elif 'Pages wired down:' in line:
                match = re.search(r'Pages wired down:\s*(\d+)', line)
                if match:
                    pages_wired.append(int(match.group(1)))

    return pages_active, pages_wired

def pages_to_gb(pages):
    """Convert pages to GB (16KB pages)."""
    return pages * 16384 / (1024**3)

def analyze_test_dir(test_dir):
    """Analyze a single test directory."""
    test_path = Path(test_dir)

    # Find the memory monitoring log (might be in subdirectory)
    memory_log = None
    for log_file in test_path.rglob("memory_monitoring.log"):
        memory_log = log_file
        break

    if not memory_log or not memory_log.exists():
        print(f"ERROR: Memory log not found in: {test_path}")
        return None

    print(f"📊 Analyzing: {test_dir}")

    # Parse memory data
    pages_active, pages_wired = parse_vm_stat_log(memory_log)

    if not pages_active:
        print("❌ No memory data found")
        return None

    # Calculate metrics
    max_active_pages = max(pages_active)
    max_wired_pages = max(pages_wired) if pages_wired else 0
    max_total_pages = max_active_pages + max_wired_pages

    max_active_gb = pages_to_gb(max_active_pages)
    max_wired_gb = pages_to_gb(max_wired_pages)
    max_total_gb = pages_to_gb(max_total_pages)

    # Find stable memory (last 10 readings, ignore spikes)
    stable_active = pages_active[-10:] if len(pages_active) >= 10 else pages_active
    stable_avg_pages = sum(stable_active) / len(stable_active)
    stable_avg_gb = pages_to_gb(stable_avg_pages)

    # Find inference peak (middle portion, avoid startup/end)
    if len(pages_active) > 20:
        middle_start = len(pages_active) // 4
        middle_end = 3 * len(pages_active) // 4
        inference_peak = max(pages_active[middle_start:middle_end])
        inference_peak_gb = pages_to_gb(inference_peak)
    else:
        inference_peak = max_active_pages
        inference_peak_gb = max_active_gb

    results = {
        'max_active_gb': max_active_gb,
        'max_wired_gb': max_wired_gb,
        'max_total_gb': max_total_gb,
        'inference_peak_gb': inference_peak_gb,
        'stable_avg_gb': stable_avg_gb,
        'samples': len(pages_active)
    }

    print(f"   🔺 Peak Active: {max_active_gb:.1f}GB")
    print(f"   🔸 Peak Wired: {max_wired_gb:.1f}GB")
    print(f"   📊 Peak Total: {max_total_gb:.1f}GB")
    print(f"   🎯 Inference Peak: {inference_peak_gb:.1f}GB")
    print(f"   📈 Samples: {results['samples']}")

    return results

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 analyze_memory.py <baseline_dir> <compressed_dir>")
        print("Example: python3 analyze_memory.py e1_results_20260103_142028 e1_results_20260103_144754")
        sys.exit(1)

    baseline_dir = sys.argv[1]
    compressed_dir = sys.argv[2]

    print("🔬 E1 Memory Analysis: Baseline vs qsparse8_v1 Compression")
    print("=" * 60)

    # Analyze both tests
    baseline_results = analyze_test_dir(baseline_dir)
    compressed_results = analyze_test_dir(compressed_dir)

    if not baseline_results or not compressed_results:
        print("❌ Analysis failed")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("🎯 COMPARISON RESULTS")
    print("=" * 60)

    # Compare key metrics
    metrics = [
        ('Peak Memory (Inference)', 'inference_peak_gb'),
        ('Absolute Max Memory', 'max_total_gb'),
        ('Stable Memory (End)', 'stable_avg_gb')
    ]

    for label, key in metrics:
        baseline_val = baseline_results[key]
        compressed_val = compressed_results[key]
        diff = compressed_val - baseline_val
        diff_pct = (diff / baseline_val) * 100 if baseline_val > 0 else 0

        print(f"\n{label}:")
        print(f"   📊 Baseline: {baseline_val:.1f}GB")
        print(f"   🗜️  Compressed: {compressed_val:.1f}GB")
        print(f"   ⚡ Difference: {diff:+.1f}GB ({diff_pct:+.1f}%)")
    print("\n" + "=" * 60)
    print("📋 INTERPRETATION")
    print("=" * 60)

    peak_diff = compressed_results['inference_peak_gb'] - baseline_results['inference_peak_gb']

    if peak_diff > 1.0:
        print("🔴 COMPRESSION INCREASES memory usage (>1GB worse)")
        print("   → H1 disproven: Wire compression makes OOM worse")
    elif peak_diff > 0.1:
        print("🟡 COMPRESSION slightly increases memory (0.1-1GB)")
        print("   → Minimal impact, investigate other causes")
    elif abs(peak_diff) <= 0.1:
        print("🟢 COMPRESSION has no significant effect")
        print("   → Wire format not the bottleneck (H1 disproven)")
    elif peak_diff < -0.5:
        print("🟢 COMPRESSION reduces memory (>0.5GB better)")
        print("   → H1 confirmed: Wire compression helps OOM")
    else:
        print("🟡 COMPRESSION slightly reduces memory (<0.5GB)")
        print("   → Small benefit, but investigate other causes")

    print(f"\n💡 Recommendation: Focus on H{'2' if abs(peak_diff) <= 0.1 else '1'} (embeddings/LM head) for OOM fix")

if __name__ == "__main__":
    main()
