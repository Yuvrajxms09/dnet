#!/usr/bin/env python3
"""
Memory Budget Calculator for Issue #73

Calculates expected memory usage per stage without running inference.
Uses model metadata to compute exact memory requirements.

Usage:
    python scripts/memory_budget.py --model <model_path> --stage-layers "0-39" "40-79" --seq-len 4096 --pools 512
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dnet.utils.model import get_model_metadata, ModelMetadata
from dnet.utils.logger import logger


def parse_layer_ranges(ranges: List[str]) -> List[List[int]]:
    """Parse layer range strings like '0-39' or '0-15,32-47' into lists of integers."""
    result = []
    for range_str in ranges:
        layer_list = []
        # Split by comma for multiple ranges like "0-15,32-47"
        for part in range_str.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-")
                layer_list.extend(range(int(start), int(end) + 1))
            else:
                layer_list.append(int(part))
        result.append(layer_list)
    return result


def get_topology_from_api(api_url: str) -> Optional[Dict[str, Any]]:
    """Fetch topology information from running API server."""
    try:
        import requests
        response = requests.get(f"{api_url}/v1/topology", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"Could not fetch topology from {api_url}: {e}")
        return None


def extract_stage_layers_from_topology(topology: Dict[str, Any]) -> List[List[int]]:
    """Extract layer assignments from topology for each stage."""
    assignments = topology.get("assignments", [])
    stage_layers = []

    for assignment in assignments:
        # For single-round (k=1), flatten all layer rounds
        layers = assignment.get("layers", [])
        if layers:
            # Flatten all rounds into a single list per stage
            flat_layers = []
            for round_layers in layers:
                flat_layers.extend(round_layers)
            stage_layers.append(sorted(flat_layers))

    return stage_layers


def calculate_weights_memory(metadata: ModelMetadata, layer_ids: List[int]) -> float:
    """Calculate memory for layer weights."""
    total_bytes = 0
    for layer_id in layer_ids:
        if layer_id in metadata.weight_info:
            layer_tensors = metadata.weight_info[layer_id]
            total_bytes += sum(tensor.size_bytes for tensor in layer_tensors.values())
    return total_bytes / (1024 * 1024)  # MB


def calculate_embedding_memory(metadata: ModelMetadata, layer_ids: List[int]) -> float:
    """Calculate memory for embeddings (if layer 0 is assigned)."""
    if 0 in layer_ids:
        return sum(tensor.size_bytes for tensor in metadata.embed_tokens.values()) / (1024 * 1024)
    return 0.0


def calculate_lm_head_memory(metadata: ModelMetadata, layer_ids: List[int]) -> float:
    """Calculate memory for LM head (if last layer is assigned)."""
    num_layers = metadata.num_layers
    last_layer = num_layers - 1

    if last_layer in layer_ids:
        # Check if LM head is tied to embeddings
        tied = getattr(metadata.config, 'tie_word_embeddings', False)
        if tied and 0 in layer_ids:
            # LM head is tied, don't double-count
            return 0.0
        return sum(tensor.size_bytes for tensor in metadata.lm_head.values()) / (1024 * 1024)
    return 0.0


def calculate_kv_cache_memory(metadata: ModelMetadata, layer_ids: List[int],
                            seq_len: int, kv_bits: int) -> float:
    """Calculate KV cache memory for given sequence length."""
    if not layer_ids:
        return 0.0

    # Extract KV cache parameters from model config
    config = metadata.model_config
    num_kv_heads = config.get('num_key_value_heads', config.get('num_attention_heads', 8))
    head_dim = config.get('head_dim', 128)
    rope_theta = config.get('rope_theta', 10000.0)

    # KV cache formula: 2 * num_layers * num_kv_heads * seq_len * head_dim * dtype_bytes
    # Factor of 2 for K and V
    num_layers_local = len(layer_ids)
    dtype_bytes = kv_bits // 8

    kv_bytes = 2 * num_layers_local * num_kv_heads * seq_len * head_dim * dtype_bytes
    return kv_bytes / (1024 * 1024)  # MB


def calculate_pools_memory(input_pool_mb: int, output_pool_mb: int) -> float:
    """Calculate memory for input/output pools."""
    return input_pool_mb + output_pool_mb


def calculate_activation_buffer_memory(metadata: ModelMetadata, wire_dtype_bits: int = 16) -> float:
    """Calculate memory for activation buffer (1 token)."""
    hidden_size = metadata.embedding_size
    dtype_bytes = wire_dtype_bits // 8
    return (hidden_size * dtype_bytes) / (1024 * 1024)  # MB


def calculate_stage_budget(metadata: ModelMetadata, layer_ids: List[int],
                          seq_len: int, kv_bits: int, input_pool_mb: int,
                          output_pool_mb: int, wire_dtype_bits: int = 16) -> Dict[str, float]:
    """Calculate complete memory budget for a stage."""

    weights_mb = calculate_weights_memory(metadata, layer_ids)
    embed_mb = calculate_embedding_memory(metadata, layer_ids)
    lm_head_mb = calculate_lm_head_memory(metadata, layer_ids)
    kv_mb = calculate_kv_cache_memory(metadata, layer_ids, seq_len, kv_bits)
    pools_mb = calculate_pools_memory(input_pool_mb, output_pool_mb)
    activation_mb = calculate_activation_buffer_memory(metadata, wire_dtype_bits)

    total_mb = weights_mb + embed_mb + lm_head_mb + kv_mb + pools_mb

    return {
        'weights_mb': weights_mb,
        'embed_mb': embed_mb,
        'lm_head_mb': lm_head_mb,
        'kv_mb': kv_mb,
        'pools_mb': pools_mb,
        'activation_mb': activation_mb,
        'total_mb': total_mb,
        'layer_count': len(layer_ids),
    }


def print_budget_table(metadata: ModelMetadata, stage_budgets: List[Dict[str, float]],
                      seq_len: int, kv_bits: int, input_pool_mb: int, output_pool_mb: int):
    """Print formatted memory budget table."""

    print("=" * 80)
    print("MEMORY BUDGET CALCULATOR - Issue #73")
    print("=" * 80)
    print(f"Model: {metadata.path}")
    print(f"Total layers: {metadata.num_layers}")
    print(f"Sequence length: {seq_len}")
    print(f"KV cache precision: {kv_bits} bits")
    print(f"Pools: {input_pool_mb}MB input + {output_pool_mb}MB output")
    print()

    # Header
    print("┌" + "─" * 78 + "┐")
    print("│ STAGE MEMORY BREAKDOWN (MB)                                           │")
    print("├" + "─" * 78 + "┤")
    print("│ Stage │ Layers │ Weights  │ Embed  │ LM Head│ KV     │ Pools  │ Total  │")
    print("├" + "─" * 78 + "┤")

    for i, budget in enumerate(stage_budgets):
        stage_type = "start" if 0 in stage_budgets[i].get('layer_ids', []) else \
                    "end" if (metadata.num_layers - 1) in stage_budgets[i].get('layer_ids', []) else "middle"

        print("│" +
              f" {i:>5} │" +
              f" {budget['layer_count']:>6} │" +
              f" {budget['weights_mb']:>7.0f} │" +
              f" {budget['embed_mb']:>6.0f} │" +
              f" {budget['lm_head_mb']:>7.0f} │" +
              f" {budget['kv_mb']:>6.0f} │" +
              f" {budget['pools_mb']:>6.0f} │" +
              f" {budget['total_mb']:>6.0f} │")

    print("└" + "─" * 78 + "┘")
    print()


def main():
    parser = argparse.ArgumentParser(description="Calculate memory budget for DNET stages")
    parser.add_argument("--model", required=True, help="Model path or HuggingFace ID")
    parser.add_argument("--stage-layers", nargs="+",
                       help="Layer ranges for each stage (e.g., '0-39' '40-79'). If not provided, fetches from --api-url")
    parser.add_argument("--api-url", default="http://localhost:8080",
                       help="API URL to fetch topology from (default: http://localhost:8080)")
    parser.add_argument("--seq-len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--kv-bits", type=int, default=16, help="KV cache bits (8, 16, 32)")
    parser.add_argument("--input-pool-mb", type=int, default=512, help="Input pool size MB")
    parser.add_argument("--output-pool-mb", type=int, default=512, help="Output pool size MB")
    parser.add_argument("--wire-dtype-bits", type=int, default=16, help="Wire dtype bits")

    args = parser.parse_args()

    try:
        # Get model metadata
        metadata = get_model_metadata(args.model)
        logger.info(f"Loaded metadata for model with {metadata.num_layers} layers")

        # Get stage layer assignments
        if args.stage_layers:
            # Manual specification
            stage_layers = parse_layer_ranges(args.stage_layers)
        else:
            # Try to fetch from API
            topology = get_topology_from_api(args.api_url)
            if topology:
                stage_layers = extract_stage_layers_from_topology(topology)
                logger.info(f"Fetched topology with {len(stage_layers)} stages from {args.api_url}")
            else:
                logger.error("No --stage-layers provided and could not fetch topology from API")
                sys.exit(1)

        if not stage_layers:
            logger.error("No stage layers found")
            sys.exit(1)

        # Calculate budget for each stage
        stage_budgets = []
        for layers in stage_layers:
            budget = calculate_stage_budget(
                metadata, layers, args.seq_len, args.kv_bits,
                args.input_pool_mb, args.output_pool_mb, args.wire_dtype_bits
            )
            budget['layer_ids'] = layers  # Store for printing
            stage_budgets.append(budget)

        # Print results
        print_budget_table(metadata, stage_budgets, args.seq_len, args.kv_bits,
                          args.input_pool_mb, args.output_pool_mb)

        # Print summary for analysis
        print("SUMMARY FOR ANALYSIS:")
        for i, budget in enumerate(stage_budgets):
            print(f"Stage {i}: {budget['total_mb']:.0f} MB expected total")
            print(f"  - Largest component: {max(budget['weights_mb'], budget['embed_mb'], budget['lm_head_mb'], budget['kv_mb'], budget['pools_mb']):.0f} MB")

    except Exception as e:
        logger.error(f"Failed to calculate budget: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
