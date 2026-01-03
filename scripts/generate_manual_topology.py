#!/usr/bin/env python3
"""
Generate Manual Topology for E1 Testing

Senior Dev Approach: Deterministic, simple, robust.
Splits layers evenly across 2 shards for fair E1 comparison.

Usage:
    python scripts/generate_manual_topology.py <model_name> <shard1_ip> <shard2_ip> [--shard1-name shard-1] [--shard2-name shard-2]

Example:
    python scripts/generate_manual_topology.py Qwen/Qwen3-32B-MLX-8bit 192.168.1.100 192.168.1.101

Outputs JSON for /v1/prepare_topology_manual API call.
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dnet.utils.model import get_model_metadata


def generate_even_split_topology(model_name: str, shard1_ip: str, shard2_ip: str,
                                shard1_name: str = "shard-1", shard2_name: str = "shard-2",
                                shard1_port: int = 8081, shard2_port: int = 8082,
                                shard1_grpc_port: int = 58081, shard2_grpc_port: int = 58082):
    """
    Generate manual topology with even layer split for 2 shards.

    Senior Dev Logic:
    - Simple deterministic split: first half layers → shard1, second half → shard2
    - No fancy load balancing - just even split for fair comparison
    - Works for any model, any number of layers
    """

    # Get model metadata
    try:
        metadata = get_model_metadata(model_name)
        num_layers = metadata.num_layers
        print(f"Model: {model_name}")
        print(f"Layers: {num_layers}")
    except Exception as e:
        print(f"ERROR: Failed to get model metadata for {model_name}: {e}")
        sys.exit(1)

    # Even split: first half to shard1, second half to shard2
    midpoint = num_layers // 2
    shard1_layers = list(range(0, midpoint))      # [0, 1, 2, ..., midpoint-1]
    shard2_layers = list(range(midpoint, num_layers))  # [midpoint, midpoint+1, ..., num_layers-1]

    print(f"Shard-1 layers: {len(shard1_layers)} ({shard1_layers[:3]}...{shard1_layers[-3:]})")
    print(f"Shard-2 layers: {len(shard2_layers)} ({shard2_layers[:3]}...{shard2_layers[-3:]})")

    # Generate topology JSON
    topology = {
        "model": model_name,
        "kv_bits": "8bit",  # Match our test setup
        "devices": [
            {
                "instance": shard1_name,
                "local_ip": shard1_ip,
                "server_port": shard1_port,
                "shard_port": shard1_grpc_port
            },
            {
                "instance": shard2_name,
                "local_ip": shard2_ip,
                "server_port": shard2_port,
                "shard_port": shard2_grpc_port
            }
        ],
        "assignments": [
            {
                "instance": shard1_name,
                "layers": [shard1_layers],  # Single round (k=1)
                "next_instance": shard2_name,
                "window_size": 64,  # Conservative default
                "residency_size": 8
            },
            {
                "instance": shard2_name,
                "layers": [shard2_layers],  # Single round (k=1)
                "next_instance": None,  # Connects back to API
                "window_size": 64,  # Conservative default
                "residency_size": 8
            }
        ],
        "num_layers": num_layers
    }

    return topology


def main():
    parser = argparse.ArgumentParser(
        description="Generate manual topology JSON for even layer split across 2 shards"
    )
    parser.add_argument("model", help="Model name (e.g., Qwen/Qwen3-32B-MLX-8bit)")
    parser.add_argument("shard1_ip", help="IP address of first shard")
    parser.add_argument("shard2_ip", help="IP address of second shard")
    parser.add_argument("--shard1-name", default="shard-1", help="Name of first shard (default: shard-1)")
    parser.add_argument("--shard2-name", default="shard-2", help="Name of second shard (default: shard-2)")
    parser.add_argument("--shard1-port", type=int, default=8081, help="HTTP port of first shard")
    parser.add_argument("--shard2-port", type=int, default=8082, help="HTTP port of second shard")
    parser.add_argument("--shard1-grpc-port", type=int, default=58081, help="gRPC port of first shard")
    parser.add_argument("--shard2-grpc-port", type=int, default=58082, help="gRPC port of second shard")

    args = parser.parse_args()

    # Generate topology
    topology = generate_even_split_topology(
        args.model, args.shard1_ip, args.shard2_ip,
        args.shard1_name, args.shard2_name,
        args.shard1_port, args.shard2_port,
        args.shard1_grpc_port, args.shard2_grpc_port
    )

    # Output JSON (can be piped to curl or saved to file)
    print("\n=== MANUAL TOPOLOGY JSON ===")
    print("# Use with: curl -X POST http://localhost:8080/v1/prepare_topology_manual \\")
    print("#           -H 'Content-Type: application/json' -d @- <<< '")
    print(json.dumps(topology, indent=2))
    print("'")

    # Also save to file for convenience
    output_file = f"manual_topology_{args.model.split('/')[-1].replace('-', '_')}.json"
    with open(output_file, 'w') as f:
        json.dump(topology, f, indent=2)
    print(f"\nSaved to: {output_file}")

    # Validation
    assignments = topology["assignments"]
    shard1_layers = assignments[0]["layers"][0]
    shard2_layers = assignments[1]["layers"][0]

    total_layers = len(shard1_layers) + len(shard2_layers)
    expected_layers = topology["num_layers"]

    if total_layers == expected_layers:
        print(f"✅ Validation: {total_layers} layers assigned correctly")
    else:
        print(f"❌ Validation: {total_layers} assigned vs {expected_layers} expected")
        sys.exit(1)

    if max(shard1_layers) < min(shard2_layers):
        print("✅ Validation: No layer overlap between shards")
    else:
        print("❌ Validation: Layer overlap detected!")
        sys.exit(1)


if __name__ == "__main__":
    main()
