#!/usr/bin/env python3
"""Debug config loading to see if DNET_TRANSPORT_COMPRESS is being read."""

import os
from src.dnet.config import get_settings

def main():
    print("=== Environment Variables ===")
    for key, value in os.environ.items():
        if 'COMPRESS' in key:
            print(f"{key}={value}")

    print("\n=== .env file contents ===")
    try:
        with open('.env', 'r') as f:
            for line in f:
                if 'COMPRESS' in line:
                    print(line.strip())
    except FileNotFoundError:
        print("No .env file found")

    print("\n=== Settings from get_settings() ===")
    settings = get_settings()
    print(f"transport.compress: {settings.transport.compress}")
    print(f"transport.wire_dtype: {settings.transport.wire_dtype}")

if __name__ == "__main__":
    main()
