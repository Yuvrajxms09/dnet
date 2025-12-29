#!/usr/bin/env python3
"""Standalone MCP server entrypoint for Smithery.ai compatibility."""

import asyncio
import logging
import os
import sys
from typing import Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for MLX availability
try:
    import mlx
    HAS_MLX = True
    logger.info("MLX is available")
except ImportError:
    HAS_MLX = False
    logger.warning("MLX not available - MCP server will run in limited mode")

# Import core components
from dnet.api.cluster import ClusterManager
from dnet.api.inference import InferenceManager
from dnet.api.model_manager import ModelManager
from dnet.api.mcp_handler import create_mcp_server
from dnet.utils.logger import logger as dnet_logger


class MCPError(Exception):
    """Custom MCP error for startup issues."""
    pass


def create_managers() -> tuple[InferenceManager, ModelManager, ClusterManager]:
    """Create the required managers for MCP server.

    In environments without MLX, we create mock managers that return appropriate errors.
    """
    if not HAS_MLX:
        logger.warning("Creating mock managers for MLX-free environment")

        # Create mock managers that gracefully handle missing MLX
        class MockInferenceManager:
            async def chat_completions(self, request):
                raise MCPError(
                    -32000,
                    "MLX not available in this environment. This MCP server requires Apple Silicon hardware."
                )

        class MockModelManager:
            def __init__(self):
                self.current_model_id = None
                self.available_models = []

        class MockClusterManager:
            def __init__(self):
                self.current_topology = None
                self.shards = {}

        return MockInferenceManager(), MockModelManager(), MockClusterManager()

    # Full implementation for Apple Silicon environments
    try:
        from dnet.api.catalog import get_available_models
        from dnet_p2p import DnetDeviceProperties

        # Create managers with proper initialization
        model_manager = ModelManager(get_available_models())

        # Create a dummy node_id for MCP server
        node_id = "mcp-server"

        # Initialize cluster manager
        cluster_manager = ClusterManager(node_id=node_id)

        # Initialize inference manager
        inference_manager = InferenceManager(
            model_manager=model_manager,
            cluster_manager=cluster_manager,
            node_id=node_id,
        )

        return inference_manager, model_manager, cluster_manager

    except Exception as e:
        logger.error(f"Failed to create managers: {e}")
        raise MCPError(-32000, f"Failed to initialize dnet components: {str(e)}")


def main():
    """Main entrypoint for the standalone MCP server."""
    try:
        logger.info("Starting dnet MCP server...")

        # Create managers
        inference_manager, model_manager, cluster_manager = create_managers()

        # Create MCP server
        mcp = create_mcp_server(inference_manager, model_manager, cluster_manager)

        logger.info("MCP server created successfully")
        logger.info(f"MLX available: {HAS_MLX}")

        # Run the MCP server
        mcp.run()

    except MCPError as e:
        logger.error(f"MCP Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
