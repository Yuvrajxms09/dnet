"""CLI entry point for dnet ring API server."""

import asyncio
import signal
from argparse import ArgumentParser

from dnet.utils.logger import logger
from dnet.ring.api import RingApiNode


async def serve(http_port: int, grpc_port: int) -> None:
    """Serve the API node.

    Args:
        http_port: HTTP server port
        grpc_port: gRPC callback port
    """
    api_node = RingApiNode(http_port=http_port, grpc_port=grpc_port)

    # Handle shutdown signals gracefully
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler(*_: object) -> None:
        logger.warning("Received termination signal. Stopping services.")
        stop_event.set()

    loop.add_signal_handler(signal.SIGINT, _signal_handler)
    loop.add_signal_handler(signal.SIGTERM, _signal_handler)

    await api_node.start(shutdown_trigger=stop_event.wait)
    await stop_event.wait()
    await api_node.shutdown()


def main() -> None:
    """Run dnet ring API server.

    The API server runs without a preloaded model. Use the following endpoints:
    - POST /v1/prepare_topology - Discover devices and solve for layer distribution
    - POST /v1/load_model - Load model on shards with prepared topology
    - POST /v1/chat/completions - Generate chat completions
    """
    ap = ArgumentParser(description="dnet ring API server")
    ap.add_argument(
        "-p",
        "--http-port",
        type=int,
        required=True,
        help="HTTP server port",
    )
    ap.add_argument(
        "-g",
        "--grpc-port",
        type=int,
        required=True,
        help="gRPC callback port",
    )
    args = ap.parse_args()

    logger.info(f"Starting API server on HTTP port {args.http_port}")
    asyncio.run(serve(args.http_port, args.grpc_port))


if __name__ == "__main__":
    main()
